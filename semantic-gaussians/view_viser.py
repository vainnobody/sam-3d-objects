import os
import cv2
import time
import torch
import viser
from copy import deepcopy
from viser.transforms import SE3, SO3
import numpy as np
from omegaconf import OmegaConf

from model import GaussianModel, render, render_chn
from model.openseg_predictor import OpenSeg
from model.lseg_predictor import LSeg
from model.render_utils import render_palette
from scene import Scene
from utils.system_utils import searchForMaxIteration, set_seed
from utils.camera_utils import get_camera_viser
from utils.sh_utils import RGB2SH, SH2RGB
from dataset.scannet.scannet_constants import COLORMAP


def to_hex(color):
    return "{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


def get_text(vocabulary, prefix_prompt=""):
    texts = [prefix_prompt + x.lower().replace("_", " ") for x in vocabulary]
    return texts


def restore_tensor_(target, source):
    if source.device == target.device:
        target.copy_(source)
    else:
        target.copy_(source.to(device=target.device, dtype=target.dtype))


def main(config):
    with torch.no_grad():
        device = torch.device(config.model.device)
        feature_dtype = torch.float16 if str(config.render.feature_dtype).lower() == "fp16" else torch.float32
        semantic_downsample = max(1, int(config.render.semantic_resolution_divisor))
        precompute_semantic = bool(config.render.precompute_semantic_on_start)
        release_semantic_cache = bool(config.render.release_semantic_cache_when_inactive)

        # Load 3D Gaussians
        scene_config = deepcopy(config)
        if config.model.dynamic:
            scene_config.scene.scene_path = os.path.join(config.scene.scene_path, "0")
        scene = Scene(scene_config.scene)
        gaussians = GaussianModel(config.model.sh_degree)

        if config.model.model_dir:
            if config.model.dynamic:
                gaussians.load_dynamic_npz(os.path.join(config.model.model_dir, "params.npz"), 0, requires_grad=False)
            else:
                if config.model.load_iteration == -1:
                    loaded_iter = searchForMaxIteration(os.path.join(config.model.model_dir, "point_cloud"))
                else:
                    loaded_iter = config.model.load_iteration
                print("Loading trained model at iteration {}".format(loaded_iter))
                gaussians.load_ply(
                    os.path.join(
                        config.model.model_dir,
                        "point_cloud",
                        f"iteration_{loaded_iter}",
                        "point_cloud.ply",
                    )
                    ,
                    requires_grad=False,
                )
        else:
            raise NotImplementedError
        
        # Load semantic embeddings
        fusion = torch.load(config.render.fusion_dir, map_location="cpu")
        features_cpu = fusion["feat"].to(dtype=feature_dtype, device="cpu")
        mask_cpu = fusion["mask_full"].to(dtype=torch.bool, device="cpu")
        mask = mask_cpu.to(device=device)

        if config.render.keep_features_on_gpu:
            features_gpu = features_cpu.to(device=device, dtype=feature_dtype)
        else:
            features_gpu = None

        if config.render.keep_edit_backup_on_gpu:
            original_opacity = gaussians._opacity.detach().clone()
            original_scale = gaussians._scaling.detach().clone()
            original_color = gaussians._features_dc.detach().clone()
            original_coord = gaussians._xyz.detach().clone()
        else:
            original_opacity = gaussians._opacity.detach().cpu().clone()
            original_scale = gaussians._scaling.detach().cpu().clone()
            original_color = gaussians._features_dc.detach().cpu().clone()
            original_coord = gaussians._xyz.detach().cpu().clone()

        text_model = None

        def get_text_model():
            nonlocal text_model
            if text_model is None:
                if config.render.model_2d == "lseg":  # 512dim CLIP
                    text_model = LSeg(None)
                else:  # 768dim CLIP
                    text_model = OpenSeg(None, "ViT-L/14@336px")
            return text_model

        def get_feature_bank():
            if features_gpu is not None:
                return features_gpu
            return features_cpu.to(device=device, dtype=feature_dtype)

        def restore_gaussians():
            restore_tensor_(gaussians._opacity, original_opacity)
            restore_tensor_(gaussians._features_dc, original_color)
            restore_tensor_(gaussians._scaling, original_scale)
            restore_tensor_(gaussians._xyz, original_coord)

        if not config.render.lazy_text_model:
            get_text_model()

        # Initialize colormap and camera
        colormap = COLORMAP
        colormap_hex = [to_hex(e) for e in colormap]
        colormap_cuda = torch.tensor(colormap, device=device)

        bg_color = [1, 1, 1] if config.scene.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device=device)
        scene_camera = scene.getTrainCameras()[0]
        width, height = scene_camera.image_width, scene_camera.image_height
        w2c = scene_camera.world_view_transform.cpu().numpy().transpose()
        soft_save = None
        color_save = None
        semantic_cache_dirty = precompute_semantic
        editing_dirty = False
        
        # Initialize viser
        server = viser.ViserServer()
        server.world_axes.visible = False
        need_update = False
        tab_group = server.add_gui_tab_group()

        # Settings tab
        with tab_group.add_tab("Settings", viser.Icon.SETTINGS):
            gui_render_mode = server.add_gui_button_group("Render mode", ("RGB", "Depth", "Semantic", "Relevancy"))
            render_mode = "RGB"
            gui_near_slider = server.add_gui_slider("Depth near", min=0, max=3, step=0.2, initial_value=1.5)
            gui_far_slider = server.add_gui_slider("Depth far", min=6, max=20, step=0.5, initial_value=6)
            gui_scale_slider = server.add_gui_slider("Gaussian scale", min=0.01, max=1, step=0.01, initial_value=1)
            resolution_scale_group = server.add_gui_button_group("Resolution scale", ("0.5x", "1x", "2x", "4x"))
            gui_background_checkbox = server.add_gui_checkbox("Remove background", False)
            gui_up_checkbox = server.add_gui_checkbox("Lock up direction", False)

            gui_prompt_input = server.add_gui_text(
                "Text prompt (divided by comma)",
                "wall,floor,cabinet,bed,chair,sofa,table,door,window,bookshelf,picture,counter,desk,curtain,refrigerator,shower curtain,toilet,sink,bathtub",
                # "wall,floor,sofa,table,television,plant,bookshelf,piano,door,speaker,slippers,bottle",
            )

            gui_prompt_button = server.add_gui_button("Apply text prompt")

        with tab_group.add_tab("Editing", viser.Icon.SETTINGS):
            gui_edit_mode = server.add_gui_button_group("Edit mode", ("Remove", "Color", "Size", "Move"))
            edit_mode = "Remove"
            gui_edit_input = server.add_gui_text("Edit prompt (divided by comma)", "")
            gui_preserve_input = server.add_gui_text("Preserve prompt (divided by comma)", "")
            gui_editing_button = server.add_gui_button("Apply editing prompt")

        # Colormap tab
        with tab_group.add_tab("Colormap", viser.Icon.COLOR_FILTER):
            gui_markdown = server.add_gui_markdown("")

        # Button callbacks
        @gui_render_mode.on_click
        def _(_) -> None:
            nonlocal render_mode
            nonlocal need_update
            render_mode = gui_render_mode.value
            if release_semantic_cache and render_mode in ("RGB", "Depth"):
                clear_semantic_cache()
            need_update = True

        @gui_edit_mode.on_click
        def _(_) -> None:
            nonlocal edit_mode
            nonlocal need_update
            edit_mode = gui_edit_mode.value
            need_update = True

        @gui_prompt_button.on_click
        def _(_) -> None:
            nonlocal semantic_cache_dirty
            nonlocal need_update
            semantic_cache_dirty = True
            clear_semantic_cache()
            need_update = True

        @gui_editing_button.on_click
        def _(_) -> None:
            nonlocal editing_dirty
            nonlocal need_update
            editing_dirty = True
            need_update = True

        @gui_scale_slider.on_update
        def _(_) -> None:
            nonlocal need_update
            need_update = True

        @resolution_scale_group.on_click
        def _(_) -> None:
            nonlocal need_update
            need_update = True

        @gui_background_checkbox.on_update
        def _(_) -> None:
            nonlocal need_update
            need_update = True

        @gui_up_checkbox.on_update
        def _(_) -> None:
            nonlocal need_update
            need_update = True

        @gui_near_slider.on_update
        def _(_) -> None:
            nonlocal need_update
            need_update = True

        @gui_far_slider.on_update
        def _(_) -> None:
            nonlocal need_update
            need_update = True

        @server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            print("new client!")
            nonlocal w2c
            nonlocal need_update
            c2w_transform = SE3.from_matrix(w2c).inverse()
            client.camera.wxyz = c2w_transform.wxyz_xyz[:4]  # np.array([1.0, 0.0, 0.0, 0.0])
            client.camera.position = c2w_transform.wxyz_xyz[4:]
            need_update = True

            # This will run whenever we get a new camera!
            @client.camera.on_update
            def _(_: viser.CameraHandle) -> None:
                nonlocal need_update
                need_update = True

        def clear_semantic_cache():
            nonlocal soft_save, color_save
            soft_save = None
            color_save = None

        def ensure_semantic_cache(target_mode):
            nonlocal soft_save, color_save, semantic_cache_dirty
            need_soft = target_mode == "Semantic"
            need_color = target_mode == "Relevancy"
            if not semantic_cache_dirty:
                if (not need_soft or soft_save is not None) and (not need_color or color_save is not None):
                    return

            text_model_cur = get_text_model()
            labelset = ["other"] + get_text(gui_prompt_input.value.split(","))
            feature_bank = get_feature_bank()
            text_features = text_model_cur.extract_text_feature(labelset).to(dtype=feature_dtype)
            label = torch.einsum("cq,dq->dc", text_features, feature_bank).argmax(dim=1)

            if need_soft:
                label_hard = torch.nn.functional.one_hot(label, num_classes=text_features.shape[0]).to(dtype=feature_dtype)
                soft_save = torch.zeros((mask.shape[0], label_hard.shape[1]), dtype=feature_dtype, device=device)
                soft_save[mask] = label_hard
            if need_color:
                palette_idx = label % len(colormap)
                colors = colormap_cuda[palette_idx] / 255
                color_save = torch.zeros((mask.shape[0], 3), dtype=feature_dtype, device=device)
                color_save[mask] = colors

            repeated_hex = (colormap_hex * ((len(labelset) + len(colormap_hex) - 1) // len(colormap_hex)))[: len(labelset)]
            color_mapping = list(zip(labelset, repeated_hex))
            content_head = "| | |\n|:-:|:-|"
            content_body = "".join(
                [
                    f"\n|![color](https://via.placeholder.com/5x5/{color}/ffffff?text=+)|{label_name}||"
                    for label_name, color in color_mapping
                ]
            )
            gui_markdown.content = content_head + content_body

            if features_gpu is None:
                del feature_bank
            del text_features, label
            semantic_cache_dirty = False
            torch.cuda.empty_cache()

        def apply_editing():
            nonlocal editing_dirty
            restore_gaussians()
            if gui_edit_input.value == "":
                editing_dirty = False
                return

            text_model_cur = get_text_model()
            feature_bank = get_feature_bank()
            edit_terms = gui_edit_input.value.split(",")
            preserve_terms = [x for x in gui_preserve_input.value.split(",") if x != ""]
            len_edit = len(edit_terms)
            edit_features = text_model_cur.extract_text_feature(["other"] + edit_terms + preserve_terms).to(
                dtype=feature_dtype
            )
            sim = torch.einsum("cq,dq->dc", edit_features, feature_bank)
            sim[sim < 0] = -2
            label = sim.argmax(dim=1)

            edit_mask = (label > 0) * (label <= len_edit)
            if edit_mode == "Remove":
                tmp = gaussians._opacity[mask]
                tmp[edit_mask] = -9999
                gaussians._opacity[mask] = tmp
            elif edit_mode == "Color":
                tmp = gaussians._features_dc[mask]
                tmp_rgb = SH2RGB(tmp[edit_mask])
                tmp_rgb = 1 - tmp_rgb
                tmp_rgb = torch.clamp(tmp_rgb, 0, 1)
                tmp[edit_mask] = RGB2SH(tmp_rgb)
                gaussians._features_dc[mask] = tmp
            elif edit_mode == "Size":
                tmp = gaussians._scaling[mask]
                tmp[edit_mask] *= 2
                gaussians._scaling[mask] = tmp
                tmp = gaussians._xyz[mask]
                tmp[edit_mask] *= 2
                gaussians._xyz[mask] = tmp
            elif edit_mode == "Move":
                tmp = gaussians._xyz[mask]
                tmp[edit_mask] += 1
                gaussians._xyz[mask] = tmp

            if features_gpu is None:
                del feature_bank
            del edit_features, sim, label
            editing_dirty = False
            torch.cuda.empty_cache()

        # Main render function. Render if camera moves or settings change.
        if config.model.dynamic:
            start_time = time.time()
            num_timesteps = config.model.num_timesteps
        while True:
            if not server.get_clients():
                time.sleep(0.05)
                continue

            if config.model.dynamic:
                passed_time = time.time() - start_time
                passed_frames = passed_time * config.render.dynamic_fps
                t = int(passed_frames % num_timesteps)
                need_update = True

            if not editing_dirty and not need_update:
                time.sleep(0.01)
                continue

            height = int(float(resolution_scale_group.value[:-1]) * scene_camera.image_height)
            width = int(float(resolution_scale_group.value[:-1]) * scene_camera.image_width)

            if editing_dirty:
                apply_editing()
                need_update = True

            # Render for each client
            for client in server.get_clients().values():
                client_info = client.camera
                w2c_matrix = (
                    SE3.from_rotation_and_translation(SO3(client_info.wxyz), client_info.position).inverse().as_matrix()
                )
                if not gui_up_checkbox.value:
                    client.camera.up_direction = SO3(client.camera.wxyz) @ np.array([0.0, -1.0, 0.0])
                new_camera = get_camera_viser(
                    scene_camera,
                    w2c_matrix[:3, :3].transpose(),
                    w2c_matrix[:3, 3],
                    client_info.fov,
                    client_info.aspect,
                )
                new_camera.cuda()
                if config.model.dynamic:
                    gaussians.load_dynamic_npz(os.path.join(config.model.model_dir, "params.npz"), t, requires_grad=False)
                if render_mode == "RGB":
                    output = render(
                        new_camera,
                        gaussians,
                        config.pipeline,
                        background,
                        scaling_modifier=gui_scale_slider.value,
                        override_shape=(width, height),
                        foreground=gaussians.is_fg if gui_background_checkbox.value else None,
                        track_grad=False,
                    )
                    rendering = output["render"].cpu().numpy().transpose(1, 2, 0)
                elif render_mode == "Depth":
                    output = render(
                        new_camera,
                        gaussians,
                        config.pipeline,
                        background,
                        scaling_modifier=gui_scale_slider.value,
                        override_shape=(width, height),
                        foreground=gaussians.is_fg if gui_background_checkbox.value else None,
                        track_grad=False,
                    )
                    rendering = output["depth"].cpu().numpy().transpose(1, 2, 0)
                    rendering = np.clip(
                        (rendering - gui_near_slider.value) * 255 / (gui_far_slider.value - gui_near_slider.value),
                        0,
                        255,
                    )
                    rendering = cv2.cvtColor(rendering.astype(np.uint8), cv2.COLOR_GRAY2RGB)
                elif render_mode == "Semantic":
                    ensure_semantic_cache("Semantic")
                    render_palette_cuda = colormap_cuda
                    if soft_save.shape[1] > colormap_cuda.shape[0]:
                        repeat = (soft_save.shape[1] + colormap_cuda.shape[0] - 1) // colormap_cuda.shape[0]
                        render_palette_cuda = colormap_cuda.repeat(repeat, 1)[: soft_save.shape[1]]
                    output = render_chn(
                        new_camera,
                        gaussians,
                        config.pipeline,
                        background,
                        scaling_modifier=gui_scale_slider.value,
                        num_channels=soft_save.shape[1],
                        override_color=soft_save,
                        override_shape=(width // semantic_downsample, height // semantic_downsample),
                        foreground=gaussians.is_fg if gui_background_checkbox.value else None,
                        track_grad=False,
                    )
                    sim = output["render"]
                    label = sim.argmax(dim=0).cpu()
                    sem = render_palette(label, render_palette_cuda.reshape(-1))
                    rendering = sem.cpu().numpy().transpose(1, 2, 0)
                else:  # relevancy
                    ensure_semantic_cache("Relevancy")
                    output = render(
                        new_camera,
                        gaussians,
                        config.pipeline,
                        background,
                        scaling_modifier=gui_scale_slider.value,
                        override_color=color_save,
                        override_shape=(width // semantic_downsample, height // semantic_downsample),
                        foreground=gaussians.is_fg if gui_background_checkbox.value else None,
                        track_grad=False,
                    )
                    rendering = output["render"].cpu().numpy().transpose(1, 2, 0)
                client.set_background_image(rendering)
            if release_semantic_cache and render_mode in ("RGB", "Depth"):
                clear_semantic_cache()
            need_update = False


if __name__ == "__main__":
    config = OmegaConf.load("./config/view_scannet.yaml")
    default_render_config = OmegaConf.create(
        {
            "render": {
                "feature_dtype": "fp16",
                "keep_features_on_gpu": False,
                "keep_edit_backup_on_gpu": False,
                "lazy_text_model": True,
                "semantic_resolution_divisor": 2,
                "precompute_semantic_on_start": False,
                "release_semantic_cache_when_inactive": True,
            }
        }
    )
    override_config = OmegaConf.from_cli()
    config = OmegaConf.merge(default_render_config, config, override_config)
    print(OmegaConf.to_yaml(config))

    set_seed(config.pipeline.seed)
    main(config)
