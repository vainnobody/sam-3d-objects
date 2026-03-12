# GeoAlign-Lift-Lite

Lightweight single-view Gaussian segmentation on top of `semantic-gaussians`.

Entry points:
- `python prepare_geoalign_cache.py ...`
- `python train_geoalign_mask.py ...`
- `python infer_geoalign_mask.py ...`

Expected workflow:
1. Train or load a 3DGS scene under `model.model_dir`.
2. Prepare SAM masks under `geoalign.cache.mask_root`.
3. Optionally export SAM3D priors as per-Gaussian `.npz` files.
4. Run cache preparation, then training, then inference.
