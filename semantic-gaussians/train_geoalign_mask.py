from omegaconf import OmegaConf

from geoalign import train_geoalign


if __name__ == '__main__':
    config = OmegaConf.load('./config/geoalign_base.yaml')
    override_config = OmegaConf.from_cli()
    config = OmegaConf.merge(config, override_config)
    print(OmegaConf.to_yaml(config))
    train_geoalign(config)
