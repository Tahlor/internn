from intern_utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def make_config_consistent_stroke(config):
    config.image_dir = Path(config.image_dir)

    config.coordconv_opts = {"zero_center":config.coordconv_0_center,
                             "method":config.coordconv_method}

    config.data_root = config.data_root_fsl if is_fsl() else config.data_root_local

    # if config.x_relative_positions not in (True, False):
    #     raise NotImplemented
    if config.TESTING:
        if "icdar" not in config.dataset_folder.lower():
            config.dataset_folder = "online_coordinate_data/8_stroke_vSmall_16"
        config.update_freq = 1
        config.save_freq = 1
        config.first_loss_epochs = 1 # switch to other loss fast
        config.test_nn_loss_freq = 3
        config.train_size = 35
        config.test_size = 35

        if not config.gpu_if_available:
            config.batch_size = 2
            config.train_size = 2
            config.test_size = 2

    if is_dalai():
        config.dataset_folder = "online_coordinate_data/3_stroke_vverysmallFull"
        config.load_path = False

    ## Process loss functions
    config.all_losses = set()
    # for key in [k for k in config.keys() if "loss_fns" in k]:
    #     for i, loss in enumerate(config[key]):
    #         loss, coef = loss.lower().split(",")
    #
    #         # Don't use inconsistent losses
    #         if "interpolated" in config.interpolated_sos and loss=="ssl":
    #             warnings.warn("Ignoring SSL, since 'interpolated' start point method doesn't use it")
    #             del config[key][i]
    #         else:
    #             config[key][i] = (loss, float(coef))
    #             config.all_losses.add(loss)
    validate_and_prep_loss(config)

    # Update dataset params
    config.dataset.gt_format = config.gt_format
    config.dataset.batch_size = config.batch_size
    config.dataset.image_prep = config.dataset.image_prep.lower()

    # Include synthetic data
    if config.dataset.include_synthetic:
        if config.TESTING:
            config.dataset.extra_dataset = ["online_coordinate_data/MAX_stroke_vFullSynthetic100kFull/train_online_coords_sample.json"]
        else:
            config.dataset.extra_dataset = ["online_coordinate_data/MAX_stroke_vFullSynthetic100kFull/train_online_coords.json",
                                            "online_coordinate_data/MAX_stroke_vBoosted2_normal/train_online_coords.json",
                                            "online_coordinate_data/MAX_stroke_vBoosted2_random/train_online_coords.json",
                                            ]

    else:
        config.dataset.extra_dataset = []

    if "loaded" in config.dataset.image_prep:
        config.dataset.img_height = 60

    config.device = "cuda" if torch.cuda.is_available() and config.gpu_if_available else "cpu"
    return config
