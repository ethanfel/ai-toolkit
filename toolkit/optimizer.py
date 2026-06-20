import torch


def get_optimizer(
        params,
        optimizer_type='adam',
        learning_rate=1e-6,
        optimizer_params=None
):
    if optimizer_params is None:
        optimizer_params = {}
    lower_type = optimizer_type.lower()
    if lower_type.startswith("dadaptation"):
        # dadaptation optimizer does not use standard learning rate. 1 is the default value
        import dadaptation
        print("Using DAdaptAdam optimizer")
        use_lr = learning_rate
        if use_lr < 0.1:
            # dadaptation uses different lr that is values of 0.1 to 1.0. default to 1.0
            use_lr = 1.0
        if lower_type.endswith('lion'):
            optimizer = dadaptation.DAdaptLion(params, eps=1e-6, lr=use_lr, **optimizer_params)
        elif lower_type.endswith('adam'):
            optimizer = dadaptation.DAdaptLion(params, eps=1e-6, lr=use_lr, **optimizer_params)
        elif lower_type == 'dadaptation':
            # backwards compatibility
            optimizer = dadaptation.DAdaptAdam(params, eps=1e-6, lr=use_lr, **optimizer_params)
            # warn user that dadaptation is deprecated
            print("WARNING: Dadaptation optimizer type has been changed to DadaptationAdam. Please update your config.")
    elif lower_type.startswith("prodigy8bit"):
        from toolkit.optimizers.prodigy_8bit import Prodigy8bit
        print("Using Prodigy optimizer")
        use_lr = learning_rate
        if use_lr < 0.1:
            # dadaptation uses different lr that is values of 0.1 to 1.0. default to 1.0
            use_lr = 1.0

        print(f"Using lr {use_lr}")
        # let net be the neural network you want to train
        # you can choose weight decay value based on your problem, 0 by default
        optimizer = Prodigy8bit(params, lr=use_lr, eps=1e-6, **optimizer_params)
    elif lower_type in ["prodigy_plus", "prodigyplus", "prodigy_plus_schedulefree",
                        "prodigyplusschedulefree", "prodigy+"]:
        from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree

        print("Using Prodigy+ Schedule-Free optimizer")
        use_lr = learning_rate
        if use_lr < 0.1:
            # prodigy uses a relative lr; 1.0 is the expected value
            use_lr = 1.0
        print(f"Using lr {use_lr}")

        # validated defaults (from the HunyuanVideo-Foley recipe); every one of
        # these is overridable via optimizer_params in the job config.
        pp_defaults = {
            'betas': [0.9, 0.999],
            'weight_decay': 0.01,
            'd_coef': 1.0,
        }
        for k, v in pp_defaults.items():
            optimizer_params.setdefault(k, v)

        optimizer = ProdigyPlusScheduleFree(params, lr=use_lr, **optimizer_params)
        # Schedule-free optimizers must start in train mode. The trainer switches
        # to eval mode (averaged weights) around sampling/saving and back to train
        # to continue. See toolkit.optimizer.optimizer_requires_eval_mode.
        optimizer.train()
    elif lower_type.startswith("prodigy"):
        from prodigyopt import Prodigy

        print("Using Prodigy optimizer")
        use_lr = learning_rate
        if use_lr < 0.1:
            # dadaptation uses different lr that is values of 0.1 to 1.0. default to 1.0
            use_lr = 1.0

        print(f"Using lr {use_lr}")
        # let net be the neural network you want to train
        # you can choose weight decay value based on your problem, 0 by default
        optimizer = Prodigy(params, lr=use_lr, eps=1e-6, **optimizer_params)
    elif lower_type == "adam8":
        from toolkit.optimizers.adam8bit import Adam8bit

        optimizer = Adam8bit(params, lr=learning_rate, eps=1e-6, **optimizer_params)
    elif lower_type == "adamw8":
        from toolkit.optimizers.adam8bit import Adam8bit

        optimizer = Adam8bit(params, lr=learning_rate, eps=1e-6, decouple=True, **optimizer_params)
    elif lower_type.endswith("8bit"):
        import bitsandbytes

        if lower_type == "adam8bit":
            return bitsandbytes.optim.Adam8bit(params, lr=learning_rate, eps=1e-6, **optimizer_params)
        if lower_type == "ademamix8bit":
            return bitsandbytes.optim.AdEMAMix8bit(params, lr=learning_rate, eps=1e-6, **optimizer_params)
        elif lower_type == "adamw8bit":
            return bitsandbytes.optim.AdamW8bit(params, lr=learning_rate, eps=1e-6, **optimizer_params)
        elif lower_type == "lion8bit":
            return bitsandbytes.optim.Lion8bit(params, lr=learning_rate, **optimizer_params)
        else:
            raise ValueError(f'Unknown optimizer type {optimizer_type}')
    elif lower_type == 'adam':
        optimizer = torch.optim.Adam(params, lr=float(learning_rate), eps=1e-6, **optimizer_params)
    elif lower_type == 'adamw':
        optimizer = torch.optim.AdamW(params, lr=float(learning_rate), eps=1e-6, **optimizer_params)
    elif lower_type == 'lion':
        try:
            from lion_pytorch import Lion
            return Lion(params, lr=learning_rate, **optimizer_params)
        except ImportError:
            raise ImportError("Please install lion_pytorch to use Lion optimizer -> pip install lion-pytorch")
    elif lower_type == 'adagrad':
        optimizer = torch.optim.Adagrad(params, lr=float(learning_rate), **optimizer_params)
    elif lower_type == 'adafactor':
        from toolkit.optimizers.adafactor import Adafactor
        if 'relative_step' not in optimizer_params:
            optimizer_params['relative_step'] = False
        if 'scale_parameter' not in optimizer_params:
            optimizer_params['scale_parameter'] = False
        if 'warmup_init' not in optimizer_params:
            optimizer_params['warmup_init'] = False
        optimizer = Adafactor(params, lr=float(learning_rate), **optimizer_params)
    elif lower_type == 'automagic':
        from toolkit.optimizers.automagic import Automagic
        optimizer = Automagic(params, lr=float(learning_rate), **optimizer_params)
    elif lower_type == 'automagic2':
        from toolkit.optimizers.automagic2 import Automagic2
        optimizer = Automagic2(params, lr=float(learning_rate), **optimizer_params)
    elif lower_type == 'automagic3':
        from toolkit.optimizers.automagic3 import Automagic3
        optimizer = Automagic3(params, lr=float(learning_rate), **optimizer_params)
    else:
        raise ValueError(f'Unknown optimizer type {optimizer_type}')
    return optimizer


def optimizer_requires_eval_mode(optimizer_type: str) -> bool:
    """Whether an optimizer needs train()/eval() mode toggling.

    Schedule-free optimizers (e.g. Prodigy+ Schedule-Free) hold the raw
    train-mode weights during optimization and only materialize the averaged
    weights when ``optimizer.eval()`` is called. The trainer must switch to eval
    mode before sampling or saving (so artifacts match inference) and back to
    train mode afterwards to continue training.
    """
    if optimizer_type is None:
        return False
    return optimizer_type.lower() in [
        "prodigy_plus", "prodigyplus", "prodigy_plus_schedulefree",
        "prodigyplusschedulefree", "prodigy+",
    ]
