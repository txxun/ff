import copy
import time

import pytorch_lightning as pl
import argparse
from lite_model import LiteModel
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint
from utils import command_line_parser, load_configs, get_project_root, safe_make_dir, save_yaml, convert_edict2dict


class LiteProgressBar(TQDMProgressBar):
    def init_train_tqdm(self):
        bar = super(LiteProgressBar, self).init_train_tqdm()
        return bar

    def on_train_epoch_start(self, *args, **kwargs):
        self.main_progress_bar = self.init_train_tqdm()
        super(LiteProgressBar, self).on_train_epoch_start(*args, **kwargs)

    def on_train_epoch_end(self, *args, **kwargs):
        super(LiteProgressBar, self).on_train_epoch_end(*args, **kwargs)
        self.main_progress_bar.close()

    def get_metrics(self, *args, **kwargs):
        items = super().get_metrics(*args, **kwargs)
        items.pop("v_num", None)
        return items

def init_exp(cfg):
    time_str = time.strftime("%Y%m%d-%H%M")

    exp_dir = get_project_root() + "/exp/" + cfg.exp_id + "/" + time_str
    safe_make_dir(exp_dir)
    cfg['exp_dir'] = exp_dir
    f_config = exp_dir + "/config.yaml"
    save_yaml(convert_edict2dict(copy.deepcopy(cfg)), f_config)
    print(f"save config to {f_config}")


if __name__ == '__main__':
    parser = command_line_parser()
    args = parser.parse_args()
    cfg = load_configs(args.cfg)
    init_exp(cfg)
    model = LiteModel(cfg)
    trainer_params = cfg.get("trainer", {})
    trainer_params.setdefault("weights_save_path", cfg.exp_dir)
    trainer_params.setdefault("default_root_dir", cfg.exp_dir)
    trainer = pl.Trainer(**trainer_params,
                         callbacks=[LiteProgressBar(), ModelCheckpoint(dirpath=cfg.exp_dir)])

    trainer.fit(model)
