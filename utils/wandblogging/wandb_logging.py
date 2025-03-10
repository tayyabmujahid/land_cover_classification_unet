import os

import wandb


class Wandblogger:
    def __init__(self, name, job_type='Training', config=None):
        self.run = wandb.init(job_type=job_type,
                              entity="fsdl2022_project051",
                              project='land_cover_segmentation',
                              name=name, config=config)
        self.job_type = job_type
        self.log_dict = {}

    def log(self, log_dict):
        # if self.wandb_current_run:
        for key, value in log_dict.items():
            self.log_dict[key] = value

    def log_artifact(self, filename, model_path, metric_val):
        artifact = wandb.Artifact(filename, type='model', metadata={'Validation score': metric_val})
        artifact.add_file(model_path)
        self.run.log_artifact(artifact)

    def end_epoch(self):
        wandb.log(self.log_dict)
        self.log_dict = {}

    def end_batch(self):
        wandb.log(self.log_dict)
        self.log_dict = {}


def generate_run_name():
    from datetime import datetime
    dt = datetime.now()
    dt_str = dt.strftime('%d%b%y_%H%M%S')
    return dt_str


            