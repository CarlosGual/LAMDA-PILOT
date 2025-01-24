#!/usr/bin/env python3

# Code from Habitat Lab:
# https://github.com/facebookresearch/habitat-lab/blob/main/habitat-baselines/habitat_baselines/common/tensorboard_utils.py

from typing import Any, Optional
import numpy as np

try:
    import wandb
except ImportError:
    ImportError("Please install wandb to use this writer")


def get_writer(args, resume_run_id: Optional[str] = None) -> Any:
    return WeightsAndBiasesWriter(args, resume_run_id)


class WeightsAndBiasesWriter:
    def __init__(
        self,
        args,
        resume_run_id: Optional[str] = None,
    ):
        r"""
        Integrates with https://wandb.ai logging service.
        """
        wb_kwargs = {}
        if args['project_name'] != "":
            wb_kwargs["project"] = args['project_name']
        if args['run_name'] != "":
            wb_kwargs["name"] = args['run_name']
        if args['entity'] != "":
            wb_kwargs["entity"] = args['entity']
        if wandb is None:
            raise ValueError(
                "Requested to log with wandb, but wandb is not installed"
            )
        if resume_run_id is not None:
            wb_kwargs["id"] = resume_run_id
            wb_kwargs["resume"] = "must"

        self.run = wandb.init(  # type: ignore[attr-defined]
            config=args,
            **wb_kwargs,
        )

    def __getattr__(self, item):
        if self.writer:
            return self.writer.__getattribute__(item)
        else:
            return lambda *args, **kwargs: None

    @staticmethod
    def add_scalars(data_dict, step_id: int = None, log_group: str = None):
        if log_group is not None:
            log_data_dict = {
                f"{log_group}/{k.replace(' ', '')}": v
                for k, v in data_dict.items()
            }
        else:
            log_data_dict = data_dict
        wandb.log(log_data_dict, step=step_id)  # type: ignore[attr-defined]

    @staticmethod
    def add_scalar(key, value, step_id: int = None):
        wandb.log({key: value}, step=step_id)  # type: ignore[attr-defined]

    @staticmethod
    def define_metric(name: str, step_metric: str = None, summary: str = "mean"):
        wandb.define_metric(name, step_metric=step_metric, summary=summary)  # type: ignore[attr-defined]

    def __enter__(self):
        return self

    def get_run_id(self) -> Optional[str]:
        return self.run.id

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.run:
            self.run.finish()

    def finish(self):
        if self.run:
            self.run.finish()
