from ultralytics.yolo.engine.trainer import BaseTrainer
from ultralytics.yolo.engine.validator import BaseValidator
from ultralytics.yolo.utils.torch_utils import get_flops, get_num_params

try:
    import wandb

    assert hasattr(wandb, "__version__")
except (ImportError, AssertionError):
    wandb = None


def on_pretrain_routine_start(trainer: BaseTrainer):
    # initialise classes in the config arguments
    trainer.args.labels = trainer.data["names"]
    # Init the run
    wandb.init(
        project=trainer.args.project or "YOLOv8",
        name=trainer.args.name,
        config=dict(trainer.args),
    ) if not wandb.run else wandb.run


def on_pretrain_routine_end(trainer: BaseTrainer):
    paths = trainer.save_dir.glob("*labels*.jpg")
    wandb.run.log({"Labels": [wandb.Image(str(x), caption=x.name) for x in paths]})


# def on_train_epoch_start(trainer: BaseTrainer):
#     # We emit the epoch number here to force wandb to commit the previous step when the new one starts,
#     # reducing the delay between the end of the epoch and metrics for it appearing.
#     wandb.run.log(
#         {"epoch": trainer.epoch + 1},
#         step=trainer.epoch + 1,
#     )


def on_train_epoch_end(trainer: BaseTrainer):
    # loss of the training dataset
    wandb.run.log(
        trainer.label_loss_items(trainer.tloss, prefix="train"), step=trainer.epoch + 1
    )
    # loss of the validation dataset
    vloss = trainer.validator.loss if hasattr(trainer.validator, "loss") else []
    wandb.run.log(trainer.label_loss_items(vloss, prefix="val"), step=trainer.epoch + 1)


def on_fit_epoch_end(trainer: BaseTrainer):
    wandb.run.log(trainer.metrics, step=trainer.epoch + 1)
    # wandb.run.log(trainer.validator.metrics.results_dict, step=trainer.epoch + 1)


def on_train_end(trainer: BaseTrainer):
    art = wandb.Artifact(type="model", name=f"run_{wandb.run.id}_model")
    if trainer.best.exists():
        art.add_file(trainer.best)
        wandb.run.log_artifact(art)
    files = [
        "results.png",
        "confusion_matrix.png",
        *(f"{x}_curve.png" for x in ("F1", "PR", "P", "R")),
    ]
    files = [
        (trainer.save_dir / f) for f in files if (trainer.save_dir / f).exists()
    ]  # filter
    wandb.run.log({"Results": [wandb.Image(str(f), caption=f.name) for f in files]})


def on_params_update(trainer: BaseTrainer, params: dict):
    wandb.run.config.update(params, allow_val_change=True)


def teardown(_trainer: BaseTrainer):
    wandb.finish()


callbacks = (
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_pretrain_routine_end": on_pretrain_routine_end,
        "on_train_epoch_end": on_train_epoch_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_end": on_train_end,
        "teardown": teardown,
    }
    if wandb
    else {}
)
from ultralytics import YOLO

model = YOLO("yolov8x.pt")
for event, func in callbacks.items():
    model.add_callback(event, func)
model.train(data="/home/initial/workspace/kikaichino/data.yaml", epochs=3)
