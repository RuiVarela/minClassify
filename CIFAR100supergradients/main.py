import argparse

from super_gradients import init_trainer
from super_gradients.common.object_names import Models
from super_gradients.training import Trainer, models
from super_gradients.training.metrics.classification_metrics import Accuracy, Top5
from super_gradients.training.dataloaders import dataloaders
from super_gradients.training.utils.distributed_training_utils import setup_device

experiment_name = "cifar_100"
checkpoints_folder = "checkpoints"

#model_kind = Models.RESNET18
#model_kind = Models.EFFICIENTNET_B0
model_kind = Models.MOBILENET_V3_SMALL

classes = 100
batch_size = 256

init_trainer()

# setup_device("cpu")
setup_device(num_gpus=-1)

train_loader = dataloaders.cifar100_train(dataset_params={"root": "data"}, dataloader_params={"batch_size": batch_size})
valid_loader = dataloaders.cifar100_val(dataset_params={"root": "data"}, dataloader_params={"batch_size": batch_size})
test_loader = dataloaders.cifar100_val(dataset_params={"root": "data"}, dataloader_params={"batch_size": batch_size})


def train():
    model = models.get(model_kind, num_classes=classes, pretrained_weights="imagenet")

    training_params = {
       
        "max_epochs": 150,

        "lr_mode": "CosineLRScheduler",
        "initial_lr": 0.1,

        "optimizer": "SGD",

        "optimizer_params": {
           "weight_decay": 0.00004
        },

        "loss": "CrossEntropyLoss",
        "train_metrics_list": [Accuracy()],
        "valid_metrics_list": [Accuracy()],
        "metric_to_watch": "Accuracy",
        "greater_metric_to_watch_is_better": True,
    }

    trainer = Trainer(experiment_name=experiment_name, ckpt_root_dir=checkpoints_folder)
    trainer.train(model=model, training_params=training_params, train_loader=train_loader, valid_loader=valid_loader)

def test(checkpoint):
    model = models.get(model_kind, num_classes=classes, checkpoint_path=checkpoint)

    test_metrics = [Accuracy(), Top5()]

    trainer = Trainer(experiment_name=experiment_name, ckpt_root_dir=checkpoints_folder)
    test_results = trainer.test(model=model, test_loader=test_loader, test_metrics_list=test_metrics)
    print(f"Test results: Accuracy: {test_results['Accuracy']}, Top5: {test_results['Top5']}")


def export(checkpoint):
    model = models.get(model_kind, num_classes=classes, checkpoint_path=checkpoint)

    x, y = next(iter(train_loader))
    shape = (1, x.shape[1], x.shape[2], x.shape[2])

    onnx_export_options = {
        "input_names": ['input'],   # the model's input names
        "output_names": ['output'], # the model's output names
        "dynamic_axes": {'input' : {0 : 'batch_size'},    # variable length axes
                         'output' : {0 : 'batch_size'}}
    }

    models.convert_to_onnx(model=model, 
                           out_path="model.onnx",
                           torch_onnx_export_kwargs=onnx_export_options,
                           prep_model_for_conversion_kwargs=dict(input_size=shape))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='classifier', description='classifier')
    parser.add_argument('-t', '--test', help="test a checkpoint")
    parser.add_argument('-e', '--export', help="export checkpoint")

    args = parser.parse_args()

    if args.export:
        export(args.export)
    elif args.test:
        test(args.test)
    else:
        train()