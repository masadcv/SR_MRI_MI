import argparse
import glob
import os

import torch

import core.function
import core.loss
import core.metrics
import core.optimisers
import dataset.slicedataset
import models.model_zoo
import utils


def parse_args():
    parser = argparse.ArgumentParser("Basic Neural Network Trainer")
    parser.add_argument(
        "--folder",
        type=str,
        default="./experiments/idn/ODSI",
        help="path to training folder",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    config_file = glob.glob(os.path.join(args.folder, "*.json"))

    # load config
    exp_config = utils.load_json(config_file[0])

    # run on cuda or cpu?
    use_cuda = exp_config["training"]["use_cuda"] and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # dataloaders
    dataloader_config = exp_config["dataloader"]

    test_loader = dataset.slicedataset.create_data_loader(
        path_to_data=dataloader_config["path_to_data"],
        challenge=dataloader_config["challenge"],
        subtask=dataloader_config["subtask"],
        data_transform=None,
        splitset='val',
        scale=dataloader_config["scale"],
        batch_size=1,
        num_workers=4,
        sample_rate=None,
    )

    # create working directories
    work_dir = os.path.join(args.folder, "eval")
    os.makedirs(work_dir, exist_ok=True)

    # create the tensorboard folder from date/time to log training params
    loss_config = exp_config["loss"]

    model_to_func = models.model_zoo.modelname_to_func

    # check if model exists in the zoo
    model_list = list(model_to_func.keys())
    model_name = exp_config["model"]
    assert (
        model_name in model_list
    ), "Invalid model provided - can only be from [%s]" % (model_list)

    model = model_to_func[model_name](scale=dataloader_config["scale"]).to(device)
    model.load_state_dict(torch.load(os.path.join(args.folder, "learned_model.pt")))

    # setup loss
    loss_config = exp_config["loss"]
    loss_content = core.loss.get_loss(loss_config["content_loss"])
    loss_aux = core.loss.get_loss(loss_config["auxilary_loss"])

    # configure and run the training loop
    train_config = exp_config["training"]
    args = {}
    args["w_aux"] = loss_config["auxilary_weight"]
    args["log_interval"] = train_config["log_interval"]
    args["device"] = device

    metrics_dict = {}
    for vm in train_config["val_metrics"]:
        metrics_dict[vm] = core.metrics.get_metric(vm).to(device)

    args["val_metrics"] = metrics_dict

    best_loss_val = 1e10
    args["epoch_name"] = ""
    accuracy, loss_val = core.function.evaluate(
        args, model, test_loader, loss_content, 0
    )


if __name__ == "__main__":
    main()
