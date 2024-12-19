import argparse
import datetime
import os

import tensorboardX
import torch

import core.function
import core.loss
import core.metrics
import core.optimisers
import core.schedulers
import dataset.slicedataset
import models.model_zoo
import utils


def parse_args():
    parser = argparse.ArgumentParser("Basic Neural Network Trainer")
    parser.add_argument("--model", type=str, default="idn", help="model to learn")
    parser.add_argument("--dataset_path", type=str, default=None, help="path to dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.json",
        help="training config parameters",
        required=False,
    )
    parser.add_argument(
        "--content_loss",
        type=str,
        default=None,
        help="loss function override",
        required=False
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=None, 
        help="SR scale",
        required=False
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # load config
    exp_config = utils.load_json(args.config)
    # add model to config
    exp_config["model"] = args.model
    
    # override content loss
    if args.content_loss is not None:
        exp_config["loss"]["content_loss"] = args.content_loss
    
    # override scale
    if args.scale is not None:
        exp_config["dataloader"]["scale"] = args.scale

    # override dataset path
    if args.dataset_path is not None:
        exp_config["dataloader"]["path_to_data"] = args.dataset_path

    # run on cuda or cpu?
    use_cuda = exp_config["training"]["use_cuda"] and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # dataloaders
    dataloader_config = exp_config["dataloader"]
    train_loader = dataset.slicedataset.create_data_loader(
        path_to_data=dataloader_config["path_to_data"],
        challenge=dataloader_config["challenge"],
        subtask=dataloader_config["subtask"],
        data_transform=None,
        splitset='train',
        scale=dataloader_config["scale"],
        batch_size=1,
        num_workers=4,
        sample_rate=1,
    )
    val_loader = dataset.slicedataset.create_data_loader(
        path_to_data=dataloader_config["path_to_data"],
        challenge=dataloader_config["challenge"],
        subtask=dataloader_config["subtask"],
        data_transform=None,
        splitset='val',
        scale=dataloader_config["scale"],
        batch_size=1,
        num_workers=4,
        sample_rate=1,
    )


    # create working directories
    work_dir = os.path.join(
        exp_config["work_dir"], args.model, dataloader_config["dataset"]
    )
    os.makedirs(work_dir, exist_ok=True)
    
    # create the tensorboard folder from date/time to log training params
    loss_config = exp_config["loss"]
    current_date = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M")
    exp_details = "{}_{}".format(args.model, loss_config["content_loss"])
    tensorboard_folder = os.path.join(work_dir, current_date + "_" + exp_details)
    tb_writer = tensorboardX.SummaryWriter(log_dir=tensorboard_folder)

    # save config
    utils.save_json(
        exp_config, os.path.join(tensorboard_folder, os.path.basename(args.config))
    )

    model_to_func = models.model_zoo.modelname_to_func

    # check if model exists in the zoo
    model_list = list(model_to_func.keys())
    assert (
        args.model in model_list
    ), "Invalid model provided - can only be from [%s]" % (model_list)

    model = model_to_func[args.model](scale=dataloader_config["scale"]).to(device)

    # setup optimiser
    optim_config = exp_config["optimiser"]
    optimiser = core.optimisers.get_optimiser(
        optim_config["type"], params=model.parameters(), lr=optim_config["lr"]
    )

    # setup scheduler
    scheduler = core.schedulers.get_scheduler(
        type=optim_config["scheduler_type"],
        optim=optimiser,
        lr_step=optim_config["scheduler_lr_step"],
        lr_factor=optim_config["scheduler_lr_factor"],
        epochs=exp_config["training"]["epochs"],
    )

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
    args["tb_writer"] = tb_writer

    metrics_dict = {}
    for vm in train_config["val_metrics"]:
        metrics_dict[vm] = core.metrics.get_metric(vm).to(device)

    args["val_metrics"] = metrics_dict

    best_loss_val = 1e10
    for epoch in range(0, train_config["epochs"]):
        print("-" * 50)
        args["epoch_name"] = "train"

        core.function.train(args, model, train_loader, optimiser, loss_content, epoch)

        scheduler.step()

        print("-" * 50)

        if epoch % train_config["eval_interval"] == 0:
            args["epoch_name"] = "val"
            accuracy, loss_val = core.function.evaluate(
                args, model, val_loader, loss_content, epoch
            )

            if loss_val < best_loss_val:
                best_loss_val = loss_val
                print("Found best validation loss {}, saving model...".format(loss_val))
                torch.save(
                    model.state_dict(),
                    os.path.join(tensorboard_folder, "learned_model.pt"),
                )

            for vm in args["val_metrics"].keys():
                args["val_metrics"][vm].reset()


if __name__ == "__main__":
    main()
