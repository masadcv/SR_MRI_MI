import torch

import utils


def evaluate(args, model, test_loader, loss_content, epoch):
    metrics_meter = {}
    if "val_metrics" in args.keys():
        for vm in args["val_metrics"].keys():
            metrics_meter[vm] = utils.AverageMeter()

    losses = utils.AverageMeter()
    model.eval()
    with torch.no_grad():
        for data_batch in test_loader:
            data_lowres, data_hires = data_batch["lowres"], data_batch["hires"]

            data_lowres, data_hires = data_lowres.to(args["device"]), data_hires.to(
                args["device"]
            )

            output = model(data_lowres)
            test_loss = loss_content(output, data_hires)

            losses.update(test_loss.item(), data_lowres.size(0))

            for vm in args["val_metrics"].keys():
                metrics_meter[vm].update(args["val_metrics"][vm](output, data_hires))

    accuracy_str = ""
    for vm in args["val_metrics"].keys():
        accuracy_str += "{}: {:.4f}  ".format(vm, metrics_meter[vm].avg)

    print(
        "\n{} set: Average loss: {:.4f}\nAccuracy -> {:s}\n".format(
            args["epoch_name"],
            losses.avg,
            accuracy_str,
        )
    )

    if "tb_writer" in args.keys():
        args["tb_writer"].add_scalar(
            "test_%s_loss" % args["epoch_name"], losses.avg, epoch
        )
        for vm in args["val_metrics"].keys():
            args["tb_writer"].add_scalar(
                "test_{}_acc_{}".format(args["epoch_name"], vm),
                metrics_meter[vm].avg,
                epoch,
            )
        utils.log_images_tensorboard(args["tb_writer"], output, data_hires, epoch=epoch)

    accuracy = {}
    for vm in args["val_metrics"].keys():
        accuracy[vm] = metrics_meter[vm].avg

    return accuracy, losses.avg


def train(args, model, train_loader, optimizer, loss_content, epoch):
    losses = utils.AverageMeter()
    model.train()
    for batch_idx, data_batch in enumerate(train_loader):
        data_lowres, data_hires = data_batch["lowres"], data_batch["hires"]

        data_lowres, data_hires = data_lowres.to(args["device"]), data_hires.to(
            args["device"]
        )

        optimizer.zero_grad()
        output = model(data_lowres)
        loss = loss_content(output, data_hires)
        loss.backward()
        losses.update(loss.item(), data_lowres.size(0))
        optimizer.step()

        if batch_idx % args["log_interval"] == 0 and batch_idx != 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * train_loader.batch_size,
                    len(train_loader) * train_loader.batch_size,
                    100.0 * batch_idx / len(train_loader),
                    losses.avg,
                )
            )

    if "tb_writer" in args.keys():
        args["tb_writer"].add_scalar(
            "train_%s_loss" % args["epoch_name"], losses.avg, epoch
        )
        utils.log_images_tensorboard(args["tb_writer"], output, data_hires, epoch=epoch)
