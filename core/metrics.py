import torchmetrics


def get_metric(metric_name):
    metricname_to_func = {
        "L2": torchmetrics.MeanSquaredError(),
        "L1": torchmetrics.MeanAbsoluteError(),
        "PSNR": torchmetrics.PeakSignalNoiseRatio(data_range=1.0),
        "SSIM": torchmetrics.StructuralSimilarityIndexMeasure(),
    }
    return metricname_to_func[metric_name]
