import torch
from thop import profile
import psutil, os
import tracemalloc


def get_param_memory(model, dtype=torch.float32):
    total_params = sum(p.numel() for p in model.parameters())
    bytes_per_param = torch.tensor([], dtype=dtype).element_size()
    memory_MB = total_params * bytes_per_param / (1024 ** 2)
    return memory_MB


def measure_inference_memory_cpu(model, bvp, eda):
    model = model.cpu()
    bvp = bvp.cpu()
    eda = eda.cpu()

    tracemalloc.start()

    with torch.no_grad():
        _ = model(bvp, eda)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # convert to MB
    peak_mb = peak / 1024 / 1024
    return peak_mb


def measure_cpu_memory(model, bvp, eda):
    process = psutil.Process(os.getpid())

    before = process.memory_info().rss
    with torch.no_grad():
        _ = model(bvp, eda)
    after = process.memory_info().rss

    print(f"CPU memory used during inference: {(after - before) / 1024 / 1024:.3f} MB")


if __name__ == '__main__':
    # from StressMeta_me.mobile_model import Model
    # from Baseline.mobile_model import Model
    # from BCSA.bcsa_mobile import DAFMPPSR
    # from Han.cnn_mobile import model_conv1d
    from ResNet.model import Model
    # from VGG.model import Model
    import argparse

    parser = argparse.ArgumentParser(description='VIAStress')
    parser.add_argument('--x_dim', type=int, default=256)
    parser.add_argument('--y_dim', type=int, default=2)
    parser.add_argument('--h_dim', type=int, default=256)
    parser.add_argument('--r_dim', type=int, default=256)

    args = parser.parse_args()

    bvp = torch.rand((1, 1, 1920))
    eda = torch.rand((1, 1, 120))

    # eda_emb = EDAEmbed(128)
    # print(eda_emb(eda).shape)

    # model = Model(args.x_dim, args.r_dim, args.h_dim, args.y_dim)
    model = Model(512, args.y_dim)
    # model = DAFMPPSR(x_dim=128, heads=4, n_bcsa=4, y_dim=2)
    # model = model_conv1d()
    # model = Net(args.y_dim)

    flops, params = profile(model, inputs=(bvp, eda))

    print("FLOPs:", flops)
    print("Params:", params)
    print("Peak inference memory (CPU):", measure_inference_memory_cpu(model, bvp, eda) + get_param_memory(model), "MB")
    # measure_cpu_memory(model, bvp, eda)
