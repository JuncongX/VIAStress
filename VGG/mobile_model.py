import torch
import torch.nn as nn

from VGG.model import Model


if __name__ == '__main__':

    bvp = torch.rand((1, 1, 1920))
    eda = torch.rand((1, 1, 120))


    model = Model(512, 2)
    torch.save(model.state_dict(), "vgg_mobile.pt")
    out = model(bvp, eda)
    print(out.shape)

    example_inputs = (bvp, eda)
    traced_model = torch.jit.trace(model, example_inputs)
    traced_model.save("vgg_mobile_traced.pt")