import argparse
import importlib
import os
import pprint

import cv2
import numpy as np
import torch
import torch.optim
import yaml
from easydict import EasyDict
from einops import rearrange
from torch import nn
from utils.misc_helper import create_logger

parser = argparse.ArgumentParser(description="UniAD")
parser.add_argument("--config", default="./config.yaml")
parser.add_argument("--class_name", default="")


def update_config(config):
    # update planes & strides
    backbone_path, backbone_type = config.net[0].type.rsplit(".", 1)
    module = importlib.import_module(backbone_path)
    backbone_info = getattr(module, "backbone_info")
    backbone = backbone_info[backbone_type]
    outplanes = []
    for layer in config.net[0].kwargs.outlayers:
        if layer not in backbone["layers"]:
            raise ValueError(
                "only layer {} for backbone {} is allowed, but get {}!".format(
                    backbone["layers"], backbone_type, layer
                )
            )
        idx = backbone["layers"].index(layer)
        outplanes.append(backbone["planes"][idx])

    config.net[2].kwargs.instrides = config.net[1].kwargs.outstrides
    config.net[2].kwargs.inplanes = [sum(outplanes)]
    return config


def load_state_decoder(path, model):
    def map_func(storage, location):
        return storage.cuda()

    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))

        checkpoint = torch.load(path, map_location=map_func)
        state_dict = checkpoint["state_dict"]

        # state_dict of decoder
        state_dict_decoder = {}
        for k, v in state_dict.items():
            if "module.reconstruction." in k:
                k_new = k.replace("module.reconstruction.", "")
                state_dict_decoder[k_new] = v

        # fix size mismatch error
        ignore_keys = []
        for k, v in state_dict_decoder.items():
            if k in model.state_dict().keys():
                v_dst = model.state_dict()[k]
                if v.shape != v_dst.shape:
                    ignore_keys.append(k)
                    print(
                        "caution: size-mismatch key: {} size: {} -> {}".format(
                            k, v.shape, v_dst.shape
                        )
                    )

        for k in ignore_keys:
            state_dict_decoder.pop(k)

        model.load_state_dict(state_dict_decoder, strict=False)

        ckpt_keys = set(state_dict_decoder.keys())
        own_keys = set(model.state_dict().keys())
        missing_keys = own_keys - ckpt_keys
        for k in missing_keys:
            print("caution: missing keys from checkpoint {}: {}".format(path, k))
    else:
        print("=> no checkpoint found at '{}'".format(path))


def main():
    args = parser.parse_args()

    with open(args.config) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    update_config(config)
    config.saver.load_path = config.saver.load_path.replace(
        "{class_name}", args.class_name
    )
    hidden_dim = config.vis_query.hidden_dim
    feat_dim = config.net[2].kwargs.inplanes[0]
    instride = config.net[2].kwargs.instrides[0]
    feat_size = [_ // instride for _ in config.data.input_size]

    config.exp_path = os.path.dirname(args.config)
    config.save_path = os.path.join(config.exp_path, config.saver.save_dir)
    config.log_path = os.path.join(config.exp_path, config.saver.log_dir)
    os.makedirs(config.save_path, exist_ok=True)
    os.makedirs(config.log_path, exist_ok=True)

    logger = create_logger(
        "global_logger", config.log_path + "/dec_{}.log".format(args.class_name)
    )
    logger.info("args: {}".format(pprint.pformat(args)))
    logger.info("config: {}".format(pprint.pformat(config)))

    # create model
    module_name, cls_name = config.net[2].type.rsplit(".", 1)
    module = importlib.import_module(module_name)
    model = getattr(module, cls_name)(**config.net[2].kwargs)
    load_state_decoder(config.saver.load_path, model)
    model.cuda()

    mean = (
        torch.tensor(config.data.pixel_mean).cuda().unsqueeze(0).unsqueeze(0)
    )  # 1 x 1 x 3
    std = (
        torch.tensor(config.data.pixel_std).cuda().unsqueeze(0).unsqueeze(0)
    )  # 1 x 1 x 3

    model_query = torch.load(config.vis_query.model_path)["state_dict"]

    # proj learned_embed from hidden_dim to feat_dim
    output_proj = nn.Linear(hidden_dim, feat_dim).cuda()
    state_dict_proj = {}
    for k in model_query.keys():
        if "module.reconstruction.output_proj." in k:
            k_new = k.replace("module.reconstruction.output_proj.", "")
            state_dict_proj[k_new] = model_query[k]
    output_proj.load_state_dict(state_dict_proj)

    queries = []
    queries.append(torch.rand(feat_size[0] * feat_size[1], hidden_dim).cuda())
    queries.append(torch.ones(feat_size[0] * feat_size[1], hidden_dim).cuda())
    queries.append(torch.zeros(feat_size[0] * feat_size[1], hidden_dim).cuda())
    for idx_layer in range(config.vis_query.num_decoder_layers):
        k_query = f"module.reconstruction.transformer.decoder.layers.{idx_layer}.learned_embed.weight"
        learned_embed = model_query[k_query].clone().detach()
        queries.append(learned_embed)

    images = []
    for learned_embed in queries:
        learned_embed = output_proj(learned_embed.cuda()).unsqueeze(0)
        learned_embed = rearrange(
            learned_embed, "b (h w) c -> b c h w", h=feat_size[0]
        )  # b x c X h x w
        input = {"feature_align": learned_embed}
        with torch.no_grad():
            output = model(input)
        image_rec = (
            output["image_rec"].squeeze(0).permute(1, 2, 0)
        )  # 1 x 3 x h x w -> h x w x 3
        image_rec = (image_rec * std + mean) * 255
        image_rec = image_rec.cpu().numpy()
        images.append(image_rec)

    # write image
    image = np.ascontiguousarray(np.concatenate(images, axis=1))  # h x 4w x 3
    if config.vis_query.with_text:
        texts = ["baseline rand", "baseline ones", "baseline zeros"]
        for idx_layer in range(config.vis_query.num_decoder_layers):
            texts.append(f"query layer{idx_layer}")
        for idx, text in enumerate(texts):
            x = idx * config.data.input_size[1] + 10
            y = config.data.input_size[0] - 10
            cv2.putText(
                image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
            )
    savepath = os.path.join(config.save_path, f"query_{args.class_name}.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(savepath, image)

    print(f"Success: Class: {args.class_name}, Saved: {savepath}")


if __name__ == "__main__":
    main()
