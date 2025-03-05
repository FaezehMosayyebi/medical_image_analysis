from torch import nn
import torch
import numpy as np
import torch.nn.functional as F


class ConvNonlin(nn.Module):
    def __init__(self, input_channels, output_channels, stride):
        super(ConvNonlin, self).__init__()

        self.conv = nn.Conv3d(
            input_channels, output_channels, kernel_size=3, stride=stride, padding=1
        )
        self.norm = nn.BatchNorm3d(output_channels)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        return self.relu(self.norm(x))


class StackedConvLayers(nn.Module):
    def __init__(
        self, input_feature_channels, output_feature_channels, num_convs, first_stride
    ):
        super(StackedConvLayers, self).__init__()

        self.input_channels = input_feature_channels
        self.output_channels = output_feature_channels

        self.blocks = nn.Sequential(
            *(
                [
                    ConvNonlin(
                        self.input_channels,
                        self.output_channels,
                        first_stride,
                    )
                ]
                + [
                    ConvNonlin(
                        self.output_channels,
                        self.output_channels,
                        (1, 1, 1),
                    )
                    for _ in range(num_convs - 1)
                ]
            )
        )

    def forward(self, x):
        return self.blocks(x)


class UNet(nn.Module):
    def __init__(self, input_channels, base_num_features, num_layers, do_ds):
        super(UNet, self).__init__()

        self.encoder_conv_blocks = []  # this is used to get skip connections and store encoder convolutions
        self.decoder_conv_blocks = []  # is used to store decoder convolutions
        self.tu = []  # is used to store upsamplings
        self.seg_outputs = []  # is used to store the outputs in each stage
        self.do_ds = do_ds  # for deep supervision

        input_features = input_channels
        output_features = base_num_features

        # downsampling (encoder)
        for d in range(num_layers):
            if d != 0:  # determine the first stride
                first_stride = (2, 2, 2)
            else:
                first_stride = (1, 1, 1)

            # add convolutions
            self.encoder_conv_blocks.append(
                StackedConvLayers(
                    input_feature_channels=input_features,
                    output_feature_channels=output_features,
                    num_convs=2,
                    first_stride=first_stride,
                )
            )
            input_features = output_features
            output_features = int(np.round(output_features * 2))
            output_features = min(output_features, 320)  # 320 is max output feature

        # bottleneck
        final_num_features = output_features

        self.encoder_conv_blocks.append(
            StackedConvLayers(
                input_feature_channels=input_features,
                output_feature_channels=output_features,
                num_convs=2,
                first_stride=(2, 2, 2),
            )
        )

        nfeatures_from_down = final_num_features

        # upsampling
        for u in range(num_layers):
            # self.encoder_conv_blocks[-1] is bottleneck, so start with -2
            nfeatures_from_skip = self.encoder_conv_blocks[-(2 + u)].output_channels

            # number of features after upsampling ang concatting skip connections
            n_features_after_tu_and_concat = nfeatures_from_skip * 2

            self.tu.append(
                nn.ConvTranspose3d(
                    nfeatures_from_down, nfeatures_from_skip, (2, 2, 2), (2, 2, 2)
                )
            )
            self.decoder_conv_blocks.append(
                nn.Sequential(
                    StackedConvLayers(
                        input_feature_channels=n_features_after_tu_and_concat,
                        output_feature_channels=nfeatures_from_skip,
                        num_convs=2,
                        first_stride=(1, 1, 1),
                    ),
                )
            )

            nfeatures_from_down = nfeatures_from_skip

        for ds in range(len(self.decoder_conv_blocks)):  # for deep supervision
            self.seg_outputs.append(
                nn.Conv3d(
                    in_channels=self.decoder_conv_blocks[ds][-1].output_channels,
                    out_channels=3,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

        # register all modules properly
        self.encoder_conv_blocks = nn.ModuleList(self.encoder_conv_blocks)
        self.decoder_conv_blocks = nn.ModuleList(self.decoder_conv_blocks)
        self.tu = nn.ModuleList(self.tu)
        self.seg_outputs = nn.ModuleList(self.seg_outputs)

    def forward(self, x):
        skips = []
        seg_outputs = []

        # down
        for d in range(len(self.encoder_conv_blocks) - 1):
            x = self.encoder_conv_blocks[d](x)
            skips.append(x)

        # bottleneck
        x = self.encoder_conv_blocks[-1](x)

        # up
        for u in range(len(self.tu)):
            x = self.tu[u](x)

            x = torch.cat((x, skips[-(u + 1)]), dim=1)

            x = self.decoder_conv_blocks[u](x)

            # output
            seg_outputs.append(self.seg_outputs[u](x))

        if self.do_ds:
            return tuple([seg_outputs[i] for i in range(len(seg_outputs) - 1, -1, -1)])
        else:
            return seg_outputs[-1]
