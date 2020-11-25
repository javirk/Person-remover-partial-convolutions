import torch
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram


def dilation_holes(hole_mask):
    b, ch, h, w = hole_mask.shape
    dilation_conv = nn.Conv2d(ch, ch, 3, padding=1, bias=False).to(device)
    torch.nn.init.constant_(dilation_conv.weight, 1.0)
    with torch.no_grad():
        output_mask = dilation_conv(hole_mask)
    updated_holes = (output_mask != 0)
    return updated_holes.float()


def total_variation_loss(image, mask):
    hole_mask = 1 - mask
    dilated_holes = dilation_holes(hole_mask)
    columns_in_Pset = dilated_holes[:, :, :, 1:] * dilated_holes[:, :, :, :-1]
    rows_in_Pset = dilated_holes[:, :, 1:, :] * dilated_holes[:, :, :-1:, :]
    loss = torch.mean(torch.abs(columns_in_Pset * (image[:, :, :, 1:] - image[:, :, :, :-1]))) + \
           torch.mean(torch.abs(rows_in_Pset * (image[:, :, :1:] - image[:, :, -1:, :])))
    return loss


class InpaintingLoss(nn.Module):
    def __init__(self, extractor):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.extractor = extractor

    def forward(self, input_im, mask, output, gt):
        loss_dict = {}
        output_comp = mask * input_im + (1 - mask) * output

        loss_dict['hole'] = self.l1((1 - mask) * output, (1 - mask) * gt)
        loss_dict['valid'] = self.l1(mask * output, mask * gt)

        feat_output_comp = self.extractor(output_comp)
        feat_output = self.extractor(output)
        feat_gt = self.extractor(gt)

        loss_dict['prc'] = 0.0
        for i in range(3):
            loss_dict['prc'] += self.l1(feat_output[i], feat_gt[i])
            loss_dict['prc'] += self.l1(feat_output_comp[i], feat_gt[i])

        loss_dict['style'] = 0.0
        for i in range(3):
            loss_dict['style'] += self.l1(gram_matrix(feat_output[i]), gram_matrix(feat_gt[i]))
            loss_dict['style'] += self.l1(gram_matrix(feat_output_comp[i]), gram_matrix(feat_gt[i]))

        loss_dict['tv'] = total_variation_loss(output_comp, mask)

        return loss_dict

    @staticmethod
    def calculate_total_loss(loss_dict, lambda_dict):
        loss = 0.0
        for key, coef in lambda_dict.items():
            value = coef * loss_dict[key]
            loss += value
        return loss
