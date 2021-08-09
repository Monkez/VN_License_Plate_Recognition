import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from modules.utils import CTCLabelConverter, AttnLabelConverter
from modules.dataset import RawDataset, AlignCollate, np_RawDataset
from modules.model import Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from PIL import Image

import sys
print(sys.argv)
import config as opt

opt.imgH =110
opt.imgW = 180
opt.num_fiducial = 8
opt.input_channel = 3
opt.output_channel =256
opt.hidden_size = 128
opt.batch_max_length = 16
opt.Transformation = 'TPS'
opt.FeatureExtraction ='ResNet'
opt.SequenceModeling ='BiLSTM'
opt.Prediction = 'Attn'
opt.saved_model = 'saved_models/best_accuracy.pth'
opt.image_folder = "None"
opt.batch_size = 192
opt.character = "0123456789abcdefghijklmnopqrstuvwxyz"
opt.sensitive = False
opt.rgb = True
opt.PAD = False
opt.workers = 0
opt.num_gpu = torch.cuda.device_count()

cudnn.benchmark = True
cudnn.deterministic = True
opt.num_gpu = torch.cuda.device_count()

if 'CTC' in opt.Prediction:
    converter = CTCLabelConverter(opt.character)
else:
    converter = AttnLabelConverter(opt.character)
opt.num_class = len(converter.character)
if opt.rgb:
    opt.input_channel = 3
model = Model(opt)
model = torch.nn.DataParallel(model).to(device)

# load model
model.load_state_dict(torch.load(opt.saved_model, map_location=device))

def predict(images_arr):
    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    images = [Image.fromarray(crop) for crop in images_arr]
    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    demo_data = np_RawDataset(images=images, opt=opt)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    # predict
    model.eval()
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            stt_c = 0
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            if 'CTC' in opt.Prediction:
                preds = model(image, text_for_pred)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                preds_index = preds_index.view(-1)
                preds_str = converter.decode(preds_index.data, preds_size.data)
            else:
                preds = model(image, text_for_pred, is_train=False)
                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            stt = 0
            out = []
            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                if 'Attn' in opt.Prediction:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]

                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                pred = pred.replace("@", "-")
                out.append(pred)
            return out