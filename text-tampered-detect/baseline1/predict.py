import torch
from torchvision import transforms as T
import cv2
import os
from preprocess.loadImage import LoadImage
import argparse

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--model-file", type=str, nargs='+', default="0")
parser.add_argument("--publish", type=bool, default=False)
parser.add_argument("--eval", type=bool, default=False)
parser.add_argument("--resize", type=int, default=256)
parser.add_argument("--e", type=float, default=0.1)
args = parser.parse_args()


trans = {384:T.Compose([LoadImage(), T.ToTensor(), T.Resize(384), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
    128:T.Compose([LoadImage(), T.ToTensor(), T.RandomCrop((128, 128))]),
    224:T.Compose([LoadImage(), T.ToTensor(),T.Resize((224,224))]),
    299:T.Compose([LoadImage(), T.ToTensor(),T.Resize((299,299)),T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
    256:T.Compose([LoadImage(), T.ToTensor(), T.Resize((256, 256)),T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
    512:T.Compose([LoadImage(), T.ToTensor(),T.Resize((512, 512)), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
    768:T.Compose([LoadImage(), T.ToTensor(), T.Resize((768, 768)),T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
    '256un':T.Compose([LoadImage(), T.ToTensor(), T.RandomCrop(384), T.Resize((256, 256)),T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
    1280:T.Compose([LoadImage(), T.ToTensor(),T.Resize((128,128))]),
    0:T.Compose([LoadImage(), T.ToTensor(), ])}

trans = trans[args.resize]
models = []
for f in args.model_file:
    model = torch.load(f)
    model = model.cuda()
    model.eval()
    models.append(model)

if args.publish:
    def detect(img_path, e=0):
        # img = cv2.imread(img_path)
        img = trans(img_path)
        img = img.unsqueeze(0).float().cuda().to(torch.float32)
        
        pred_res = 0.0
        for model in models:
            res = model(img)
            prob = torch.nn.functional.softmax(res)

            pred_res += prob[:,1].detach().cpu().numpy()[0]

        pred_res /= len(models)
        # if pred_res > 1-e or pred_res < e:
        #     pred_res = 0.5

        return pred_res

    test_path = '../../../data/text_manipulation_detection/test/imgs'
    with open(f"submission.csv", "w") as f:
        for p in os.listdir(test_path):
            rela_path = os.path.join(test_path, p)
            # print(rela_path)
            res = detect(rela_path, args.e)
            f.write(f"{p} {res}\n")
            f.flush()

if args.eval:
    from eval import eval
    res = eval(model, trans, e=args.e, hard=True)
    print(res)
