import numpy as np
import torch
import cv2

from src.hrnet.pose_hrnet import get_pose_net
from src.utils.torch_utils import select_device
from src.utils.utils_hrnet import get_max_preds

class HRNetModel:

    def __init__(self, cfg, img_size:(tuple,list), device:str, make_preprocess=True):

        self.cfg = cfg
        self.device = select_device(device)
        self.img_size = img_size
        self.make_preprocess = make_preprocess
        self.keypoints_names = self.cfg.KEYPOINTS_NAMES
        self.model = get_pose_net(self.cfg, is_train=False)
        self.model.load_state_dict(torch.load(self.cfg.OUTPUT_DIR))
        if self.device.type != 'cpu':
            self.model = torch.nn.DataParallel(self.model, device_ids=self.cfg.GPUS).cuda()

    def crop_image(self, img:np.ndarray, bbox:(tuple,list)) -> np.ndarray:
        h,w,_  = img.shape
        cropped_img = img[bbox[1]*h:bbox[3]*h, bbox[0]*w:bbox[2]*w, :]
        return cropped_img

    def preprocess_img(self, img: np.ndarray) -> np.ndarray:

        img_res = cv2.resize(img, (self.img_size[1], self.img_size[0]))
        img_res = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
        h_ratio = img.shape[0] / img_res.shape[0] * 4
        w_ratio = img.shape[1] / img_res.shape[1] * 4
        img_res = img_res.transpose(2, 0, 1)

        return img_res, h_ratio, w_ratio

    def get_tensor(self, img: np.ndarray) -> torch.Tensor:

        img_tensor = torch.tensor(img, dtype=torch.float32).to(self.device)
        #         img_tensor = img_tensor.float()  # uint8 to fp16/32
        #         img_tensor /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        return img_tensor

    def predict_tensor(self, tensor: torch.Tensor) -> tuple:

        output = self.model(tensor)

        return output

    def postprocess(self, pred: torch.Tensor, h_ratio: (int, float), w_ratio: (int, float)) -> tuple:

        preds_res = get_max_preds(pred.cpu().detach().numpy())
        preds = np.zeros_like(preds_res[0])
        preds[0, :, 0] = preds_res[0][0, :, 0] * w_ratio
        preds[0, :, 1] = preds_res[0][0, :, 1] * h_ratio

        return preds[0], preds_res[1][0]


    def __call__(self, img: (str, np.ndarray), h_ratio=None, w_ratio=None) -> tuple:

        if isinstance(img, str):
            img = cv2.imread(img)

        if self.make_preprocess:
            img, h_ratio, w_ratio = self.preprocess_img(img)

        tensor = self.get_tensor(img)
        preds = self.predict_tensor(tensor)
        coords, confs = self.postprocess(preds, h_ratio, w_ratio)

        return coords, confs

    def predict(self, image:np.ndarray, bboxes:(tuple,list), ndigits=3) -> list:

        answers = []
        for bbox in bboxes:
            result = dict()
            crp_img = self.crop_image(image, bbox)
            coords, confs = self(crp_img)
            for i, k_name in enumerate(self.keypoints_names):
                result[k_name] = {'x': float(coords[i][0]),
                                  'y': float(coords[i][1]),
                                  'proba': float(round(confs[i][0], ndigits))}
            answers.append(result)

        return answers