import math
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms


from src.hrnet.pose_hrnet import get_pose_net
from src.utils.torch_utils import select_device
from src.utils.utils_hrnet import get_max_preds

class HRNetModel:

    def __init__(self, cfg, img_size:(tuple,list), device:str, make_preprocess=False):

        # cudnn related setting
        cudnn.benchmark = cfg.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
        torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

        self.cfg = cfg
        self.device = select_device(device)
        self.img_size = list(img_size)
        self.make_preprocess = make_preprocess
        self.keypoints_names = self.cfg.KEYPOINTS_NAMES
        self.model = get_pose_net(self.cfg, is_train=False)
        self.model.load_state_dict(torch.load(self.cfg.OUTPUT_DIR), strict=False)
        if self.device.type != 'cpu':
            self.model = torch.nn.DataParallel(self.model, device_ids=self.cfg.GPUS).cuda()
        self.model.eval()
        self.normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            self.normalize,
        ])

    def _get_dir(self, src_point, rot_rad):
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)

        src_result = [0, 0]
        src_result[0] = src_point[0] * cs - src_point[1] * sn
        src_result[1] = src_point[0] * sn + src_point[1] * cs

        return src_result

    def _get_3rd_point(self, a, b):
        direct = a - b
        return b + np.array([-direct[1], direct[0]], dtype=np.float32)

    def _get_affine_transform(self, center,
                              scale,
                              rot,
                              output_size,
                              shift=np.array([0, 0], dtype=np.float32),
                              inv=0):
        if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
            print(scale)
            scale = np.array([scale, scale])
        src_w = scale[0]
        dst_w = output_size[0]
        dst_h = output_size[1]

        rot_rad = np.pi * rot / 180
        src_dir = self._get_dir([0, src_w * -0.5], rot_rad)
        dst_dir = np.array([0, dst_w * -0.5], np.float32)

        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center + scale * shift
        src[1, :] = center + src_dir + scale * shift
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

        src[2:, :] = self._get_3rd_point(src[0, :], src[1, :])
        dst[2:, :] = self._get_3rd_point(dst[0, :], dst[1, :])

        if inv:
            trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

        return trans

    def affine_transform(self, pt, t):
        new_pt = np.array([pt[0], pt[1], 1.]).T
        new_pt = np.dot(t, new_pt)
        return new_pt[:2]

    def crop_image(self, image, bbox):
        """
        Вырезает прямоугольник из картинки
        с использованием афинных трансформаций.
        Взят у авторов сети.
        """
        # img = image.copy()
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        x, y, w, h = bbox
        # w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        # aspect_ratio = cfg.input_shape[1] / cfg.input_shape[0]
        aspect_ratio = self.img_size[1] / self.img_size[0]
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > aspect_ratio * h:
            h = w / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio
        if center[0] != -1:
            scale = np.array([w, h], dtype=np.float32) * 1.25
        rotation = 0

        # trans = self._get_affine_transform(center, scale, rotation, (cfg.input_shape[1], cfg.input_shape[0]))
        # cropped_img = cv2.warpAffine(img, trans, (cfg.input_shape[1], cfg.input_shape[0]), flags=cv2.INTER_LINEAR)

        trans = self._get_affine_transform(center, scale, rotation, (192,256))
        cropped_img = cv2.warpAffine(img, trans, (192,256), flags=cv2.INTER_LINEAR)

        crop_info = np.asarray([center[0] - scale[0] * 0.5, center[1] - scale[1] * 0.5, center[0] + scale[0] * 0.5,
                                center[1] + scale[1] * 0.5])

        h_ratio = img.shape[0] / cropped_img.shape[0] * 4
        w_ratio = img.shape[1] / cropped_img.shape[1] * 4

        meta = {'center': center,
                'scale': scale,
                'rotation': rotation,
                'trans': trans}

        return cropped_img, meta

    def transform_preds(self,coords, center, scale, output_size):
        target_coords = np.zeros(coords.shape)
        trans = self._get_affine_transform(center, scale, 0, output_size, inv=1)
        for p in range(coords.shape[0]):
            target_coords[p, 0:2] = self.affine_transform(coords[p, 0:2], trans)
        return target_coords

    def preprocess_bbox(self, bbox:(list, tuple), width, height) -> list:
        x1, y1, x2, y2 = bbox
        w = x2-x1
        h = y2-y1
        x1 = np.max((0, x1))
        y1 = np.max((0, y1))
        x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
        y2 = np.min((height - 1, y1 + np.max((0, h - 1))))

        return np.array([x1,y1, x2-x1, y2-y1],dtype=np.float64)

    def preprocess_img(self, img: np.ndarray) -> np.ndarray:
        img_res = img
        # img_res = cv2.resize(img, (self.img_size[1], self.img_size[0]))
        # img_res = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
        # h_ratio = img.shape[0] / img_res.shape[0] * 4
        # w_ratio = img.shape[1] / img_res.shape[1] * 4
        img_res = img_res.transpose(2, 0, 1)

        # return img_res, h_ratio, w_ratio
        return img_res

    def get_tensor(self, img: np.ndarray) -> torch.Tensor:
        img_tensor = img
        # img_tensor = torch.tensor(img, dtype=torch.float32).to(self.device)
        # img_tensor = self.normalize(img_tensor)
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        return img_tensor

    def predict_tensor(self, tensor: torch.Tensor) -> tuple:

        output = self.model(tensor)

        return output

    def postprocess(self, heatmaps: torch.Tensor, meta:dict) -> tuple:

        heatmap_height = heatmaps.shape[2]
        heatmap_width = heatmaps.shape[3]

        coords, maxvals = get_max_preds(heatmaps.cpu().detach().numpy())

        for p in range(coords.shape[1]):
            hm = heatmaps[0][p]
            px = int(math.floor(coords[0][p][0] + 0.5))
            py = int(math.floor(coords[0][p][1] + 0.5))
            if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                diff = np.array(
                    [
                        (hm[py][px+1] - hm[py][px-1]).cpu().detach().numpy(),
                        (hm[py+1][px]-hm[py-1][px]).cpu().detach().numpy()
                    ]
                )
                coords[0][p] += np.sign(diff) * .25

        new_preds = coords.copy()

        # Transform back
        new_preds[0] = self.transform_preds(
            coords[0], meta['center'], meta['scale'], [heatmap_width, heatmap_height]
        )

        return new_preds[0], maxvals[0]


    def __call__(self, img: (str, np.ndarray), meta: dict) -> tuple:

        if isinstance(img, str):
            img = cv2.imread(img)

        # if self.make_preprocess:
        #     # img, h_ratio, w_ratio = self.preprocess_img(img)
        #     img = self.preprocess_img(img)
        with torch.no_grad():
            tensor = self.get_tensor(self.transforms(img))
            torch.save(tensor, 'input_tensor.pt')
            output = self.predict_tensor(tensor)
            torch.save(output, 'output_tensor.pt')
            coords, confs = self.postprocess(output, meta)

        return coords, confs

    def predict(self, image:np.ndarray, bboxes:(tuple,list), ndigits=3) -> list:

        answers = []
        for bbox in bboxes:
            bbox = self.preprocess_bbox(bbox, image.shape[1], image.shape[0]) # reverse width and height because it is like in original implementation
            result = dict()
            crp_img,meta = self.crop_image(image, bbox)
            coords, confs = self(crp_img,meta)
            for i, k_name in enumerate(self.keypoints_names):
                result[k_name] = {'x': float(coords[i][0]),
                                  'y': float(coords[i][1]),
                                  'proba': float(round(confs[i][0], ndigits))}
                answers.append(result)

        return answers