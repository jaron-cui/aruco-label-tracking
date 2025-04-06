import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import List, Tuple


HSV_MAXIMUMS = [179, 255, 255]


class TruncatedNormal(torch.distributions.Normal):
    def __init__(self, loc: float, scale: float, lo: float, hi: float):
        super().__init__(loc, scale)
        self.lo = lo
        self.hi = hi

    def sample(self, sample_shape=torch.Size()):
        sampled = super().sample(sample_shape)
        return sampled.clamp(self.lo, self.hi)


class LabelInsertion:
    def __init__(
        self,
        aruco_background_reference_hsv: Tuple[int, int, int],
        label_hsv: Tuple[int, int, int],
        border_hsv: Tuple[int, int, int],
        global_hsv_std: Tuple[int, int, int],
        local_hsv_std: Tuple[int, int, int],
        label_resolution: Tuple[int, int] = (100, 100),
        border_width_range: Tuple[int, int] = (8, 12)
    ):
        # introduce some variance in the overall label colors
        label_hsv, border_hsv = [[
            TruncatedNormal(x, std, 0, hi).sample() for x, std, hi in zip(hsv, global_hsv_std, HSV_MAXIMUMS)
        ] for hsv in (label_hsv, border_hsv)]

        # pin the label colors as relative to the aruco label background colors so that label colors match the env
        self.label_hsv_shift, self.border_hsv_shift = [np.array([
            target - source for target, source in zip(target_hsv, aruco_background_reference_hsv)
        ]) for target_hsv in [label_hsv, border_hsv]]

        self.local_variation = [TruncatedNormal(0, std, -100, 100) for std in local_hsv_std]
        self.label_resolution = label_resolution
        self.border_width = np.random.randint(*border_width_range)

    def transform(self, image: np.ndarray, aruco_corners: np.ndarray) -> np.ndarray:
        label_corners = scale_quadrilateral(aruco_corners, 1.3)
        label_mask = self._get_label_mask(image, label_corners)
        aruco_background_hsv = get_aruco_background_color(image, label_mask)

        label_hsv_color = aruco_background_hsv + self.label_hsv_shift
        border_hsv_color = aruco_background_hsv + self.border_hsv_shift

        label_size = np.array(self.label_resolution) - 2 * self.border_width
        label_hsv = np.stack([np.full(label_size, x) for x in label_hsv_color], axis=0)
        label_image = np.stack([np.full(self.label_resolution, x) for x in border_hsv_color], axis=0)
        label_image[self.border_width:-self.border_width, self.border_width:-self.border_width, :] = label_hsv

        noise = np.stack([dist.sample(self.label_resolution) for dist in self.local_variation], axis=0)
        label_image += noise

        cv2.cvtColor(label_image, cv2.COLOR_HSV2RGB, dst=label_image)
        cv2.blur(label_image, (3, 3), dst=label_image)

        return label_image


hue_range = [22, 37]
saturation_range = [0, 255]
value_range = [180, 255]
hue_mean = 30
saturation_mean = 200
value_mean = 255
hue_dist = TruncatedNormal(TruncatedNormal(hue_mean, 7, *hue_range).sample(), 3, *hue_range)
hue_dist2 = TruncatedNormal(TruncatedNormal(120, 9, 105, 130).sample(), 5, 105, 130)
saturation_dist = TruncatedNormal(saturation_mean, 200, *saturation_range)
value_dist = TruncatedNormal((value_mean - value_range[0]) / (value_range[1] - value_range[0]), 0.5, 0, 1)

channel_shape = 100, 100
border_width = np.random.randint(8, 12)
hue = hue_dist.sample(channel_shape)
hue2 = hue_dist2.sample(channel_shape)
hue, hue2 = cv2.blur(np.array(hue), (3, 3)), cv2.blur(np.array(hue2), (3, 3))
hue2[border_width:-border_width, border_width:-border_width] = hue[border_width:-border_width, border_width:-border_width]
saturation = saturation_dist.sample(channel_shape)
value_interval = saturation / (saturation_range[1] - saturation_range[0]) * (value_range[1] - value_range[0])
value = value_range[1] - (1 - value_dist.sample(channel_shape)) * value_interval

background = np.stack([hue2, saturation, value], axis=2).astype(np.uint8)

lemon_label_image = cv2.cvtColor(background, cv2.COLOR_HSV2RGB)
lemon_label_image = cv2.blur(lemon_label_image, (4, 4))
plt.imshow(cv2.resize(lemon_label_image, (50, 50), interpolation=cv2.INTER_AREA))


#
# line segment intersection using vectors
# see Computer Graphics by F.S. Hill
#
def perp(a) :
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b


# line segment a given by endpoints a1, a2
# line segment b given by endpoints b1, b2
# return
def seg_intersect(a1,a2, b1,b2) :
    da = a2 - a1
    db = b2 - b1
    dp = a1 - b1
    dap = perp(da)
    denom = np.dot(dap, db)
    num = np.dot(dap, dp)
    return (num / denom.astype(float)) * db + b1


def scale_quadrilateral(points: np.ndarray, scale_factor: float) -> List[Tuple[float]]:
  center = seg_intersect(points[0], points[2], points[1], points[3])
  return [(point - center) * scale_factor + center for point in points]
