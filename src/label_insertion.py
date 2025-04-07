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
        aruco_background_reference_hsv: np.ndarray,
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

        self.aruco_background_reference_hsv = aruco_background_reference_hsv

        # pin the label colors as relative to the aruco label background colors so that label colors match the env
        self.label_hsv_shift, self.border_hsv_shift = [np.array([
            target - source for target, source in zip(target_hsv, aruco_background_reference_hsv)
        ]) for target_hsv in [label_hsv, border_hsv]]

        self.local_variation = [TruncatedNormal(0, std, -100, 100) for std in local_hsv_std]
        self.label_resolution = label_resolution
        self.border_width = np.random.randint(*border_width_range)

    def transform(self, image: np.ndarray, aruco_corners: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        label_corners = scale_quadrilateral(aruco_corners, 1.34)
        label_mask = get_label_mask(image, label_corners)
        aruco_background_hsv = get_aruco_background_color(hsv, label_mask, self.aruco_background_reference_hsv)

        label_hsv_color = aruco_background_hsv + self.label_hsv_shift
        border_hsv_color = aruco_background_hsv + self.border_hsv_shift

        # assemble basic label and border colors
        label_size = np.array(self.label_resolution) - 2 * self.border_width
        label_hsv = np.stack([np.full(label_size, x) for x in label_hsv_color], axis=2)
        label_image = np.stack([np.full(self.label_resolution, x) for x in border_hsv_color], axis=2)
        label_image[self.border_width:-self.border_width, self.border_width:-self.border_width] = label_hsv

        # add local noise
        noise = np.stack([dist.sample(self.label_resolution) for dist in self.local_variation], axis=2)
        label_image += noise
        label_image = label_image.clip(0, 255).astype(np.uint8)

        cv2.cvtColor(label_image, cv2.COLOR_HSV2RGB, dst=label_image)
        cv2.blur(label_image, (3, 3), dst=label_image)

        # map onto image
        pts1 = np.float32([[0, 0], [0, self.label_resolution[1]], self.label_resolution, [self.label_resolution[0], 0]])
        pts2 = np.float32(label_corners)
        H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

        mapped_label = cv2.warpPerspective(label_image, H, (image.shape[1], image.shape[0]))
        gray_mapped_label = cv2.cvtColor(mapped_label, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_mapped_label, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        dest_bg = cv2.bitwise_and(image, image, mask=mask_inv)
        warped_fg = cv2.bitwise_and(mapped_label, mapped_label, mask=mask)
        final_img = cv2.add(dest_bg, warped_fg)

        return final_img


def get_label_mask(image: np.ndarray, label_corners: np.ndarray) -> np.ndarray:
    mask = np.zeros(image.shape[:-1], dtype=np.uint8)
    # print(label_corners, image.shape, mask.shape)
    return cv2.fillPoly(mask, [label_corners.astype(np.int32)], 255)


def get_aruco_background_color(
    image: np.ndarray, label_mask: np.ndarray, aruco_background_hsv: np.ndarray
) -> np.ndarray:
    label_only = cv2.bitwise_and(image, image, mask=label_mask)
    hsv_similarity_by_pixel = -np.abs(label_only - aruco_background_hsv).sum(axis=2)
    most_similar_pixel = np.argmax(hsv_similarity_by_pixel)
    # print(np.unravel_index(most_similar_pixel, image.shape[:-1]))
    return image[*np.unravel_index(most_similar_pixel, image.shape[:-1])]





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


def scale_quadrilateral(points: np.ndarray, scale_factor: float) -> np.ndarray:
  center = seg_intersect(points[0], points[2], points[1], points[3])
  return (points - center) * scale_factor + center

import decord
import matplotlib.pyplot as plt
from pathlib import Path
import cv2.aruco as aruco
def video_frames_extractor(video_path: Path):
  vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
  frames = []
  for i in range(len(vr)):
    frame = vr[i].asnumpy()
    # draw_marker(frame)
    frames.append(frame)
  return frames


def show(image):
  plt.imshow(np.flip(cv2.resize(image, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_AREA).transpose(1, 0, 2), axis=0))
  plt.figure()

frame = video_frames_extractor(Path('../data/RGB_2025-03-05-14_58_10.mp4'))[90]
# frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
parameters =  aruco.DetectorParameters()
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
detector = aruco.ArucoDetector(aruco_dict, parameters)

# Detect ArUco markers
marker_corners, marker_ids, _ = detector.detectMarkers(gray)
marker_corners, marker_ids = np.array(marker_corners).squeeze(1), np.array(marker_ids).squeeze(1)
# marker_corners = [scale_quadrilateral(corners, 1.3) for corners in marker_corners]
# frame = video_frames_extractor(Path('../data/RGB_2025-03-05-14_58_10.mp4'))[90]
# transform = LemonTransmutation()
# out = transform.transform(frame, np.array([220, 480]))
label_color = get_aruco_background_color(cv2.cvtColor(frame, cv2.COLOR_RGB2HSV), get_label_mask(frame, marker_corners[0]), np.array([65, 15, 255]))
color = np.zeros((10, 10, 3), dtype=np.uint8)
color[:, :] = label_color
# print('Label color:', label_color)
cv2.cvtColor(color, cv2.COLOR_HSV2RGB, dst=color)

label_insertion = LabelInsertion(
        aruco_background_reference_hsv=label_color,
        label_hsv=(28, 220, 200),
        border_hsv=(117, 180, 240),
        global_hsv_std=(4, 20, 20),
        local_hsv_std=(2, 10, 10),
    )


# show(label_insertion.transform(frame, marker_corners[0]))
# plt.show()

# label = np.zeros((100, 100, 3), dtype=np.uint8)
# label[:, :] = np.array([255, 0, 0])
# label[12:-12, 12:-12] = np.array([0, 255, 255])
# cv2.imwrite('../out/lemon-label.png', label)
# label[12:-12, 12:-12] = np.array([50, 205, 50])
# cv2.imwrite('../out/lime-label.png', label)
# label[12:-12, 12:-12] = np.array([0, 165, 255])
# cv2.imwrite('../out/orange-label.png', label)