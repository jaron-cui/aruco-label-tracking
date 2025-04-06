import cv2
import numpy as np


class LemonTransmutation:
  def __init__(self):
    pass

  def transform(self, image: np.ndarray, lemon_pixel: np.ndarray, inplace: bool = False) -> np.ndarray:
    """
    Recolors a lemon in an image specified by a pixel on the lemon.

    :param image: a (height x width x 3) RGB image
    :param lemon_pixel: the 2D pixel coordinate of a known pixel on the lemon
    :param inplace: apply the transformation in-place
    :return: the image with the lemon recolored
    """
    if not inplace:
      image = image.copy()
    lemon_pixel = lemon_pixel.astype(np.uint64)

    # original image and hue-shifted image
    cv2.cvtColor(image, cv2.COLOR_RGB2HSV, dst=image)
    hue_shifted = self._shift_hue(image.copy())

    # calculate mask for lemon and background
    yellow_mask = self._get_yellow_mask(image)
    lemon_mask = np.zeros((image.shape[0] + 2, image.shape[1] + 2), np.uint8)
    cv2.floodFill(yellow_mask.astype(np.uint8), lemon_mask, lemon_pixel, (1,), [0], [0], flags=cv2.FLOODFILL_MASK_ONLY)
    lemon_mask = lemon_mask[1:-1, 1:-1]
    background_mask = cv2.bitwise_not(lemon_mask * 255)

    # combine the background of the original image with the hue-shifted lemon
    background = cv2.bitwise_and(image, image, mask=background_mask)
    lime = cv2.bitwise_and(hue_shifted, hue_shifted, mask=lemon_mask)
    lemon_to_lime = cv2.add(background, lime)
    cv2.cvtColor(lemon_to_lime, cv2.COLOR_HSV2RGB, dst=lemon_to_lime)

    return lemon_to_lime

  def _get_yellow_mask(self, hsv: np.ndarray):
    # we only want to select the upper triangular portion of the saturation/value plot for a given hue
    hue_channel = hsv[:, :, 0]
    saturation_channel = hsv[:, :, 1]
    value_channel = hsv[:, :, 2]

    min_saturation = 60
    min_value = np.minimum(255 - saturation_channel, 240)

    hue_match_mask = (14 <= hue_channel) & (hue_channel <= 37)
    saturation_match_mask = saturation_channel >= min_saturation
    value_match_mask = value_channel >= min_value

    yellow_mask = hue_match_mask & saturation_match_mask & value_match_mask

    return yellow_mask

  def _shift_hue(self, hsv: np.ndarray):
    yellow_hue = 28
    green_hue = 28 + 18

    hue_scaling = 0.7
    value_scaling = 0.8

    hsv[:, :, 0] = (hsv[:, :, 0] - yellow_hue) * hue_scaling + green_hue
    # hue_shifted[:, :, 1] *= 1.4
    hsv[:, :, 2] = hsv[:, :, 2] * value_scaling

    return np.clip(hsv, 0, 255).astype(np.uint8)

# import decord
# import matplotlib.pyplot as plt
# from pathlib import Path
# def video_frames_extractor(video_path: Path):
#   vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
#   frames = []
#   for i in range(len(vr)):
#     frame = vr[i].asnumpy()
#     # draw_marker(frame)
#     frames.append(frame)
#   return frames
#
#
# def show(image):
#   plt.imshow(np.flip(cv2.resize(image, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_AREA).transpose(1, 0, 2), axis=0))
#   plt.figure()
#
#
# frame = video_frames_extractor(Path('../data/RGB_2025-03-05-14_58_10.mp4'))[90]
# transform = LemonTransmutation()
# out = transform.transform(frame, np.array([220, 480]))
# show(out)
# plt.show()
