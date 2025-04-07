from pathlib import Path

import cv2
import cv2.aruco as aruco
import decord
import numpy as np

from aruco_processing import TrackingAnnotator
from label_insertion import LabelInsertion
from lemon_transmutation import LemonTransmutation
import matplotlib.pyplot as plt
def show(image):
  plt.imshow(np.flip(cv2.resize(image, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_AREA).transpose(1, 0, 2), axis=0))
  plt.figure()

def transmute_lemons_to_limes(frames: np.ndarray):
    lemon_to_lime = LemonTransmutation()
    lemon_location = np.array([[220, 480]], dtype=np.float32)

    for frame_index in range(frames.shape[0] - 1):
        next_lemon_location = cv2.calcOpticalFlowPyrLK(
            frames[frame_index], frames[frame_index + 1], lemon_location, None)[0]
        frames[frame_index] = lemon_to_lime.transform(frames[frame_index], lemon_location[0], inplace=True)
        # show(frames[frame_index])
        # plt.show()
        # return
        lemon_location = next_lemon_location
    frames[-1] = lemon_to_lime.transform(frames[-1], lemon_location[0], inplace=True)
    return frames


def insert_labels(frames: np.ndarray):
    insert_label = LabelInsertion(
        aruco_background_reference_hsv=np.array([41, 70, 208]),
        label_hsv=(28, 220, 200),
        border_hsv=(117, 180, 240),
        global_hsv_std=(4, 20, 20),
        local_hsv_std=(2, 10, 10),
    )

    def annotate_frame(frame: np.ndarray, markerId: int, corners: np.ndarray):
        if markerId != 1:
            return
        insert_label.transform(frame, corners)

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    annotator = TrackingAnnotator(aruco_dict, annotate_frame)
    annotator.annotate_frames(frames)

# path = '../data/RGB_2025-03-05-14_58_10.mp4'
path = "C:/Users\clack\Downloads\AnySense/aruco-lemon-sorting-jaron/2025-03-05-18_00_16\RGB_2025-03-05-18_00_16.mp4"
vr = decord.VideoReader(path, ctx=decord.cpu(0))
fps = vr.get_avg_fps()
frames = []
for i in range(len(vr)):
    frame = vr[i].asnumpy()
    # draw_marker(frame)
    frames.append(frame)

frames = transmute_lemons_to_limes(np.stack(frames, axis=0))
insert_labels(frames)
# fourcc = cv2.VideoWriterProperties(*'mp4v')  # Codec for video encoding
video = cv2.VideoWriter('../out/transmutation.mp4', -1, fps, (frames[0].shape[1], frames[0].shape[0]))
for frame in frames:
    video.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

cv2.destroyAllWindows()
video.release()


