from dataclasses import dataclass
from functorch import einops
import torch
import decord
import numpy as np
import cv2
import cv2.aruco as aruco
import typing
from pathlib import Path

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)


@dataclass
class DetectionGap:
    start_frame_index: int
    start_corners: np.ndarray  # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    end_frame_index: int = None


def annotate_frame(frame: np.ndarray, markerId: int, corners: np.ndarray):
    corners = corners.astype("int")
    text_position = (corners[0, 0].item(), corners[0, 1].item() - 15)
    cv2.polylines(frame, [corners], True, (0, 255, 0), 2)
    cv2.putText(frame, str(markerId), text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


class TrackingAnnotator:
    def __init__(
        self,
        aruco_dict: aruco.Dictionary,
        annotate_frame: typing.Callable[[np.ndarray, int, np.ndarray], None]
    ):
        self.aruco_dict = aruco_dict
        self.annotate_frame = annotate_frame

    def annotate_frames(self, frames: np.ndarray | typing.List[np.ndarray]):
        grays = np.zeros(frames.shape[:-1], dtype=np.uint8)
        for frame_index in range(frames.shape[0]):
            grays[frame_index] = cv2.cvtColor(frames[frame_index], cv2.COLOR_RGB2GRAY)

        detections: typing.List[typing.Tuple[np.ndarray, np.ndarray]] = []
        for frame_index, frame in enumerate(frames):
            parameters = aruco.DetectorParameters()
            detector = aruco.ArucoDetector(aruco_dict, parameters)

            # Detect ArUco markers
            marker_corners, marker_ids, _ = detector.detectMarkers(grays[frame_index])

            if marker_ids is None:
                marker_ids = []
            if marker_corners:
                marker_corners, marker_ids = np.array(marker_corners).squeeze(1), np.array(marker_ids).squeeze(1)

            detections.append((marker_corners, marker_ids))

            # annotate the detected markers
            for i, marker_id in enumerate(marker_ids):
                self.annotate_frame(frame, marker_id, marker_corners[i])
        self._annotate_gaps(frames, grays, detections)

    def find_detection_gaps(
        self,
        frames: typing.List[np.ndarray],
        detections: typing.List[typing.Tuple[np.ndarray, np.ndarray]]
    ) -> typing.Dict[int, typing.List[DetectionGap]]:
        # detection gaps for which a detected frame prior and detected frame after have been found
        detection_gaps: typing.Dict[int, typing.List[DetectionGap]] = {}
        # detection gaps preceded by a detected frame but for which detection has not yet resumed
        unterminated_gaps: typing.Dict[int, DetectionGap] = {}

        previous_corners = []
        for frame_index, frame in enumerate(frames):
            # record when detection for an aruco marker comes back online
            marker_corners, marker_ids = detections[frame_index]
            for detection_index in range(len(marker_ids)):
                marker_id = marker_ids[detection_index].item()
                if marker_id not in unterminated_gaps:
                    continue
                gap = unterminated_gaps[marker_id]
                gap.end_frame_index = frame_index
                if marker_id not in detection_gaps:
                    detection_gaps[marker_id] = []
                detection_gaps[marker_id].append(gap)
                del unterminated_gaps[marker_id]

            # record when detection for an aruco marker goes offline
            for marker_id in [1, 2, 3]:
                if marker_id in marker_ids or marker_id in unterminated_gaps or marker_id not in previous_corners:
                    continue
                gap = DetectionGap(
                    frame_index - 1,
                    previous_corners[marker_id]
                )
                unterminated_gaps[marker_id] = gap

            previous_corners = {marker_ids[i]: marker_corners[i] for i in range(len(marker_ids))}

        # terminate any gaps for which detection never came back online
        for marker_id, partial_gap in unterminated_gaps.items():
            if marker_id not in detection_gaps:
                detection_gaps[marker_id] = []
            partial_gap.end_frame_index = len(frames)
            detection_gaps[marker_id].append(partial_gap)

        return detection_gaps

    def _annotate_gaps(
        self,
        frames: typing.List[np.ndarray],
        grays: np.ndarray,
        detections: typing.List[typing.Tuple[np.ndarray, np.ndarray]]
    ):
        forward_detection_gaps = self.find_detection_gaps(frames, detections)
        backward_detection_gaps = self.find_detection_gaps(list(reversed(frames)), list(reversed(detections)))

        forward_interpolations = self._interpolated_marker_positions(grays, forward_detection_gaps)
        backward_interpolations = self._interpolated_marker_positions(np.flip(grays, axis=0), backward_detection_gaps)

        combined_interpolation = {}
        for marker_id in forward_interpolations:
            forward_interpolation, backward_interpolation = forward_interpolations[marker_id], backward_interpolations[
                marker_id]
            interpolation = [None] * len(frames)
            for frame_index, (forward_corners, backward_corners) in enumerate(
                zip(forward_interpolation, reversed(backward_interpolation))):
                if forward_corners is None and backward_corners is None:
                    continue
                if forward_corners is None:
                    interpolation[frame_index] = backward_corners
                    continue
                if backward_corners is None:
                    interpolation[frame_index] = forward_corners
                    continue
                interpolation[frame_index] = (forward_corners + backward_corners) / 2
            combined_interpolation[marker_id] = interpolation
        missed = 0
        for marker_id, interpolation in combined_interpolation.items():
            for frame_index, corners in enumerate(interpolation):
                if corners is None:
                    if marker_id == 1:
                        missed += 1
                    continue
                self.annotate_frame(frames[frame_index], marker_id, corners)

    def _interpolated_marker_positions(
        self,
        grays: np.ndarray,
        detection_gaps: typing.Dict[int, typing.List[DetectionGap]]
    ) -> typing.Dict[int, typing.List[np.ndarray | None]]:
        interpolated_corners = {marker_id: [None] * len(grays) for marker_id in detection_gaps}
        for marker_id, marker_detection_gaps in detection_gaps.items():
            for detection_gap in marker_detection_gaps:
                corners = detection_gap.start_corners.copy()
                for frame_index in range(detection_gap.start_frame_index + 1, detection_gap.end_frame_index):
                    gray = grays[frame_index]
                    corners = cv2.calcOpticalFlowPyrLK(grays[frame_index - 1], grays[frame_index], corners, None)[0]
                    # if st.sum() != 4:
                    #   continue
                    corner_out_of_bounds = False
                    for corner_index, corner in enumerate(corners):
                        y, x = int(corner[0]), int(corner[1])
                        tolerance = 30
                        if not (-tolerance <= y < gray.shape[1] + tolerance
                                and -tolerance <= x < gray.shape[0] + tolerance):
                            corner_out_of_bounds = True
                            break
                    if corner_out_of_bounds:
                        continue
                    interpolated_corners[marker_id][frame_index] = corners
        return interpolated_corners


def label_frames(video_path: Path):
    vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
    fps = vr.get_avg_fps()
    frames = []
    for i in range(len(vr)):
        frame = cv2.cvtColor(vr[i].asnumpy(), cv2.COLOR_RGB2BGR)
        # draw_marker(frame)
        frames.append(frame)
    annotator = TrackingAnnotator(aruco_dict, None)
    annotator.annotate_frames(frames)
    # fourcc = cv2.VideoWriterProperties(*'mp4v')  # Codec for video encoding
    video = cv2.VideoWriter('../out/dual-optical-flow.mp4', -1, fps, (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()


# label_frames(Path('../data/RGB_2025-03-05-14_58_10.mp4'))