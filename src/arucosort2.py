import numpy as np
import cv2
from filterpy.kalman import KalmanFilter
from shapely.geometry import Polygon


def polygon_iou(poly1, poly2):
  """
  Compute IoU for two polygons using Shapely.
  """
  poly1 = Polygon(poly1.reshape(4, 2))
  poly2 = Polygon(poly2.reshape(4, 2))
  if not poly1.is_valid or not poly2.is_valid:
    return 0.0
  intersection = poly1.intersection(poly2).area
  union = poly1.union(poly2).area
  return intersection / union if union > 0 else 0


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
  """
  Assigns detections to tracked objects based on polygon IoU.
  """
  if len(trackers) == 0:
    return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

  iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
  for d, det in enumerate(detections):
    for t, trk in enumerate(trackers):
      iou_matrix[d, t] = polygon_iou(det[:8], trk[:8])

  from scipy.optimize import linear_sum_assignment
  x, y = linear_sum_assignment(-iou_matrix)
  matches = np.array([[x[i], y[i]] for i in range(len(x)) if iou_matrix[x[i], y[i]] > iou_threshold])
  # print(matches.shape)
  unmatched_detections = [d for d in range(len(detections)) if matches.size > 0 and d not in matches[:, 0]]
  unmatched_trackers = [t for t in range(len(trackers)) if matches.size > 0 and t not in matches[:, 1]]

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class KalmanPolygonTracker:
  """
  Kalman Filter tracker for ArUco markers (using four corner points instead of bounding boxes).
  """
  count = 0

  def __init__(self, bbox):
    self.kf = KalmanFilter(dim_x=16, dim_z=8)  # 4 points * (x, y)
    self.kf.F = np.eye(16)  # Constant velocity model
    self.kf.H = np.eye(8, 16)

    self.kf.R *= 10
    self.kf.P *= 10
    self.kf.Q *= 0.01

    self.kf.x[:8] = bbox.reshape((8, 1))
    self.time_since_update = 0
    self.id = KalmanPolygonTracker.count
    KalmanPolygonTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

  def update(self, bbox):
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(bbox.reshape((8, 1)))

  def predict(self):
    self.kf.predict()
    self.age += 1
    if self.time_since_update > 0:
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(self.kf.x[:8].reshape((4, 2)))
    return self.history[-1]

  def get_state(self):
    return np.append(self.kf.x[:8].reshape((4, 2)), self.id)


class ArucoSort:
  """
  SORT-based tracking for ArUco markers using quadrilateral shapes.
  """

  def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
    self.max_age = max_age
    self.min_hits = min_hits
    self.iou_threshold = iou_threshold
    self.trackers = []
    self.frame_count = 0

  def update(self, detections=np.empty((0, 9))):
    self.frame_count += 1
    trks = np.zeros((len(self.trackers), 9))
    to_del = []
    ret = []

    for t, trk in enumerate(self.trackers):
      pos = trk.get_state().flatten()
      trks[t, :] = pos
      if np.any(np.isnan(pos)):
        to_del.append(t)

    trks = np.delete(trks, to_del, axis=0)
    self.trackers = [self.trackers[i] for i in range(len(self.trackers)) if i not in to_del]

    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(detections, trks, self.iou_threshold)

    for m in matched:
      self.trackers[m[1]].update(detections[m[0], :8])

    for i in unmatched_dets:
      trk = KalmanPolygonTracker(detections[i, :8])
      self.trackers.append(trk)

    for trk in self.trackers:
      d = trk.get_state().flatten()
      if trk.time_since_update < 1 and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
        ret.append(d)

    self.trackers = [trk for trk in self.trackers if trk.time_since_update <= self.max_age]

    return np.array(ret) if len(ret) > 0 else np.empty((0, 9))