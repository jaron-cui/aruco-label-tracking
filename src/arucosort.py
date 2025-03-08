"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import os
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io

import glob
import time
import argparse
from filterpy.kalman import KalmanFilter

np.random.seed(0)


def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i], i] for i in x if i >= 0])  #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
  """
  From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2,x3,y3,x4,y4]
  """

  # NOTE: we're going to simplify here for performance's sake (and because I am lazy) and just treat the bboxes
  #       like orthogonal rectangles
  # print(bb_test.shape, bb_gt.shape)
  # print('bb_oh', bb_test, bb_gt)
  bb_test_x = bb_test[:, [0, 2, 4, 6]]
  bb_test_y = bb_test[:, [1, 3, 5, 7]]
  # print('n', bb_test_x.shape)
  bb_test = np.column_stack([bb_test_x.min(-1), bb_test_y.min(-1), bb_test_x.max(-1), bb_test_y.max(-1)])
  bb_gt_x = bb_gt[:, [0, 2, 4, 6]]
  bb_gt_y = bb_gt[:, [1, 3, 5, 7]]
  bb_gt = np.column_stack([bb_gt_x.min(-1), bb_gt_y.min(-1), bb_gt_x.max(-1), bb_gt_y.max(-1)])
  # print('test', bb_test)
  # print('gt', bb_gt)

  # normal code for orthogonal rectangular bbox iou
  bb_gt = np.expand_dims(bb_gt, 0)
  bb_test = np.expand_dims(bb_test, 1)
  # print(bb_test.shape, bb_gt.shape)

  xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
  yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
  xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
  yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
            + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
  return (o)


def convert_bbox_to_z(bbox):
  return bbox[:8].reshape((8, 1))


def convert_x_to_bbox(x, score=None):
  return (x if score is None else np.concatenate([x, np.array([score])], axis=0)).reshape((1, -1))


class KalmanBoxTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  count = 0

  def __init__(self, bbox):
    """
    Initialises a tracker using initial bounding box.
    """
    # define constant velocity model
    self.kf = KalmanFilter(dim_x=16, dim_z=8)
    pos_vel_dependency = np.tile(np.array([1, 0, 0, 0, 0, 0, 0, 0]), 2)
    vel_dependency = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    self.kf.F = np.array([np.roll(pos_vel_dependency, i) for i in range(8)]
                         + [np.roll(vel_dependency, i) for i in range(8)])
    pos_index = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    self.kf.H = np.array([np.roll(pos_index, i) for i in range(8)])
    # print(self.kf.F, self.kf.H)

    self.kf.R *= 0.001  # kf.R represents sensor uncertainty
    # give high uncertainty to the unobservable initial velocities (kf.P represents state uncertainty)
    self.kf.P[8:, 8:] *= 1000.
    self.kf.P[:8, :8] *= 0.1
    self.kf.Q *= 0.01
    # self.kf.Q[8:, 8:] *= 10

    self.kf.x[:8] = convert_bbox_to_z(bbox)
    self.time_since_update = 0
    # self.id = KalmanBoxTracker.count
    # KalmanBoxTracker.count += 1
    self.id = bbox[-1]
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

  def update(self, bbox):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(convert_bbox_to_z(bbox))

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    # if ((self.kf.x[6] + self.kf.x[2]) <= 0):
    #   self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    if (self.time_since_update > 30):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.x))
    # print(self.kf.x.shape)
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.2):
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if (len(trackers) == 0):
    return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 9), dtype=int)

  iou_matrix = iou_batch(detections, trackers)
  # print(iou_matrix)

  if min(iou_matrix.shape) > 0:
    a = (iou_matrix > iou_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
      matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(-iou_matrix)
  else:
    matched_indices = np.empty(shape=(0, 2))

  unmatched_detections = []
  for d, det in enumerate(detections):
    if (d not in matched_indices[:, 0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if (t not in matched_indices[:, 1]):
      unmatched_trackers.append(t)

  # filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if (iou_matrix[m[0], m[1]] < iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1, 2))
  if (len(matches) == 0):
    matches = np.empty((0, 2), dtype=int)
  else:
    matches = np.concatenate(matches, axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
  def __init__(self, max_age=1, min_hits=3, iou_threshold=0.2):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.iou_threshold = iou_threshold
    self.trackers = []
    self.frame_count = 0

  def update(self, dets=np.empty((0, 9))):
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    # get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers), 9))
    to_del = []
    ret = []
    for t, trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]
      # print(pos.shape)
      trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6], pos[7], 0]
      if np.any(np.isnan(pos)):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

    # update matched trackers with assigned detections
    for m in matched:
      self.trackers[m[1]].update(dets[m[0], :])

    # create and initialise new trackers for unmatched detections
    # print('unmatched', len(unmatched_dets))
    for i in unmatched_dets:
      trk = KalmanBoxTracker(dets[i, :])
      self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
      d = trk.get_state()[0]
      if (trk.time_since_update < 10) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
        ret.append(np.concatenate((d, [trk.id])).reshape(1, -1))
      i -= 1
      # remove dead tracklet
      if (trk.time_since_update > self.max_age):
        # print('removing tracker')
        self.trackers.pop(i)
    if (len(ret) > 0):
      return np.concatenate(ret)
    return np.empty((0, 9))

