import cv2
import sys
import os
import random
import numpy as np
import threading
import time

import retico_core
from retico_core import abstract
from retico_core.text import SpeechRecognitionIU
from retico_vision.vision import ImageIU
from retico_core.dialogue import GenericDictIU

_MINT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_COZMO_SDK = os.path.join(_MINT_DIR, "cozmo-python-sdk", "src")
if _COZMO_SDK not in sys.path:
    sys.path.insert(0, _COZMO_SDK)

import math

import cozmo
from cozmo.util import degrees, distance_mm, speed_mmps

_HEAD_MAX          = 25.0
_HEAD_CLOSE_BOOST  = 12.0
_DIST_CLOSE_RATIO  = 0.8
_DIST_FAR_RATIO    = 1.2

# Depth fallback: estimated real face width used with pixel-width depth formula.
_FACE_REAL_WIDTH_MM = 150

# Triangulation: minimum robot displacement between stored observations.
_FACE_OBS_MIN_DIST = 50    # mm
_FACE_OBS_MAX      = 15    # keep this many observations

# Bearing stability: when the robot is closer than this to the stored world
# position, the bearing computed from that position is unreliable (estimate
# error dominates). Fall back to the last stored world-frame bearing instead.
_MIN_BEARING_DIST = 120    # mm


class FaceDetectionModule(retico_core.AbstractModule):
    """Rotates Cozmo while scanning for a face. Once found, wanders randomly.
    During wander, re-estimates the person's world position whenever the face is
    visible — keeps the estimate fresh and corrects odometry drift. On trigger word,
    turns in place to face the person and adjusts head angle. Detects pose
    delocalization via origin_id and re-scans automatically.

    Inputs:
        * ``ImageIU`` — Cozmo camera frames.
        * ``SpeechRecognitionIU`` — trigger word fires orientation.
    """

    @staticmethod
    def name():
        return "Cozmo Face Explorer"

    @staticmethod
    def description():
        return (
            "Scans until a face is detected, wanders with live pose tracking, "
            "and turns to face the person on trigger word."
        )

    @staticmethod
    def input_ius():
        return [ImageIU, SpeechRecognitionIU]

    @staticmethod
    def output_iu():
        return GenericDictIU

    def __init__(self, robot, scan_step_deg=15, **kwargs):
        super().__init__(**kwargs)
        self.robot         = robot
        self.scan_step_deg = scan_step_deg
        self.faceCascade   = None
        self._lock         = threading.Lock()

        # Camera intrinsics — populated in setup() from robot.camera.config.
        self._cam_focal_x  = None
        self._cam_center_x = None

        self._scanning      = False
        self._wandering     = False
        self._repositioning = False

        self._wander_action = None

        # Bearing observations: list of (robot_x, robot_y, world_bearing_rad).
        # Each entry is a ray from a known robot position toward the person.
        # Least-squares intersection of these rays gives the world position.
        self._face_observations = []

        # Person's estimated world position — updated on every face sighting.
        self._face_world_x         = None  # mm
        self._face_world_y         = None  # mm
        self._face_world_bearing   = None  # world-frame bearing (rad) at last sighting
        self._face_last_dist_mm    = None  # estimated distance at last sighting
        self._face_last_head_angle = None  # head angle at last sighting
        self._face_last_origin_id  = None  # robot.pose.origin_id at last sighting

        self._scan_thread   = None
        self._last_asr_text = ""
        self._asr_final     = False

    def setup(self):
        self.faceCascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        cam_cfg = self.robot.camera.config
        self._cam_focal_x  = cam_cfg.focal_length.x
        self._cam_center_x = cam_cfg.center.x

        self._wandering     = False
        self._repositioning = False
        self._start_scan()

    def shutdown(self):
        with self._lock:
            self._scanning      = False
            self._wandering     = False
            self._repositioning = False

    # ------------------------------------------------------------------
    # Scan
    # ------------------------------------------------------------------

    def _start_scan(self):
        with self._lock:
            self._scanning  = True
            self._wandering = False
            # Clear all face knowledge — starting fresh in case of rescan.
            self._face_observations    = []
            self._face_world_x         = None
            self._face_world_y         = None
            self._face_world_bearing   = None
            self._face_last_dist_mm    = None
            self._face_last_head_angle = None
            self._face_last_origin_id  = None
        self._scan_thread = threading.Thread(target=self._scan_loop, daemon=True)
        self._scan_thread.start()
        print("[FaceDetection] Scanning for a face...")

    def _scan_loop(self):
        _HEAD_ANGLES = [10, 20, 0, -10]
        head_idx = 0
        try:
            self.robot.set_head_angle(degrees(_HEAD_ANGLES[head_idx])).wait_for_completed()
        except Exception:
            pass
        steps_per_rotation = max(1, int(360 / self.scan_step_deg))
        steps = 0
        while True:
            with self._lock:
                if not self._scanning:
                    break
            try:
                self.robot.turn_in_place(degrees(self.scan_step_deg)).wait_for_completed()
            except Exception:
                pass
            steps += 1
            if steps >= steps_per_rotation:
                steps = 0
                head_idx = (head_idx + 1) % len(_HEAD_ANGLES)
                print(f"[FaceDetection] Full rotation with no face — backing up, head={_HEAD_ANGLES[head_idx]}°")
                try:
                    self.robot.drive_straight(
                        distance_mm(-150), speed_mmps(80)
                    ).wait_for_completed()
                except Exception:
                    pass
                try:
                    self.robot.set_head_angle(degrees(_HEAD_ANGLES[head_idx])).wait_for_completed()
                except Exception:
                    pass
            time.sleep(0.05)

    # ------------------------------------------------------------------
    # Wander
    # ------------------------------------------------------------------

    def _wander_loop(self):
        while True:
            with self._lock:
                if not self._wandering:
                    break
            dist  = random.uniform(30, 80)
            angle = random.uniform(-90, 90)
            try:
                action = self.robot.drive_straight(distance_mm(dist), speed_mmps(80))
                with self._lock:
                    self._wander_action = action
                action.wait_for_completed()
            except Exception:
                pass
            finally:
                with self._lock:
                    self._wander_action = None
            with self._lock:
                if not self._wandering:
                    break
            try:
                action = self.robot.turn_in_place(degrees(angle))
                with self._lock:
                    self._wander_action = action
                action.wait_for_completed()
            except Exception:
                pass
            finally:
                with self._lock:
                    self._wander_action = None
            time.sleep(0.1)

    # ------------------------------------------------------------------
    # Face knowledge storage — triangulation
    # ------------------------------------------------------------------

    def _triangulate_face_position(self):
        """Least-squares intersection of bearing rays stored in _face_observations.

        Each observation (rx, ry, alpha) is a ray from (rx, ry) in world direction
        alpha. Minimises sum of squared perpendicular distances to find the point
        that best fits all rays. Returns (None, None) when degenerate (rays parallel).
        Must be called under _lock.
        """
        obs = self._face_observations
        if len(obs) < 2:
            return None, None

        A = np.zeros((2, 2))
        b = np.zeros(2)
        for (px, py, alpha) in obs:
            d = np.array([math.cos(alpha), math.sin(alpha)])
            M = np.eye(2) - np.outer(d, d)
            p = np.array([px, py])
            A += M
            b += M @ p

        det = abs(float(np.linalg.det(A)))
        if det < 0.01:
            # Rays are nearly parallel — no useful lateral baseline yet.
            return None, None

        P = np.linalg.solve(A, b)

        # The solution must be in front of the most recent observation ray.
        px0, py0, a0 = obs[-1]
        fwd = (P[0] - px0) * math.cos(a0) + (P[1] - py0) * math.sin(a0)
        if fwd < 50:
            return None, None

        return float(P[0]), float(P[1])

    def _store_face_knowledge(self, face_box):
        """Must be called under _lock.

        Converts the face bounding box into a world-position estimate using:
        1. The real camera intrinsics (focal length, principal point) from the SDK.
        2. Multi-view triangulation when the robot has moved enough laterally.
        3. Pixel-width depth as a fallback when triangulation is degenerate.
        """
        focal_x  = self._cam_focal_x
        center_x = self._cam_center_x

        x, _y, w, _h = face_box
        face_cx = x + w / 2

        # Horizontal bearing correction: face offset from image center → angle offset.
        # Right in image = clockwise in world (decreasing yaw) → subtract.
        horiz_offset_rad = math.atan((face_cx - center_x) / focal_x)
        yaw           = self.robot.pose.rotation.angle_z.radians
        world_bearing = yaw - horiz_offset_rad

        # Pixel-width depth estimate — used as fallback only.
        dist_mm_depth = (_FACE_REAL_WIDTH_MM * focal_x) / max(w, 1)

        rx = self.robot.pose.position.x
        ry = self.robot.pose.position.y

        # Append new observation if the robot has moved far enough since the last one.
        if not self._face_observations:
            self._face_observations.append((rx, ry, world_bearing))
        else:
            lx, ly, _ = self._face_observations[-1]
            if math.hypot(rx - lx, ry - ly) >= _FACE_OBS_MIN_DIST:
                self._face_observations.append((rx, ry, world_bearing))
                if len(self._face_observations) > _FACE_OBS_MAX:
                    self._face_observations.pop(0)

        # Prefer triangulated position; fall back to single-view depth estimate.
        tx, ty = self._triangulate_face_position()
        if tx is not None:
            tri_dist = math.hypot(tx - rx, ty - ry)
            if 100 < tri_dist < 6000:
                self._face_world_x      = tx
                self._face_world_y      = ty
                self._face_last_dist_mm = tri_dist
                print(
                    f"[FaceDetection] Triangulated position ({tx:.0f}, {ty:.0f})mm  "
                    f"dist={tri_dist:.0f}mm  obs={len(self._face_observations)}"
                )
            else:
                # Triangulation gave an implausible result — use depth estimate.
                self._face_world_x      = rx + dist_mm_depth * math.cos(world_bearing)
                self._face_world_y      = ry + dist_mm_depth * math.sin(world_bearing)
                self._face_last_dist_mm = dist_mm_depth
        else:
            self._face_world_x      = rx + dist_mm_depth * math.cos(world_bearing)
            self._face_world_y      = ry + dist_mm_depth * math.sin(world_bearing)
            self._face_last_dist_mm = dist_mm_depth

        self._face_world_bearing   = world_bearing
        self._face_last_head_angle = round(self.robot.head_angle.degrees, 1)
        self._face_last_origin_id  = self.robot.pose.origin_id

    # ------------------------------------------------------------------
    # Orient to face on trigger
    # ------------------------------------------------------------------

    def _orient_to_face(self):
        """Turn in place toward the person's world position and set head angle. No driving."""
        with self._lock:
            self._repositioning = True
            self._wandering     = False
            target_x      = self._face_world_x
            target_y      = self._face_world_y
            stored_bearing = self._face_world_bearing
            last_dist      = self._face_last_dist_mm
            last_head      = self._face_last_head_angle
            stored_orig    = self._face_last_origin_id

        with self._lock:
            action = self._wander_action
            self._wander_action = None
        if action is not None:
            try:
                action.abort()
            except Exception:
                pass
            time.sleep(0.2)

        # Delocalization check: pose.origin_id increments on pickup/slip.
        # The stored world position is in the old frame — must rescan.
        if stored_orig is not None and self.robot.pose.origin_id != stored_orig:
            print("[FaceDetection] Pose delocalized since last face sighting — rescanning")
            with self._lock:
                self._repositioning = False
            self._start_scan()
            return

        print(f"[FaceDetection] Orienting to person at world ({target_x:.0f}, {target_y:.0f})mm")
        try:
            cur = self.robot.pose
            dx  = target_x - cur.position.x
            dy  = target_y - cur.position.y
            cur_dist = math.hypot(dx, dy)

            if cur_dist < _MIN_BEARING_DIST:
                # The robot is very close to the stored estimate — the estimate's
                # depth error would make the bearing flip. Use the world-frame
                # bearing from the last reliable sighting instead.
                bearing = stored_bearing
                print(
                    f"[FaceDetection] cur_dist={cur_dist:.0f}mm < {_MIN_BEARING_DIST}mm — "
                    f"using stored bearing {math.degrees(bearing):.1f}°"
                )
            else:
                bearing = math.atan2(dy, dx)

            cur_yaw = cur.rotation.angle_z.radians
            turn_d  = (bearing - cur_yaw + math.pi) % (2 * math.pi) - math.pi
            self.robot.turn_in_place(degrees(math.degrees(turn_d))).wait_for_completed()

            # Head angle: compare current distance to distance at last face sighting.
            ratio = (cur_dist / last_dist) if (last_dist and last_dist > 0) else 1.0
            if ratio < _DIST_CLOSE_RATIO:
                head_angle = min(_HEAD_MAX, (last_head or 0.0) + _HEAD_CLOSE_BOOST)
            elif ratio > _DIST_FAR_RATIO:
                head_angle = 0.0
            else:
                head_angle = last_head if last_head is not None else 0.0

            self.robot.set_head_angle(degrees(head_angle)).wait_for_completed()
            print(
                f"[FaceDetection] Oriented — bearing={math.degrees(bearing):.1f}°  "
                f"cur_dist={cur_dist:.0f}mm  last_dist={last_dist:.0f}mm  "
                f"ratio={ratio:.2f}  head={head_angle:.1f}°"
            )
        except Exception as e:
            print(f"[FaceDetection] Orient error: {e}")

        # Pause so the user can register that Cozmo is looking at them.
        time.sleep(3)

        with self._lock:
            self._repositioning = False
            self._wandering     = True
        threading.Thread(target=self._wander_loop, daemon=True).start()

    # ------------------------------------------------------------------
    # Face confirmed at scan time
    # ------------------------------------------------------------------

    def _confirm_face(self, faces, input_iu):
        largest_face = max(faces, key=lambda f: f[2])
        with self._lock:
            self._scanning  = False
            self._wandering = True
            self._store_face_knowledge(face_box=largest_face)
            world_x = self._face_world_x
            world_y = self._face_world_y
            dist    = self._face_last_dist_mm
            head    = self._face_last_head_angle

        threading.Thread(target=self._wander_loop, daemon=True).start()

        face_w = int(largest_face[2])
        print(f"\n[FaceDetection] *** Face found! ***")
        print(
            f"[FaceDetection] Person at world ({world_x:.0f}, {world_y:.0f})mm  "
            f"dist≈{dist:.0f}mm  head={head}°  face_px={face_w}"
        )

        face_dict = {
            i: {"box": [(int(x), int(y)), (int(x + w), int(y + h))]}
            for i, (x, y, w, h) in enumerate(faces)
        }
        face_dict["speech_input"] = self._speech_snapshot()
        output_iu = self.create_iu(input_iu)
        output_iu.set_payload(face_dict)
        return output_iu

    # ------------------------------------------------------------------
    # retico core
    # ------------------------------------------------------------------

    def process_update(self, update_message):
        for iu, um in update_message:
            if isinstance(iu, SpeechRecognitionIU):
                if um == abstract.UpdateType.COMMIT:
                    self._process_asr_iu(iu)
                continue
            if um != abstract.UpdateType.ADD:
                continue
            if isinstance(iu, ImageIU):
                output_iu = self._process_image_iu(iu)
                if output_iu is not None:
                    return retico_core.UpdateMessage.from_iu(
                        output_iu, retico_core.UpdateType.ADD
                    )
        return None

    def _process_asr_iu(self, iu):
        trigger = False
        with self._lock:
            self._last_asr_text = iu.text
            self._asr_final     = iu.final
            if (
                "look" in iu.text.lower()
                and self._face_world_x is not None
                and not self._repositioning
            ):
                trigger = True

        if trigger:
            print("[FaceDetection] Trigger heard — orienting to face")
            threading.Thread(target=self._orient_to_face, daemon=True).start()

    def _speech_snapshot(self):
        with self._lock:
            return {"last_asr_text": self._last_asr_text, "asr_final": self._asr_final}

    def _process_image_iu(self, input_iu):
        with self._lock:
            if self._repositioning:
                return None
            is_scanning  = self._scanning
            is_wandering = self._wandering

        if not is_scanning and not is_wandering:
            return None

        image = np.asarray(input_iu.image)
        gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        display = image.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(display, (x, y), (x + w, y + h), (250, 0, 180), 2)
        cv2.imshow("Cozmo Camera", display)
        cv2.waitKey(1)

        if is_wandering:
            # Opportunistically update the face world position whenever the face is
            # visible during wander. New observation added only when robot has moved
            # far enough to improve the triangulation baseline.
            if len(faces) > 0:
                largest_face = max(faces, key=lambda f: f[2])
                with self._lock:
                    self._store_face_knowledge(face_box=largest_face)
            return None

        if is_scanning:
            if len(faces) > 0:
                return self._confirm_face(faces, input_iu)
            return None

        return None
