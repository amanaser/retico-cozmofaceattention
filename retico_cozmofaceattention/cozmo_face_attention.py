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

# Local sweep positions after repositioning: (turn_delta_deg, head_offset_from_base_deg).
# turn_delta is relative to previous position; head_offset is absolute from stored base angle.
# Covers ±45° horizontal, ±10° vertical before giving up to full rescan.
_SWEEP_STEPS = [
    (-15,   0),   # 15° left,  base head
    (-15,   0),   # 30° left,  base head
    (-15,   0),   # 45° left,  base head
    (+45,  10),   # center,    head up
    (+30,  10),   # 30° right, head up
    (  0, -10),   # 30° right, head down
    (-30, -10),   # center,    head down
    (-30, -10),   # 30° left,  head down
]
_HEAD_MIN = -25.0
_HEAD_MAX  =  25.0
# Only store opportunistic poses where the face appears at a reasonable distance.
# Frames where the face is wider than this threshold mean Cozmo is too close;
# returning to that stored position would put the face out of frame again.
_MAX_OPPORTUNISTIC_FACE_PX = 100

CLOSE_THRESHOLD_PX  = 80     # face wider than this at storage time → too close; back up during sweep
PROXIMITY_BACKUP_MM = 150.0  # distance to reverse before sweeping when too close


class FaceDetectionModule(retico_core.AbstractModule):
    """Rotates Cozmo while scanning for a face. Once found, wanders randomly.
    While wandering, opportunistically updates the stored face pose whenever the
    face is visible, keeping it fresh. On "look", navigates back to the most
    recent stored pose and verifies. If the face is not immediately visible, runs
    a local sweep (small turns + head tilts) before falling back to full rescan.

    Inputs:
        * ``ImageIU`` — Cozmo camera frames.
        * ``SpeechRecognitionIU`` — "look" triggers repositioning.
    """

    @staticmethod
    def name():
        return "Cozmo Face Explorer"

    @staticmethod
    def description():
        return (
            "Scans until a face is detected, wanders with live pose tracking, "
            "and returns to the last known face pose on 'look'."
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

        # --- state flags ---
        self._scanning      = False   # rotation scan loop active
        self._wandering     = False   # random drive loop active
        self._repositioning = False   # go_to_pose in progress
        self._verifying     = False   # camera check after reposition
        self._sweeping      = False   # local sweep search active
        self._sweep_moving  = False   # within sweep: robot currently moving

        self._verify_deadline = 0.0
        self._sweep_found     = threading.Event()   # set by _confirm_face
        self._wander_action   = None                # current wander SDK action for abort

        # --- stored face knowledge ---
        self._last_face_pose     = None   # cozmo.util.Pose
        self._last_head_angle    = None   # float degrees
        self._last_face_width_px = None   # int pixels — distance proxy

        self._scan_thread   = None
        self._last_asr_text = ""
        self._asr_final     = False

    def setup(self):
        self.faceCascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._wandering     = False
        self._repositioning = False
        self._verifying     = False
        self._sweeping      = False
        self._sweep_moving  = False
        self._verify_deadline = 0.0
        self._start_scan()

    def shutdown(self):
        with self._lock:
            self._scanning      = False
            self._wandering     = False
            self._repositioning = False
            self._verifying     = False
            self._sweeping      = False
            self._sweep_moving  = False

    # ------------------------------------------------------------------
    # Scan
    # ------------------------------------------------------------------

    def _start_scan(self):
        with self._lock:
            self._scanning     = True
            self._wandering    = False
            self._verifying    = False
            self._sweeping     = False
            self._sweep_moving = False
        self._scan_thread = threading.Thread(target=self._scan_loop, daemon=True)
        self._scan_thread.start()
        print("[FaceDetection] Scanning for a face...")

    def _scan_loop(self):
        _HEAD_ANGLES = [10, 20, 0, -10]   # cycle through on each backup
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
    # Pose storage
    # ------------------------------------------------------------------

    def _store_face_pose(self, face_width_px=None):
        """Must be called under _lock."""
        self._last_face_pose  = self.robot.pose
        self._last_head_angle = round(self.robot.head_angle.degrees, 1)
        if face_width_px is not None:
            self._last_face_width_px = face_width_px

    # ------------------------------------------------------------------
    # Reposition
    # ------------------------------------------------------------------

    def _reposition_and_verify(self):
        with self._lock:
            self._repositioning = True
            self._wandering     = False
            self._sweeping      = False
            self._verifying     = False
            target_pose = self._last_face_pose
            target_head = self._last_head_angle

        # Abort the current wander action so the robot is free for new commands.
        with self._lock:
            action = self._wander_action
            self._wander_action = None
        if action is not None:
            try:
                action.abort()
            except Exception:
                pass
            time.sleep(0.2)

        print(f"[FaceDetection] Returning to stored pose, head={target_head}°")
        try:
            cur  = self.robot.pose
            dx   = target_pose.position.x - cur.position.x
            dy   = target_pose.position.y - cur.position.y
            dist = math.sqrt(dx * dx + dy * dy)

            if dist > 20:
                # Turn to face the target XY, then drive straight there
                face_rad   = math.atan2(dy, dx)
                cur_yaw    = cur.rotation.angle_z.radians
                turn_delta = (face_rad - cur_yaw + math.pi) % (2 * math.pi) - math.pi
                self.robot.turn_in_place(degrees(math.degrees(turn_delta))).wait_for_completed()
                self.robot.drive_straight(distance_mm(dist), speed_mmps(150)).wait_for_completed()

            # Turn to the stored body heading
            target_yaw    = target_pose.rotation.angle_z.radians
            cur_yaw       = self.robot.pose.rotation.angle_z.radians
            heading_delta = (target_yaw - cur_yaw + math.pi) % (2 * math.pi) - math.pi
            self.robot.turn_in_place(degrees(math.degrees(heading_delta))).wait_for_completed()

            # Set stored head tilt
            self.robot.set_head_angle(degrees(target_head)).wait_for_completed()
        except Exception as e:
            print(f"[FaceDetection] Reposition error: {e}")

        with self._lock:
            self._repositioning   = False
            self._verifying       = True
            self._verify_deadline = time.time() + 2.0

        print("[FaceDetection] Verifying face at stored position...")

    # ------------------------------------------------------------------
    # Local sweep
    # ------------------------------------------------------------------

    def _sweep_search(self):
        """Small-angle search before falling back to full rotation rescan."""
        with self._lock:
            self._verifying    = False
            self._sweeping     = True
            self._sweep_moving = True
            base_head          = self._last_head_angle
            face_width_px      = self._last_face_width_px

        print("[FaceDetection] Starting local sweep search...")

        if face_width_px is not None and face_width_px > CLOSE_THRESHOLD_PX:
            print(f"[FaceDetection] Sweep: stored face close ({face_width_px}px) — backing up {PROXIMITY_BACKUP_MM:.0f}mm")
            try:
                self.robot.drive_straight(
                    distance_mm(-PROXIMITY_BACKUP_MM), speed_mmps(80)
                ).wait_for_completed()
            except Exception as e:
                print(f"[FaceDetection] Sweep backup error: {e}")
            self._sweep_found.clear()
            with self._lock:
                self._sweep_moving = False
            if self._sweep_found.wait(timeout=0.8):
                return

        for turn_d, head_off in _SWEEP_STEPS:
            with self._lock:
                if not self._sweeping:   # face already found by image processing
                    return
                self._sweep_moving = True

            try:
                if abs(turn_d) > 0:
                    self.robot.turn_in_place(degrees(turn_d)).wait_for_completed()
                target_head = max(_HEAD_MIN, min(_HEAD_MAX, base_head + head_off))
                self.robot.set_head_angle(degrees(target_head)).wait_for_completed()
            except Exception as e:
                print(f"[FaceDetection] Sweep move error: {e}")

            # Clear event BEFORE enabling detection to avoid a race where
            # the previous iteration's event fires into the new window.
            self._sweep_found.clear()
            with self._lock:
                self._sweep_moving = False

            if self._sweep_found.wait(timeout=0.8):
                return   # _confirm_face was called by _process_image_iu

        with self._lock:
            self._sweeping = False
        print("[FaceDetection] Sweep search failed — rescanning")
        self._start_scan()

    # ------------------------------------------------------------------
    # Shared face-confirmation helper
    # ------------------------------------------------------------------

    def _confirm_face(self, faces, input_iu, start_wander):
        largest_w = max(f[2] for f in faces)
        with self._lock:
            self._scanning     = False
            self._verifying    = False
            self._sweeping     = False
            self._sweep_moving = False
            self._wandering    = start_wander
            self._store_face_pose(face_width_px=largest_w)
            head = self._last_head_angle
            pose = self._last_face_pose

        # Unblock sweep thread (harmless when not sweeping).
        self._sweep_found.set()

        if start_wander:
            threading.Thread(target=self._wander_loop, daemon=True).start()
        else:
            # After confirming via "look", pause briefly then resume wandering.
            def _delayed_wander():
                time.sleep(3)
                with self._lock:
                    if not self._wandering and not self._repositioning and not self._scanning:
                        self._wandering = True
                threading.Thread(target=self._wander_loop, daemon=True).start()
            threading.Thread(target=_delayed_wander, daemon=True).start()

        label = "scan" if start_wander else "look"
        print(f"\n[FaceDetection] *** Face confirmed ({label})! ***")
        print(f"[FaceDetection] Pose x={pose.position.x:.1f}  y={pose.position.y:.1f}  "
              f"heading={pose.rotation.angle_z.degrees:.1f}°  head={head}°  "
              f"face_px={largest_w}")

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
                and self._last_face_pose is not None
                and not self._repositioning
            ):
                trigger = True

        if trigger:
            print("[FaceDetection] 'look' heard — repositioning to last face pose")
            threading.Thread(target=self._reposition_and_verify, daemon=True).start()

    def _speech_snapshot(self):
        with self._lock:
            return {"last_asr_text": self._last_asr_text, "asr_final": self._asr_final}

    def _process_image_iu(self, input_iu):
        with self._lock:
            if self._repositioning:
                return None
            is_scanning    = self._scanning
            is_wandering   = self._wandering
            is_verifying   = self._verifying
            is_sweeping    = self._sweeping
            sweep_moving   = self._sweep_moving
            deadline       = self._verify_deadline

        active = is_scanning or is_wandering or is_verifying or is_sweeping
        if not active:
            return None

        image = np.asarray(input_iu.image)

        # During a sweep move: just keep the display live, skip detection.
        if is_sweeping and sweep_moving:
            cv2.imshow("Cozmo Camera", image)
            cv2.waitKey(1)
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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

        # --- WANDER: opportunistic tracking, no state change ---
        if is_wandering:
            if len(faces) > 0:
                largest_w = max(f[2] for f in faces)
                if largest_w <= _MAX_OPPORTUNISTIC_FACE_PX:
                    with self._lock:
                        self._store_face_pose(face_width_px=largest_w)
            return None

        # --- SCAN: first (or re-)detection ---
        if is_scanning:
            if len(faces) > 0:
                return self._confirm_face(faces, input_iu, start_wander=True)
            return None

        # --- VERIFY: quick check after go_to_pose ---
        if is_verifying:
            if len(faces) > 0:
                return self._confirm_face(faces, input_iu, start_wander=False)
            if time.time() > deadline:
                print("[FaceDetection] Face not visible — starting local sweep")
                threading.Thread(target=self._sweep_search, daemon=True).start()
            return None

        # --- SWEEP: checking at current sweep position ---
        if is_sweeping:
            if len(faces) > 0:
                return self._confirm_face(faces, input_iu, start_wander=False)
            return None

        return None
