# retico-cozmofaceattention

A [retico](https://github.com/retico-team/retico-core) module that gives the Anki Cozmo robot face attention behavior. Cozmo scans for a human face, wanders freely once found, and snaps back to look at the face on a voice trigger.

## How it works

The module runs as a state machine with five modes:

**Scan → Wander → (trigger) → Reposition → Verify → Sweep**

1. **Scan** — Cozmo rotates in place 15° at a time looking for a face via OpenCV Haar cascade. After each full 360° with no detection it backs up 150mm and changes its head angle (cycling through `10° → 20° → 0° → -10°`), handling the case where it is too close for the camera to capture the face.
2. **Wander** — Once a face is found, Cozmo drives and turns randomly to appear natural. On every frame where the face is visible at a safe distance, the robot's exact pose and head angle are silently recorded, keeping the stored location fresh.
3. **Reposition** — When the word **"look"** is heard, Cozmo aborts the current wander movement and navigates to the exact position and heading it was at when it last saw the face: turn to face the target → drive straight there → turn to stored heading → set stored head angle.
4. **Verify** — Cozmo holds position for 2 seconds and checks whether the face is visible. If found, it resumes wandering after a short pause.
5. **Sweep** — If verification fails, a structured local search runs: small turns and head tilts across ±45° horizontal and ±10° vertical. If the stored face was close (face pixel width > 80px), it backs up 150mm first before sweeping. If the sweep exhausts all steps, a full rescan starts.

## Inputs / Outputs


| Direction | IU type               | Source                              |
| --------- | --------------------- | ----------------------------------- |
| Input     | `ImageIU`             | Cozmo camera module                 |
| Input     | `SpeechRecognitionIU` | Whisper ASR module                  |
| Output    | `GenericDictIU`       | Face bounding boxes + last ASR text |


## Dependencies

```
retico-core
retico-whisperasr
opencv-python
pyaudio
cozmo (Anki SDK)
```

## Usage

```python
from retico_cozmofaceattention import FaceDetectionModule

module = FaceDetectionModule(robot=cozmo_robot)
```

Wire it into a retico pipeline with a `CozmoCameraModule` and a `WhisperASRModule` as inputs.

