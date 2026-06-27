# MAR dynamic threshold and red warning cards

Changes in this build:

1. MAR threshold is no longer forced to the same 0.12 floor for most users.
   - MAR is computed from floating-point face landmarks instead of rounded integer pixels.
   - DynamicMAR uses each user's calibration baseline plus a personalized margin.
   - The safety floor is reduced to 0.05, with `mar_gap=0.04` by default.

2. The UI highlights abnormal features in red.
   - EAR card turns red when `EAR < EAR_threshold`.
   - MAR card turns red when `MAR > MAR_threshold`.
   - PERCLOS, blink, yawn, closed frames, pitch, and head nod cards also turn red when their warning condition is active.

3. The mobile title is simplified to `Driver Drowsiness Monitor`.

Calibration note: keep the mouth naturally closed during the first calibration frames. If the user speaks or opens the mouth during calibration, press `Reset calibration` and calibrate again.
