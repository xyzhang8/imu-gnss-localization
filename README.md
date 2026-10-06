# Robust Vehicle Localization Through GNSS Outages<br><sup>*EKF vs UKF Fusion of IMU, GNSS and Wheel Odometry*</sup>

Ground vehicles such as cars, automated guided vehicles (AGVs) and mobile robots
depend on the Global Navigation Satellite System (GNSS) to know where they are.
Most of the time it simply works: on open roads a consumer receiver holds position
to a metre or two, continuously. The difficulty is that this depends entirely on
the surroundings, and changes without warning. Move between tall buildings, under
a bridge, or into a garage, and satellites are blocked, signals arrive only after
bouncing off facades, and the receiver can report a position that is confidently
wrong well before it stops reporting at all. Of the three sequences used here, two
show no degradation at all; in the third the fix is missing for 98% of the run.

When GNSS goes, what is left is an inertial measurement unit and a wheel odometer,
whose errors compound quickly once there is nothing to correct them.

This project asks how far and how long a vehicle can hold an accurate position once GNSS degrades or drops out. And a second question that matters just as much: does the filter's own uncertainty stay honest while it does?

A note on what is being claimed. While GNSS is available, the filter does not beat a raw GNSS fix by much on position, and it is not meant to. The claim is narrower and harder: that position stays usable once the fix degrades or stops, and that the uncertainty the filter reports still describes the error it is actually making.

## Status

Work in progress

## Why this problem matters 

## Dataset: NavINST 

For this project we make use of the NavINST dataset[^1].

## Methods

## Results

## Limitations

<!-- Placeholder, to be written properly. Points to cover:
     - The NavINST indoor garage sequences ship no GNSS data at all (verified:
       no gnss.bag in any of Indoor01-05, and their reference.bag carries only
       the LiDAR map registration pose). Synthesizing GNSS from the reference
       was rejected as circular, so the urban sequences are used instead.
     - Synthetic outages remove good measurements cleanly, which is kinder than
       the real degradation that precedes a real outage.
     - 2D planar state, so no gravity compensation and no 3D attitude.
     - Accelerometer bias is not estimated; see the model-order argument.
-->

## How to reproduce

## Citation and License

[^1]: Araujo, P., Mounier, E., Bader, Q., Dawson, E., Kaoud Abdelaziz, S., Zekry, A., Elhabiby, M., & Noureldin, A. (2025). The NavINST Dataset for Multi-Sensor Autonomous Navigation. *IEEE Access*. https://doi.org/10.1109/ACCESS.2025.3565878




