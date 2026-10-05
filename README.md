# Robust Vehicle Localization Through GNSS Outages<br><sup>*EKF vs UKF Fusion of IMU, GNSS and Wheel Odometry*</sup>

Ground vehicles such as cars, automated guided vehicles (AGVs) and mobile robots depend on the Global Navigation Satellite System (GNSS) to know where they are. Drive into a dense city and that reliability breaks down. Buildings block satellites, signals arrive only after bouncing off facades, and the receiver can report a position that is confidently wrong by tens of metres before it stops reporting at all. What is left is an inertial measurement unit and a wheel odometer, whose small errors compound into tens of metres within a minute.

This project asks how far and how long a vehicle can hold an accurate position once GNSS degrades or drops out. And a second question that matters just as much: does the filter's own uncertainty stay honest while it does?

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




