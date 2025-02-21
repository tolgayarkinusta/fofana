# Fofana - RoboBoat 2025 Control Software

Autonomous Surface Vehicle (ASV) control software for the RoboBoat 2025 competition. This project integrates ZED2i camera vision processing with CUDA acceleration and PWM motor control for autonomous navigation and task completion.

## Features
- ZED2i stereo camera integration with CUDA 12.6 acceleration
- Real-time object detection and tracking
- Autonomous navigation and obstacle avoidance
- PWM-based dual motor control
- Multi-process task management

## Requirements
- Python 3.8+
- CUDA 12.6
- ZED SDK
- Windows 10

## Project Structure
```
fofana/
├── src/
│   └── fofana/
│       ├── core/        # Core functionality and motor control
│       ├── vision/      # ZED2i camera and vision processing
│       ├── navigation/  # Path planning and obstacle avoidance
│       └── tasks/       # Competition task implementations
└── tests/              # Test suite
```
