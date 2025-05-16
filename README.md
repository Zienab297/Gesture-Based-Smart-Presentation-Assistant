# Face and Hand Gesture-Controlled Presentation System

## Overview

This project implements a presentation control system using a combination of real-time face recognition, emotion analysis, and hand gesture tracking. The system ensures that only authorized users can interact with the presentation. Gesture-based controls allow for navigation and annotation of slides, while emotion detection adds an additional layer of interaction management by pausing the system when confusion or negative emotions are detected.

## Features

### Face Recognition
- Authenticates users using facial recognition via the DeepFace library.
- Each authorized user is associated with a folder containing sample facial images.
- Access is restricted to authorized users only.

### Emotion Detection
- Analyzes facial expressions to detect dominant emotions such as surprise, fear, or confusion.
- System pauses input upon detecting such emotions to prevent unintended gestures or navigation.

### Hand Gesture Control
- Uses a hand tracking module to recognize gestures and control slides accordingly.
- Supported gestures:
  - **Single finger up**: Previous slide
  - **Pinky finger up**: Next slide
  - **All fingers down**: Go to first slide
  - **All fingers up**: Go to last slide
  - **Index and middle fingers up**: Show pointer
  - **Index finger up**: Start/continue annotation
  - **Three middle fingers up**: Erase the last annotation

## Requirements

- Python 3.7+
- OpenCV
- DeepFace
- MediaPipe
- NumPy
- Custom `HandTracker` module
