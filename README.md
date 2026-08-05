# i-Blink: AI-Assisted Eye Blink Interaction System

iBlink is an AI-powered accessibility system that enables hands-free computer interaction through eye blink-based commands. It seeks to provide an alternative input method for individuals who may experience difficulties using traditional peripherals such as keyboards and microphones to communicate    
Using real-time computer vision, iBlink detects and interprets eye blink patterns captured through a camera and translates them into meaningful user actions. The system focuses on creating a more natural and accessible human-computer interaction experience by leveraging vision-based input.

>_Ideal for motor-impaired or paralysed users who cannot use traditional methods to communicate, by providing affordable and accessible assistive technology that enables hands-free communication and digital interaction._

## Contributers
>Sagnik Ghosh - Computer Vision Pipeline, Blink Detection   
>Kopal Kumar - Project Concept, SLM Integration, System Improvements

## Features

- Real‑time face and eye landmark detection  
- Eye Aspect Ratio (EAR) for blink detection  
- On‑screen gaze keyboard for character selection   
- Runs with a standard webcam (no extra hardware)  
- Calibration for personalized gaze mapping


## Technology Stack

| Technology | Purpose |
|------------|---------|
| Python | Core language |
| OpenCV | Video capture & drawing UI |
| InsightFace | Face + 106 landmark detection |
| Numpy | Computation |
| pyttsx3 | Text‑to‑speech |
| Google Gemma | SLM |

## Work in Progress

- SLM integration
- User Friendly UI
- Audio feedback with text‑to‑speech 

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/<YOUR_USERNAME>/eye-gaze-blink-keyboard.git
   cd eye-gaze-blink-keyboard
