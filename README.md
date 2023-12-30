# AI in Logistics: Container Number Recognition

[[`Project Writeup`](https://medium.com/@jonathanlawhh) [`My Website`](https://jonathanlawhh.com/)]

## Project Overview
![AI in Logistics: Container Number Recognition header image](/assets/AI%20in%20Logistics%20Container%20Number%20Recognition%20Header.jpg)
Traditional container tracking often relies on manual scans and tedious paperwork, creating inefficiencies and bottlenecks.
This project leverages Optical Character Recognition (OCR) technology to automatically read container numbers directly from images, offering innovation in logistics management.

Companies using this AI solution can now enjoy real-time visibility into container movement within their premises.

## References
- [Azure AI Vision](https://azure.microsoft.com/en-us/products/ai-services/ai-vision) by Microsoft Azure
- [OpenCV](https://opencv.org/)

## Setup and Usage

### Software Requirements
- Python >= 3.10
- [Microsoft Azure Vision API](https://azure.microsoft.com/en-us/products/ai-services/ai-vision) API keys

### Installation

1. Clone this repository:
```bash
git clone https://github.com/jonathanlawhh/container-number-recognition-ai.git
```
2. Install required libraries:
```bash
pip install -R requirements.txt
```

### Usage

1. Place your container images in the .\data\ folder.
2. Rename `.env-sample` to `.env`
3. Fill up both values in .env `VISION_ENDPOINT` and `VISION_KEY` from your Microsoft Azure Vision API project.
4. Run the script.
```bash
python main.py
```

## Closing thoughts

- Using a ready built service such as Azure Vision AI offloads most of the image processing task
- Azure Vision API is more reliable than building using Tesseract OCR if the environment is dynamic, performance is more consistent compared to running on a local hardware
- Can be integrated with in-house Transport Management Systems