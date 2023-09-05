# tello
Experiments with the Ryze tello drone.


## Setup

You may need to initially run the script with a direct connection to the internet in order to populate the model `cache` directory.

```bash
virtualenv venv --python=python3.10
. venv/bin/activate
pip install --editable .
```


## Usage

```bash
tello-example --model_name "facebook/detr-resnet-50"
```


## Links
  - [Alternative Firmware](https://github.com/MrJabu/RyzeTelloFirmware)
  - [Awesome List](https://github.com/Matthias84/awesome-tello)
  - [Android APK](https://service-adhoc.dji.com/download/app/android/ba88a046-6f7e-4cbb-a969-27851eb4bbf5)
  - [Python Example](https://github.com/damiafuentes/DJITelloPy/blob/master/examples/record-video.py)
