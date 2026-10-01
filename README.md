# g1-auslan-rh56

Auslan recognition + Inspire RH56 / Unitree G1 signing.

## Stage 1 (webcam → speech → hands)

```powershell
pip install -r auslan_requirements.txt
# place hand_landmarker.task in this folder if missing
python auslan_app.py --dry-run
python auslan_collect.py
python auslan_train.py --backend rf
python auslan_app.py
```

Also: `hand_mimic_screen.py` (Teams region), `hand_mimic_realsense.py`, `realsense_hand_demo.py`.

## Stage 2

G1 arms + fingers learn by watching humans sign — see Issues.

Hand IPs in app: Left `192.168.123.210`, Right `192.168.123.211`.
