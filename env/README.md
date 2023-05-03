1. Unzip env/candas.zip and env/nupack-4.0.1.7.zip
2. Use the following commands to set up the environment.
```bash
cd env
sudo apt install python3.9
sudo apt install python3.9-venv
python3.9 -m venv can_manuscript
source can_manuscript/bin/activate
pip install --upgrade pip==23.0.1 wheel==0.40.0
pip install -r requirements.txt
```
