#!/bin/bash
echo "FILES: "
ls
python3 -m pip install requests
python3 -m pip install PIL
pip install --upgrade pip
pip install -r /data/requirements.txt
python3 /data/train_parallel.py
chown $USERID:$USERID -R /data
