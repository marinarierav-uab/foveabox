#!/bin/bash

# activate environment
#python("conda activate fovea")

# run test
python "tools/test.py" "configs/exp26_cfg.py" "work_dirs/exp26/epoch_2.pth" --eval bbox --json_out "json/exp26-ep2.json"
python "tools/test.py" "configs/exp26_cfg.py" "work_dirs/exp26/epoch_3.pth" --eval bbox --json_out "json/exp26-ep3.json"

python "tools/test.py" "configs/exp27_cfg.py" "work_dirs/exp27/epoch_5.pth" --eval bbox --json_out "json/exp27-ep5.json"
python "tools/test.py" "configs/exp27_cfg.py" "work_dirs/exp27/epoch_6.pth" --eval bbox --json_out "json/exp27-ep6.json"
#python "tools/test.py" "configs/exp27_cfg.py" "work_dirs/exp27/epoch_7.pth" --eval bbox --json_out "json/exp27-ep7.json"

python "tools/test.py" "configs/exp28_cfg.py" "work_dirs/exp28/epoch_4.pth" --eval bbox --json_out "json/exp28-ep4.json"
python "tools/test.py" "configs/exp28_cfg.py" "work_dirs/exp28/epoch_5.pth" --eval bbox --json_out "json/exp28-ep5.json"

python "tools/test.py" "configs/exp25_cfg.py" "work_dirs/exp25/epoch_5.pth" --eval bbox --json_out "json/exp25-ep5.json"


