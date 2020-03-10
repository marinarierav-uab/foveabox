#!/bin/bash

# generate csv
python "tools/csv_generator.py" --original data/CVC-VideoClinicDBtrain_valid/annotations/test.json --json json/0217-exp27-ep10.bbox.json --out 0217-exp27-ep10
python "tools/csv_generator.py" --original data/CVC-VideoClinicDBtrain_valid/annotations/test.json --json json/0217-exp27-ep15.bbox.json --out 0217-exp27-ep15
python "tools/csv_generator.py" --original data/CVC-VideoClinicDBtrain_valid/annotations/test.json --json json/0217-exp27-ep20.bbox.json --out 0217-exp27-ep20
python "tools/csv_generator.py" --original data/CVC-VideoClinicDBtrain_valid/annotations/test.json --json json/0217-exp27-ep25.bbox.json --out 0217-exp27-ep25

# calculate average results
python "tools/challenge_validation.py" --team 0217-exp27-ep10 --list
python "tools/challenge_validation.py" --team 0217-exp27-ep15 --list
python "tools/challenge_validation.py" --team 0217-exp27-ep20 --list
python "tools/challenge_validation.py" --team 0217-exp27-ep25 --list



python "tools/csv_generator.py" --original data/CVC-VideoClinicDBtrain_valid/annotations/test.json --json json/0217-exp31-ep20.bbox.json --out 0217-exp31-ep20
python "tools/csv_generator.py" --original data/CVC-VideoClinicDBtrain_valid/annotations/test.json --json json/0217-exp32-ep20.bbox.json --out 0217-exp32-ep20
python "tools/challenge_validation.py" --team 0217-exp31-ep20 --list
python "tools/challenge_validation.py" --team 0217-exp32-ep20 --list





python "tools/csv_generator.py" --original data/CVC-VideoClinicDBtrain_valid/annotations/test.json --json work_dirs/exp33/0217-exp33-ep20.bbox.json --out 0217-exp33-ep20
python "tools/challenge_validation.py" --team 0217-exp33-ep20 --list

python "tools/csv_generator.py" --original data/CVC-VideoClinicDBtrain_valid/annotations/test.json --json json/0309-exp34-ep20.bbox.json --out 0309-exp34-ep20
python "tools/challenge_validation.py" --team 0309-exp34-ep20
