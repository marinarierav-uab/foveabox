#!/bin/bash

# generate csv
python "tools/csv_generator.py" --original data/CVC-VideoClinicDBtrain_valid/annotations/test.json --json json/0217-exp25.bbox.json --out 0217-exp25
python "tools/csv_generator.py" --original data/CVC-VideoClinicDBtrain_valid/annotations/test.json --json json/0217-exp26.bbox.json --out 0217-exp26
python "tools/csv_generator.py" --original data/CVC-VideoClinicDBtrain_valid/annotations/test.json --json json/0217-exp27.bbox.json --out 0217-exp27
python "tools/csv_generator.py" --original data/CVC-VideoClinicDBtrain_valid/annotations/test.json --json json/0217-exp28.bbox.json --out 0217-exp28

# calculate average results
python "tools/challenge_validation.py" --team 0217-exp25 --list
python "tools/challenge_validation.py" --team 0217-exp26 --list
python "tools/challenge_validation.py" --team 0217-exp27 --list
python "tools/challenge_validation.py" --team 0217-exp28 --list



