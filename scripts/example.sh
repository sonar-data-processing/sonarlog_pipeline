#!/bin/bash

/home/romulo/workspace/sonar_toolkit/features/build/features \
    --input-files=/home/romulo/workspace/sonar_toolkit/data/logs/balsa.0.log \
    --input-files=/home/romulo/workspace/sonar_toolkit/data/logs/gemini-jequitaia.0.log \
    --stream-name="gemini.sonar_samples"
