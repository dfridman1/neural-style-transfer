#!/usr/bin/env bash

cd `dirname $0`

python main.py --content_path images/content/chicago.jpg --style_path images/style/wave.jpg --output_path images/output/chicago_wave.jpg
