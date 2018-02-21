#!/usr/bin/env bash

cd `dirname $0`

python main.py --content_path images/content.jpg --style_path images/style.jpg --output_path images/styled.jpg