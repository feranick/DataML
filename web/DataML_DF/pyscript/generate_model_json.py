#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*****************************************************
* Generate list of models from the folder names
* version: 2026.04.07.1
* By: Nicola Ferralis <feranick@hotmail.com>
*****************************************************
'''

import os
import json

# Find all directories that contain a config.txt
models = [d for d in os.listdir('.') if os.path.isdir(d) and os.path.exists(os.path.join(d, 'config.txt'))]

# Write to a JSON file
with open('models_list.json', 'w') as f:
    json.dump(sorted(models), f)
    
print(f"Updated models_list.json with {len(models)} models.")
