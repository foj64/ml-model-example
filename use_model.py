#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 14:41:31 2024

@author: foj
"""

import pickle
import numpy as np

classifier = pickle.load(open("classifier.pickle", 'rb'))

scaler = pickle.load(open("sc.pickle", 'rb'))

nparr = scaler.transform(np.array([[40,40000]]))
new_pred = classifier.predict(nparr)

new_prob = classifier.predict_proba(nparr)[:,1]

