# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 14:39:09 2021

@author: ahmed
"""
from tuner_hps import *
try:
  print("Starting HPS guitar tuner...")
  with sd.InputStream(channels=1, callback=callback, blocksize=WINDOW_STEP, samplerate=SAMPLE_FREQ):
    while True:
      time.sleep(0.5)
except Exception as exc:
  print(str(exc))