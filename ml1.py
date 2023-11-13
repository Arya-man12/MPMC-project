# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 00:04:25 2023

@author: Aryaman Kumar
"""
import tf
import tflite
converter = tf.lite.TFLiteConverter.from_saved_model("C:/Users\Aryaman Kumar\Desktop\mpmc\mood-detection-and-live-camera.ipynb")
tflite_model = converter.convert()