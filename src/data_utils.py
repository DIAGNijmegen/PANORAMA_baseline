# Copyright 2024 Diagnostic Image Analysis Group, Radboud
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import SimpleITK as sitk
import time
import os
from report_guided_annotation import extract_lesion_candidates

def resample_img(itk_image, out_spacing=[2.0, 2.0, 2.0], is_label=False, out_size = [], out_origin = [], out_direction= []):
    original_spacing = itk_image.GetSpacing()
    original_size    = itk_image.GetSize()
    

    if not out_size:
        out_size = [ int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
                        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
                        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]
    
    # set up resampler
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    if not out_direction:
        out_direction = itk_image.GetDirection()
    resample.SetOutputDirection(out_direction)
    if not out_origin:
        out_origin = itk_image.GetOrigin()
    resample.SetOutputOrigin(out_origin)
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)
    # perform resampling
    itk_image = resample.Execute(itk_image)

    return itk_image


def CropPancreasROI(image, low_res_segmentation, margins):
     
    pancreas_mask_np = sitk.GetArrayFromImage(low_res_segmentation)
    assert(len(np.unique(pancreas_mask_np))==2)    
    
    pancreas_mask_nonzeros = np.nonzero(pancreas_mask_np)
    
    min_x = min(pancreas_mask_nonzeros[2])
    min_y = min(pancreas_mask_nonzeros[1])
    min_z = min(pancreas_mask_nonzeros[0])
    
    max_x = max(pancreas_mask_nonzeros[2])
    max_y = max(pancreas_mask_nonzeros[1])
    max_z = max(pancreas_mask_nonzeros[0])
    
    start_point_coordinates = (int(min_x), int(min_y), int(min_z))
    finish_point_coordinates = (int(max_x), int(max_y), int(max_z))          
    
    start_point_physical = low_res_segmentation.TransformIndexToPhysicalPoint(start_point_coordinates)
    finish_point_physical = low_res_segmentation.TransformIndexToPhysicalPoint(finish_point_coordinates)
    
    start_point = image.TransformPhysicalPointToIndex(start_point_physical)
    finish_point = image.TransformPhysicalPointToIndex(finish_point_physical)


    spacing = image.GetSpacing()
    size = image.GetSize()
        
    marginx = int(margins[0]/spacing[0])
    marginy = int(margins[1]/spacing[1])
    marginz = int(margins[2]/spacing[2])
    
    x_start = max(0, start_point[0] - marginx)
    x_finish = min(size[0], finish_point[0] + marginx)
    y_start = max(0, start_point[1] - marginy)
    y_finish = min(size[1], finish_point[1] + marginy)
    z_start = max(0, start_point[2] - marginz)
    z_finish = min(size[2], finish_point[2] + marginz)
    
    cropped_image = image[x_start:x_finish, y_start:y_finish, z_start:z_finish]

    crop_coordinates = {'x_start': x_start,
                        'x_finish': x_finish,
                        'y_start': y_start,
                        'y_finish': y_finish,
                        'z_start': z_start,
                        'z_finish': z_finish}
      
    return cropped_image, crop_coordinates


def GetFullSizDetectionMap(cropped_prediction, cropp_coordinates, full_image):
    prediction_np = cropped_prediction['probabilities'][1]
    prediction_np = prediction_np.astype(np.float32)

    lesion_candidates, confidences, indexed_pred = extract_lesion_candidates(prediction_np)


    patient_level_prediction = float(np.max(lesion_candidates))


    full_size_detection_map = np.zeros(sitk.GetArrayFromImage(full_image).shape)
    full_size_detection_map = full_size_detection_map.astype(np.float32)


    # Use integer slicing, ensuring no slice is empty
    z_slice = slice(int(cropp_coordinates['z_start']), int(cropp_coordinates['z_finish']))
    y_slice = slice(int(cropp_coordinates['y_start']), int(cropp_coordinates['y_finish']))
    x_slice = slice(int(cropp_coordinates['x_start']), int(cropp_coordinates['x_finish']))

    full_size_detection_map[z_slice, y_slice, x_slice] = lesion_candidates
    full_size_detection_map = full_size_detection_map.astype(np.float32)

    detection_map_image = sitk.GetImageFromArray(full_size_detection_map)
    detection_map_image.CopyInformation(full_image)
    return detection_map_image, patient_level_prediction
        

 
