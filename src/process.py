#  Copyright 2024 Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os
import glob
import SimpleITK as sitk
import numpy as np

from evalutils import SegmentationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)

# imports required for running nnUNet algorithm
import subprocess
from subprocess import check_output, STDOUT, CalledProcessError
from pathlib import Path
import json
# imports required for my algorithm
from data_utils import resample_img, CropPancreasROI, GetFullSizDetectionMap, PostProcessing

import warnings
warnings.filterwarnings("ignore")

class PDACDetectionContainer(SegmentationAlgorithm):
    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )
        # input / output paths for nnUNet

        self.nnunet_input_dir_lowres = Path("/opt/algorithm/nnunet/input_lowres") 
        self.nnunet_input_dir_fullres = Path("/opt/algorithm/nnunet/input_fullres")
        self.nnunet_output_dir_lowres = Path("/opt/algorithm/nnunet/output_lowres")
        self.nnunet_output_dir_fullres = Path("/opt/algorithm/nnunet/output_fullres")
        self.nnunet_model_dir = Path("/opt/algorithm/nnunet/results")
       
        # input / output paths
        self.ct_ip_dir         = Path("/input/images/venous-ct")
        self.output_dir        = Path("/output")

        self.output_dir_images = Path(os.path.join(self.output_dir,"images")) 
        self.output_dir_tlm    = Path(os.path.join(self.output_dir_images,"pdac-detection-map")) 
        self.detection_map     = self.output_dir_tlm / "detection_map.mha"

        # ensure required folders exist
        self.nnunet_input_dir_lowres.mkdir(exist_ok=True, parents=True)
        self.nnunet_input_dir_fullres.mkdir(exist_ok=True, parents=True)
        self.nnunet_output_dir_lowres.mkdir(exist_ok=True, parents=True)
        self.nnunet_output_dir_fullres.mkdir(exist_ok=True, parents=True)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.output_dir_tlm.mkdir(exist_ok=True, parents=True)

        mha_files = glob.glob(os.path.join(self.ct_ip_dir, '*.mha'))
        print(mha_files)

        # Check if any .mha files were found
        if mha_files:
            # Assuming you want to read the first .mha file found
            self.ct_image = mha_files[0]
        else:
            print('No mha images found in input directory')

    def process(self):
        """
        Load CT scan and Generate Heatmap for Pancreas Cancer  
        """
        itk_img    = sitk.ReadImage(self.ct_image, sitk.sitkFloat32)

        #Get low resolution pancreas segmentation 
        #dowsample image to (4.5, 4.5, 9.0)
        new_spacing = (4.5, 4.5, 9.0)
        image_resampled = resample_img(itk_img, new_spacing, is_label=False, out_size = [])
        sitk.WriteImage(image_resampled, str(self.nnunet_input_dir_lowres / "scan_0000.nii.gz"))

        #predict pancreas mask using nnUnet
        self.predict(
            input_dir=self.nnunet_input_dir_lowres,
            output_dir=self.nnunet_output_dir_lowres,
            task="Dataset103_PANORAMA_baseline_Pancreas_Segmentation"
        )

        mask_pred_path = str(self.nnunet_output_dir_lowres / "scan.nii.gz")
        mask_low_res = sitk.ReadImage(mask_pred_path)

        crop_margins = [100,50,15]
        cropped_image, crop_coordinates = CropPancreasROI(itk_img, mask_low_res, crop_margins)

        sitk.WriteImage(cropped_image, str(self.nnunet_input_dir_fullres / "scan_0000.nii.gz"))

        self.predict(
        input_dir=self.nnunet_input_dir_fullres,
        output_dir=self.nnunet_output_dir_fullres,
        task="Dataset104_PANORAMA_baseline_PDAC_Detection",
        trainer="nnUNetTrainer_Loss_CE_checkpoints",
        checkpoint= 'checkpoint_best_panorama.pth'
        )

        pred_path_npz = str(self.nnunet_output_dir_fullres / "scan.npz")
        prediction = np.load(pred_path_npz)
        pred_path_nifti = str(self.nnunet_output_dir_fullres / "scan.nii.gz")

        prediction_postprocessed = PostProcessing(prediction, pred_path_nifti)
        detection_map, patient_level_prediction = GetFullSizDetectionMap(prediction_postprocessed, crop_coordinates, itk_img)
        
        sitk.WriteImage(detection_map, self.detection_map)
        write_json_file(location=self.output_dir / "pdac-likelihood.json", content=patient_level_prediction)




    def predict(self, input_dir, output_dir, task="Task103_AllStructures", trainer="nnUNetTrainer",
                    configuration="3d_fullres", checkpoint="checkpoint_final.pth", folds="0,1,2,3,4", 
                    store_probability_maps=True):
            """
            Use trained nnUNet network to generate segmentation masks
            """

            # Set environment variables
            os.environ['RESULTS_FOLDER'] = str(self.nnunet_model_dir)

            # Run prediction script
            cmd = [
                'nnUNetv2_predict',
                '-d', task,
                '-i', str(input_dir),
                '-o', str(output_dir),
                '-c', configuration,
                '-tr', trainer,
                '--disable_progress_bar',
                '--continue_prediction'
            ]

            if folds:
                cmd.append('-f')
                cmd.extend(folds.split(','))

            if checkpoint:
                cmd.append('-chk')
                cmd.append(checkpoint)

            if store_probability_maps:
                cmd.append('--save_probabilities')



            cmd_str = " ".join(cmd)
            subprocess.check_call(cmd_str, shell=True)

def write_json_file(*, location, content):
    # Writes a json file
    with open(location, 'w') as f:
        f.write(json.dumps(content, indent=4))

if __name__ == "__main__":
    PDACDetectionContainer().process()
