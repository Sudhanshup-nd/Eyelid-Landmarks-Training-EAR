--------Steps to get the annotation data and annotation csv---------------

1. We already have the face crops which were given for labelling
1.a. if you dont have you can download the face crops form cvat during the xml file export

2. Now go to the project on cvat and export the xml file in cvat for images 1.1 format (you can select to download the face crops too)

3. Give your file name and put it in cloud/production (for eg I named it eyelid_labels_phase_2.1)

4. Now go to s3/netradyne-labelling-production/ and search your folder eyelid_labels_phase_2.1 download the zip and unzip it locally

5. Now run /inwdata2a/sudhanshu/Unet_training_script/dataset_generation/generate-eye-landmark-csv.py which takes 3 things
  the xml file (from 4) and face crops folder and then the output csv name
  the csv gives the eyecrops on using logic of making eye_bbox such that all 6 landmarks are inside the bbox and then bbox is cenrely padded and made square

6. you can use /inwdata2a/sudhanshu/Unet_training_script/dataset_generation/check_annotations.py to confirm if the eye bbox are correct or not

7. Now combine both the phase 1 and phase 2 csvs and split using /inwdata2a/sudhanshu/Unet_training_script/dataset_generation/split_train_val_test.py


