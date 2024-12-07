import os
import SimpleITK as sitk
from radiomics import featureextractor
import pandas
import numpy as np


# Load the file with the configurations params for extracting the radiomics features
params = os.getcwd() + '/Params.yaml'

# Define the extractor based on the params
extractor = featureextractor.RadiomicsFeatureExtractor(params)
# hang on to all our features
features = {}
diagnostic = []

# Load the diagnostic for each of the images included in the dataset
info_table  = pandas.read_csv(r'/Users/eliesfustergarcia/OneDrive - UPV/FEINA/DOCENCIA/DOCENCIA_2021/BDS (ESTINF)/BDS-CD-ETSINF/semana 11_radiomics/Semminar/Data/name_mapping.csv')
tumor_grade = info_table.values[:,0]
tumor_name  = info_table.values[:,5]

# Define the dataset location and obtain the foldernames
base_path = r'/Users/eliesfustergarcia/Desktop/Data for semminar Week 12/Imaging_data/'
cases = sorted(os.listdir(base_path))
cases = cases[1:]

# Select which cases to include in the study
sel_cases = np.array(list(range(1,360)))

# Extract the features from the ROI
for t in range(0, sel_cases.shape[0]):
    path = base_path+cases[sel_cases[t]]
    print(path)
    table_idx = [i for i, item in enumerate(tumor_name) if cases[sel_cases[t]] in item]
    image = sitk.ReadImage(path + "/T1c.nii.gz")
    mask = sitk.ReadImage(path + "/Segmentation.nii.gz")
    features[t] = extractor.execute ( image, mask, label=1 )
    diagnostic.append(tumor_grade[table_idx[0]])

# A list of the valid features, sorted
feature_names = list(sorted(filter ( lambda k: k.startswith("original_"), features[1] )))

# Make a numpy array of all the values
samples = np.zeros((sel_cases.shape[0], len(feature_names)))
for case_id in range(0,sel_cases.shape[0]):
    a = np.array([])
    for feature_name in feature_names:
        a = np.append(a, features[case_id][feature_name])
    samples[case_id , :] = a

# May have NaNs
samples = np.nan_to_num(samples)

# At this point we have the Radiomics Features in "samples" array and the tumor type in "diagnostic" variable