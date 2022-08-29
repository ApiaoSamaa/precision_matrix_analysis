readme-MDA133Files.txt
MDA133 CEL, MBEI, Clinical Data, and Normalizer files


This MDA133 data set contains files with Affymetrix U133A microarray .CEL (version 3) files along with clinical data for 133 cases of neo-adjuvant treated breast cancer is a supplement to: Hess, et. al, Pharmacogenomic Predictor of Sensitivity to Preoperative Chemotherapy With Paclitaxel and 5-Fluorouracil, Doxorubicin, and Cyclophosphamide in Breast Cancer, Journal of Clinical Oncology, 24 (26), 2006. 




#######   MDA133-CELFiles.zip contains 4 zip-files:

## 200304-U133ACelFilesInMDA82.zip
## 200309-U133ACelFilesInMDA82.zip
## 200401-U133ACelFilesInMDA82.zip
These 3 .zip files contain the CEL files from the original 82 cases.

## MDAValidation51CELfiles.zip
This .zip file has the CEL files from the 51 case Validation set.




#######   MDA133-ClinicalAndMBEI.zip contains the following files:

## MDA133CompleteInfo20070319.xls
The clinical data and a code book for the clinical data is in this MS Excel file:


## CEL_file_info_MDA133.txt
This file is a 2-column .txt file with the CEL file name and then the IDtxt case identifier.This information also exists in the clinical information file.
This file can be used as the sample_info.txt file in dChip.


## MDA133PredictorTrainAndValidation.xls
Excel formatted file.  With the main data being 22284 rows by 134 columns.  First column is Affy probeset ID and first row has MDA IDtxt identifiers.
This data is the dChip Normalized Model Based Expression Index values for the MDA133 cases.
These arrays were normalized to M155 (MDA-BCNorm.CEL) and the UTMDACC-BreastCancerNormalizer.psi was used to calculate the MBEI values.




#######   MDABCNormCELandPSI.zip

## dChipReferenceV2-20060801.doc
This file has the instructions for using dChip with the MDA-BCNorm.CEL and .PS file below


## MDA-BCNorm.CEL
This is the digital standard chosen from the first 50 cases in the MDA133 data set.
It corresponds to case M155 and has a median intensity of 163.


## UTMDACC-BreastCancerNormalizer.psi
This is the probe sensitivity index file built from the first 50 cases in the MDA155 dataset.
It was built using version 3 CEL-files and the PM-Only model.


