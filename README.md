<!--
{% comment %}
Licensed to the Apache Software Foundation (ASF) under one or more
contributor license agreements.  See the NOTICE file distributed with
this work for additional information regarding copyright ownership.
The ASF licenses this file to you under the Apache License, Version 2.0
(the "License"); you may not use this file except in compliance with
the License.  You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
{% endcomment %}
-->

# Patch Based Analysis with Machine Learning for Breast Cancer Recurrence Prediction

This project contains code for using machine learning to predict breast cancer recurrence from whole slide images.

The files in the patch_processing folder can be used to generate patches for each slide and and extract features for each image patch. Extracted features will be stored in NumPy files. Code is also available to aggregate patch features into bags based on their associated case.

The code in the logistic_regression model can be used to train and evaluate a logistic regression model on the feature bags.

## Processing steps

1. Use patch_extract.ipbynb to extract patches from your whole slide image directory. Make sure to update the file paths for your local file structure.

2. Run extract_patch_features.py to generate features for the image patches.

3. Use make_bags.py to create feature bags for each case.

4. Use logistic_regression_crossval.py and logistic_regression_holdout.py to train and evaluate a logistic regression model on the dataset.