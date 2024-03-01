# Active Learning Transfer Learning Pipeline for Caltech256 Using ResNet
This README document provides an overview of the active learning transfer learning pipeline designed for the Caltech256 dataset using a ResNet model. This pipeline integrates the concepts of transfer learning and active learning to efficiently use a small labeled dataset for initial training and iteratively improve model performance by selectively labeling uncertain samples from an unlabeled dataset.

### Overview
The pipeline is structured around two main phases: the initial training phase using transfer learning on a subset of the Caltech256 dataset, and the iterative improvement phase using active learning to selectively label and add data to the training set. This approach allows for efficient use of human labeling effort by focusing on samples where the model is most uncertain.

### Initial Training Phase
Dataset Preparation: The Caltech256 dataset, which contains images across 256 object categories, is divided into a small labeled subset for initial training and a larger unlabeled pool for active learning.
Model Selection: A pre-trained ResNet model is used as the base model due to its strong performance in image classification tasks. The choice of ResNet variant (e.g., ResNet50, ResNet101) can be adjusted based on computational resources and performance requirements.
Transfer Learning: The pre-trained ResNet model is fine-tuned on the small labeled subset of the Caltech256 dataset. This involves adjusting the final layers of the model to output 256 classes and training the model to adapt to the Caltech256 domain.

### Active Learning Phase
Uncertainty Sampling: The fine-tuned ResNet model is used to make predictions on the unlabeled dataset. Samples for which the model has the highest uncertainty are identified. Uncertainty can be measured through various methods, such as entropy or least confidence.
Human Labeling: The selected uncertain samples are presented to human annotators for labeling. This step leverages expert knowledge to provide accurate labels for difficult or ambiguous cases.
Dataset Update: The newly labeled samples are added to the training dataset, augmenting the labeled data pool.
Model Retraining: The ResNet model is retrained on the updated training dataset, incorporating the newly labeled samples into the learning process. This step iteratively improves the model's performance over time.

### Implementation Steps
Environment Setup: Ensure that all necessary libraries and frameworks (e.g., PyTorch, TensorFlow) are installed. Additionally, access to the Caltech256 dataset is required.
Pre-processing: Normalize the images and apply any necessary augmentations to increase the diversity of the training data.
Model Training: Implement the initial training phase using transfer learning techniques to fine-tune the ResNet model on the labeled subset of the Caltech256 dataset.
Active Learning Loop:
-Evaluate model uncertainty on the unlabeled dataset.
-Select samples with the highest uncertainty for labeling.
-Label the selected samples and add them to the training dataset.
-Retrain the model on the updated dataset.
-Repeat the active learning loop until the desired model performance is achieved or the labeling budget is exhausted.

