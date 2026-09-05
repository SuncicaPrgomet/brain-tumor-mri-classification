Brain Tumor MRI Classification — A Comparative ML Survey

A broad, comparative exploration of machine learning and deep learning techniques for classifying brain MRI images into four classes: glioma tumor, meningioma tumor, pituitary tumor, and no tumor. Rather than settling on a single model, this project systematically works through classical ML, dimensionality reduction, multiple CNN architectures, regularization strategies, and unsupervised methods on the same dataset — comparing what actually helps versus what doesn't.

Academic / personal project. Images are 150×150×3 RGB MRI scans, loaded from a labeled local dataset (not included in this repository).

Dataset

~1,634 labeled MRI images across 4 classes (glioma, meningioma, pituitary, no tumor), 150×150 pixels, RGB. Images were min-max scaled to [0, 1] before modeling.

Approach
1. Dimensionality reduction & visualization
t-SNE: projected the flattened 67,500-dimensional image vectors into 2D for visual inspection of class separability
PCA: reduced dimensionality from 67,500 to 607 components while retaining 95% of explained variance — used both for visualization and as a preprocessing step for classical models
2. Classical machine learning baselines
SVM (both directly on flattened pixels and on PCA-reduced features, including a OneVsRestClassifier comparison)
Logistic Regression (multinomial vs. one-vs-rest)
Gaussian Naive Bayes — investigated why it underperformed here (violates the feature-independence assumption for spatially correlated pixel data; also tested feature normality with a Kolmogorov–Smirnov test)
Decision Trees (including a pruning experiment) and Random Forest
KNN, with GridSearchCV to select the best k
K-Means clustering (unsupervised), with the elbow method and silhouette score for cluster quality
3. Convolutional Neural Networks
Iteratively built three CNN architectures of increasing depth and complexity (from a simple 2-block sequential model to a 4-block architecture with progressively more filters)
Compared against a plain fully-connected (FC) network on the same data to demonstrate why convolutional structure matters for image inputs
4. Regularization & training strategies

Systematically tested against the best-performing baseline CNN:

L2 weight regularization (swept across multiple λ values)
Dropout
Early stopping (on validation loss)
Data augmentation (flips, shifts, zoom, rotation via ImageDataGenerator)
5. Autoencoders
A simple dense autoencoder and a convolutional autoencoder, both trained to reconstruct input images through a compressed latent representation — used to explore unsupervised feature learning on the same image set
Key findings
CNNs substantially outperformed every classical ML approach (SVM, Logistic Regression, Naive Bayes, Decision Trees, Random Forest, KNN) and outperformed a plain fully-connected network on the same data — consistent with the expectation that convolutional structure is important for spatially correlated image data.
Naive Bayes performed poorly, as expected: its feature-independence assumption doesn't hold for image pixels, and a Kolmogorov–Smirnov test confirmed the feature distributions aren't normal either.
On this dataset, every regularization technique tested (L2, Dropout, Early Stopping, Data Augmentation) reduced accuracy relative to the unregularized baseline CNN. This is a genuinely useful negative result: it suggests the baseline model wasn't overfitting badly enough for these techniques to help, and that blindly adding regularization isn't always the right move — it should be justified by an actual overfitting signal in the baseline's train/validation curves, not applied by default.
PCA compressed the input from 67,500 to 607 dimensions while retaining 95% of the variance — a useful preprocessing step for the classical models, though it doesn't replace what a CNN learns end-to-end.
Tech stack

Python · TensorFlow/Keras · scikit-learn · OpenCV · imbalanced-learn (RandomOverSampler) · seaborn / matplotlib

Note on scope

This project is intentionally broad rather than narrowly optimized — the goal was to build hands-on intuition for when and why different ML/DL approaches succeed or fail on the same image classification task, including techniques (like the regularization sweep) that didn't improve results. That comparative, empirically-driven process is the main value of the project, not a single polished leaderboard score.
