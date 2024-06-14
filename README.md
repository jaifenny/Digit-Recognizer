# Digit Recognizer
### 👉 [Click here to enter the competition](https://www.kaggle.com/competitions/digit-recognizer)

### Competition Description ###
MNIST ("Modified National Institute of Standards and Technology") is the de facto “hello world” dataset of computer vision. 
Since its release in 1999, this classic dataset of handwritten images has served as the basis for benchmarking classification algorithms. 
As new machine learning techniques emerge, MNIST remains a reliable resource for researchers and learners alike.

In this competition, your goal is to correctly identify digits from a dataset of tens of thousands of handwritten images. 
We’ve curated a set of tutorial-style kernels which cover everything from regression to neural networks. 
We encourage you to experiment with different algorithms to learn first-hand what works well and how techniques compare.

### Dataset
- `train.csv` - the training set.
- `test.csv` - the test set. The testing data contains the same information as the training set, but without the label column, it needs to be predicted.
- `sample_submission.csv` -Your submission file should be in the following format: For each of the 28000 images in the test set, output a single line containing the ImageId and the digit you predict. For example, if you predict that the first image is of a 3, the second image is of a 7, and the third image is of a 8, then your submission file would look like:
    ```
    ImageId,Label
    1,3
    2,7
    3,8 
    (27997 more lines)
    ```
