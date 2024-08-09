

---

# Trigger Word Detection

This project demonstrates how to build a trigger word detection system similar to those used in voice-activated devices like Amazon Alexa and Google Home. The trigger word is "activate," and a "chiming" sound is played when the system detects it within an audio stream.

## Table of Contents

* [Overview](#overview)
* [Data Synthesis](#data-synthesis)
* [Model](#model)
* [Predictions](#predictions)
* [Key Takeaways](#key-takeaways)
* [Usage](#usage)
* [Dependencies](#dependencies)
* [Contributing](#contributing)
* [Acknowledgements](#acknowledgements)
* [License](#license)

## Overview

This project guides you through the process of creating a trigger word detection system using deep learning. It covers various aspects such as:

* Generating synthetic speech data for training the model.
* Preprocessing raw audio signals into spectrograms, which are a visual representation of the frequencies present in the audio over time.
* Building and training a neural network model capable of identifying the trigger word within an audio stream.
* Making predictions on new audio files and incorporating a chime feedback mechanism when the trigger word is detected.

## Data Synthesis

Since collecting and labeling large amounts of real-world speech data can be challenging, this project leverages the power of data synthesis.

* **Audio Components:** The dataset comprises three main components:
    * Background noises recorded in various environments.
    * Positive examples containing the trigger word "activate."
    * Negative examples consisting of random words other than "activate."

* **Spectrogram Conversion:**
    * Raw audio recordings are first converted into spectrograms. Spectrograms visualize the intensity of different frequencies present in the audio over time, making it easier for the model to learn the patterns associated with the trigger word.

* **Training Data Generation:**
    * Training examples are synthesized by strategically overlaying "activate" clips and negative word clips onto background noise segments at random, non-overlapping time intervals.
    * Corresponding labels are generated, indicating precisely when the trigger word is present in each synthesized example. 

## Model

* **Architecture:**
    * The neural network employs a combination of 1D convolutional layers, GRU (Gated Recurrent Unit) layers, and dense layers.
    * The 1D convolutional layer processes the spectrogram, extracting meaningful features from the frequency and time dimensions.
    * GRU layers, a type of recurrent neural network (RNN), are used to analyze the sequential nature of audio data, capturing temporal dependencies and patterns relevant to trigger word identification.
    * Dense layers, including a final time-distributed dense layer with sigmoid activation, produce the probability of the trigger word being present at each time step.

* **Training and Evaluation:**
    * A pre-trained model is provided for convenience, but the code also allows for further training or fine-tuning using the generated dataset.
    * The model's performance is assessed on a separate development set of real-world audio recordings to ensure its ability to generalize to new, unseen data.

## Predictions

* **Trigger Word Detection:**
    * The `detect_triggerword` function takes an audio file as input, preprocesses it, and feeds it to the model to obtain predictions on the presence of the trigger word at each time step.
    * These predictions are then visualized, allowing you to inspect the model's confidence in its detections.

* **Chime Feedback:**
    * The `chime_on_activate` function analyzes the model's predictions and overlays a chime sound onto the audio if the trigger word is detected with a high probability.
    * It also incorporates a mechanism to prevent multiple chimes for a single trigger word occurrence.

## Key Takeaways

* **Synthetic data generation** is a valuable tool for addressing the scarcity of labeled speech data.
* **Spectrograms** provide a powerful representation of audio signals for deep learning models.
* **End-to-end deep learning** enables the development of effective trigger word detection systems.

## Usage

1. **Clone the repository.**
2. **Install the required dependencies.**
3. **Run the Jupyter Notebook (`Trigger Word Detection.ipynb`).**
4. **Follow the instructions within the notebook to train or fine-tune the model and make predictions on audio files.**

## Dependencies

* Python 3.x
* NumPy
* pydub
* TensorFlow/Keras
* Matplotlib
* IPython

Install dependencies

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests.


## Acknowledgements

https://www.coursera.org/learn/nlp-sequence-models
https://www.deeplearning.ai/program/deep-learning-specialization/

## License

This project is licensed under the MIT License.

---

