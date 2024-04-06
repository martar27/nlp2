# NLP2 Codebase Overview

This codebase is designed to work with the SAMSum dataset for dialogue summarization using a sequence-to-sequence model with attention mechanisms. Below is a detailed overview of each module and how to use the codebase.

## Modules

### `data_loader.py`
- **Purpose**: Handles the loading and preprocessing of the SAMSum dataset. It includes functionality for tokenizing dialogues and summaries, building a vocabulary, and preparing data for training, validation, and testing.
- **Key Functions**:
  - `load_and_preprocess_data()`: Loads the SAMSum dataset, builds the vocabulary, and returns processed train, validation, and test sets.

### `model.py`
- **Purpose**: Defines the neural network architecture including an Encoder, Decoder, and Attention mechanism, which are essential components of the sequence-to-sequence model.
- **Key Classes**:
  - `Encoder`: Encodes the input dialogue sequences into a set of hidden states.
  - `Attention`: Computes attention weights and generates a context vector based on the encoder's output and the current decoder state.
  - `Decoder`: Generates the output summary sequence based on the encoder's output and attention context.

### `train.py`
- **Purpose**: Contains the training loop for the neural network, including forward pass, loss computation, backpropagation, and model evaluation on the validation set. It also includes logic for saving the best model based on validation loss.
- **Key Functions**:
  - `train_model()`: Executes the training process, evaluates the model on the validation set, and saves the best-performing model.

### `inference.py`
- **Purpose**: Provides functionality to generate summaries for new dialogues using the trained model.
- **Key Functions**:
  - `generate_summary()`: Generates a summary for a given dialogue input.

## Usage

1. **Preprocess the data**: Run `data_loader.py` to load the SAMSum dataset and build the vocabulary.
2. **Train the model**: Execute `train.py` to train the model on the preprocessed data.
3. **Generate summaries**: Use `inference.py` to generate summaries for new dialogues using the trained model.

For more detailed instructions, refer to the comments within each script.
1. **Preprocess the data**: Run `data_loader.py` to load and preprocess the SAMSum dataset and build the vocabulary.
2. **Train the model**: Execute `train.py` to train the model on the preprocessed data.
3. **Generate summaries**: Use `inference.py` to generate summaries for new dialogues using the trained model.

For more detailed instructions, refer to the comments within each script.
