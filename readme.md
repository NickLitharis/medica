# Image Captioning with Deep Learning

This project is a university thesis and implements an image captioning system using deep learning techniques, combining Convolutional Neural Networks (CNNs) for image feature extraction and Recurrent Neural Networks (RNNs) with attention mechanisms for generating descriptive captions.

## Results

Trained and evaluated on the **ImageCLEF 2017 Medical Image Captioning** benchmark, scored with BLEU against the published baselines the thesis compares with:

| Model | BLEU-1 | BLEU-2 |
| --- | --- | --- |
| ImageCLEF 2017 reference model | 0.142 | 0.070 |
| Retinal Image Captioning | 0.158 | 0.076 |
| Global-Local Visual Extractor | 0.148 | 0.074 |
| **OLIVER — this work (CNN + attention RNN)** | **0.309** | **0.205** |

The hybrid CNN-RNN model with integrated attention — named *OLIVER* in the thesis — scored highest on both metrics. Scores are reported in Appendix D of the thesis; `model_usage.ipynb` regenerates the evaluation.

## Notebooks

- `model_training_preprocessing.ipynb`: Data loading, preprocessing, tokenization, train/test split, data generator, model building, training, and saving the model.
- `model_usage.ipynb`: Model and tokenizer loading, caption generation, evaluation, and visualization.

## Project Structure

- `model_training_preprocessing.ipynb`: Notebook for model training and data preprocessing.
- `model_usage.ipynb`: Notebook for model inference, evaluation, and visualization.
- `pyproject.toml` / `uv.lock`: Declared dependencies and the locked resolution.
- `LICENSE` / `LICENSE-THESIS.txt` / `DATA_LICENSE.txt`: Licensing for the code, the thesis document, and the dataset respectively.

The preprocessed dataset is **not committed** (see `DATA_LICENSE.txt`). To reproduce the pipeline, supply your own `dataset/` folder containing:
	- `image_features.pkl`: Extracted image features (typically from a CNN).
	- `captions.pkl`: Dictionary mapping image IDs to their captions.
	- `max_length.pkl`: Maximum caption length for padding.
	- `tokenizer.pkl`: Fitted tokenizer for text preprocessing.
	- `vocab_size.pkl`: Vocabulary size for the model.

## Workflow Overview

1. **Data Loading**: Loads image features and captions from preprocessed pickle files.
2. **Text Preprocessing**: Adds start/end tokens to captions, tokenizes text, and calculates vocabulary size.
3. **Train/Test Split**: Splits image IDs into training and test sets.
4. **Data Generator**: Prepares batches for model training, pairing image features with partial caption sequences.
5. **Model Architecture**:
	 - Encoder: Processes image features.
	 - Decoder: Processes caption sequences using Embedding and Bidirectional LSTM layers.
	 - Attention Mechanism: (optional) Improves alignment between image regions and words.
	 - Output: Predicts next word in the caption sequence.
6. **Training**: Trains the model using the generator, with callbacks for early stopping, learning rate reduction, checkpointing, and TensorBoard logging.
7. **Caption Generation**: Generates captions for new images using the trained model.
8. **Evaluation**: Computes BLEU, METEOR, ROUGE, and CIDEr scores to assess caption quality.
9. **Visualization**: Displays images with actual and predicted captions.

## Usage

1. Install dependencies:
	```
	uv sync
	```
2. Run `model_training_preprocessing.ipynb` to train and save the model.
3. Run `model_usage.ipynb` to generate captions, evaluate, and visualize results.
4. Ensure all required data files are present in the `dataset/` folder. Adjust paths as needed for your environment.

## Requirements

Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`.

TensorFlow is pinned to 2.15, the last release on the Keras 2 API that these notebooks
are written against — 2.16 ships Keras 3 and the model code does not run on it. That
pin also caps Python at 3.11, which `requires-python` states explicitly; uv fetches a
suitable interpreter for you, so `uv sync` works on a machine whose default Python is
newer. On Windows the resolver also swaps in a `tensorflow-io-gcs-filesystem` build
that has a `win_amd64` wheel, which is what makes the install work here as well as on
Linux (Kaggle / Colab).

## Notes

- The notebook is designed for flexibility: you can experiment with different model architectures, optimizers, and training strategies.
- Paths in the notebook may reference Kaggle directories; update them for local use as needed.
- Evaluation and visualization code is provided but commented out for modularity.

## References

- [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555)
- [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044)


---
For questions or improvements, feel free to open an issue or contribute.

## License

This repository uses different licenses for code, the thesis document, and the dataset:

- Code (notebooks and scripts): MIT License — see `LICENSE` for full text.
- Thesis (PDF): Creative Commons Attribution-NonCommercial 4.0 — see `LICENSE-THESIS.txt`.
- Dataset: Provided for research and educational use only — see `DATA_LICENSE.txt` for details and restrictions.

If you require a different license for any of these artifacts, please contact the author.
