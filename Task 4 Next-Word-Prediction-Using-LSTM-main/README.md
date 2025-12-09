# ✨ AI Word Predictor - Next Word Prediction App

A beautiful, production-ready web application for next-word prediction using LSTM neural networks. Built with Streamlit and TensorFlow.

## 🌟 Features

- **🤖 AI-Powered Predictions**: Advanced LSTM neural network for accurate word predictions
- **⚡ Real-Time Generation**: Generate text with smooth animations
- **🎨 Modern UI**: Beautiful, responsive design with gradient themes
- **📊 Analytics Dashboard**: Track usage statistics and generation history
- **💡 Quick Start Options**: Pre-defined seed texts for quick testing
- **📝 History Management**: Save and revisit your generated texts
- **🔮 Multiple Prediction Modes**: Top 1, Top 3, or Top 5 suggestions
- **⚙️ Customizable Settings**: Adjust generation speed and word count

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Trained model files: `my_model.h5` and `tokenizer.pkl`

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Task-4-Next-Word-Prediction-Using-LSTM-main
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure you have the model files:
   - `my_model.h5` (trained LSTM model)
   - `tokenizer.pkl` (fitted tokenizer)

### Running the App

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## 📖 Usage

### Generate Text

1. Enter a seed text in the input field
2. Adjust the number of words to generate (1-50)
3. Click "🚀 Generate" to create text with animation
4. Or click "🔮 Predict Next" for instant next-word suggestions

### Settings

- **Words to Generate**: Control how many words to generate (1-50)
- **Prediction Mode**: Choose between Top 1, Top 3, or Top 5 suggestions
- **Animation Speed**: Adjust the speed of text generation (0.1-1.0)
- **Show Probabilities**: Display prediction confidence scores

### Features

- **Quick Start**: Use pre-defined seed texts for instant testing
- **History**: View and reuse previously generated texts
- **Copy/Save**: Copy generated text or save to file
- **Statistics**: Track words generated and predictions made

## 🎨 UI Features

- **Gradient Themes**: Beautiful purple gradient design
- **Responsive Layout**: Works on desktop and mobile
- **Smooth Animations**: Real-time text generation with progress indicators
- **Interactive Elements**: Hover effects and transitions
- **Professional Styling**: Modern card-based layout

## 📁 Project Structure

```
.
├── app.py                          # Main Streamlit application
├── my_model.h5                     # Trained LSTM model
├── tokenizer.pkl                   # Fitted tokenizer
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── LSTM_based_text_model_professional.ipynb  # Training notebook
```

## 🔧 Configuration

The app uses the following default settings:
- Maximum sequence length: 16
- Vocabulary size: 1783
- Model architecture: Embedding(100) → LSTM(150) → Dense(1783)

## 📊 Model Information

- **Architecture**: LSTM-based neural network
- **Embedding Dimension**: 100
- **LSTM Units**: 150
- **Output Layer**: Dense with softmax activation
- **Total Parameters**: ~598,133

## 🛠️ Troubleshooting

### Model Not Loading
- Ensure `my_model.h5` and `tokenizer.pkl` are in the same directory as `app.py`
- Check that the model files are not corrupted

### Slow Performance
- Reduce the number of words to generate
- Increase animation speed
- Use "Predict Next" instead of full generation for faster results

### Memory Issues
- Reduce batch size if processing large texts
- Close other applications to free up memory

## 📝 License

This project is part of a technical internship task.

## 👨‍💻 Author

Created as part of Techinfo Internship Tasks - Task 4

## 🙏 Acknowledgments

- TensorFlow/Keras for the deep learning framework
- Streamlit for the web framework
- The Dracula novel text used for training

---

**Enjoy generating text with AI! 🚀✨**

