# Weather-Forecasting-Using-CNNs.
Predict weather conditions with CNNs! This project employs custom architectures and transfer learning (VGG16, ResNet152V2) for accurate forecasting. Explore data augmentation and visualize model performance over 20 epochs. 🌦️✨
Certainly! Here's a more detailed README file for your weather forecasting project:

---

# Weather Forecasting with Convolutional Neural Networks (CNNs)

Predict weather conditions accurately using state-of-the-art Convolutional Neural Networks (CNNs). This project incorporates custom CNN architectures and leverages transfer learning with VGG16 and ResNet152V2, enhancing forecasting precision. Explore advanced data augmentation techniques to enrich the diversity of the training dataset. Visualize model performance over 20 epochs for comprehensive insights.

## Author
- **Rahul Raj Singh**

## Features

- **Data Preprocessing:** Utilize ImageDataGenerator for effective data augmentation, preparing datasets for training, validation, and testing.
  
- **Model Architectures:**
  - **VGG16-based Model (`cnn_model`):** Transfer learning with VGG16, customized for weather forecasting.
  - **Custom CNN Model (`model2`):** A custom-designed CNN architecture.
  - **Transfer Learning with VGG16 (`model_t`):** Utilizes VGG16 as a feature extractor.
  - **Transfer Learning with ResNet152V2 (`model_res`):** Uses ResNet152V2 as a feature extractor.

- **Training and Evaluation:** The models are compiled using the Adam optimizer and trained over 20 epochs. Training history is visualized for each model, and the models are evaluated on a separate test set.

- **Data Augmentation for Forecasting:** Includes an example of data augmentation applied to a sample image for forecasting purposes.

## Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/weather-forecast-cnn.git
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Script:**
   ```bash
   python MODEL.PY
   ```

## Results

Explore detailed model performance, including loss and accuracy trends, in the Jupyter notebooks and `RESULTS.md`.

## Contributing

Contributions are welcome! Check out the [Contribution Guidelines](CONTRIBUTING.md) for more details.

