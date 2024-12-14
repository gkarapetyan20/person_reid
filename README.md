# Person_reid

---

## Description

This project focuses on **Person Re-Identification** (ReID) using the **Market-1501 dataset**. The goal of person re-identification is to recognize and match individuals across different images or camera viewpoints.

The project leverages a **Siamese Neural Network (Siamese NN)** architecture implemented in **Python** with **PyTorch**.

---

## Technologies Used

- **Python**
- **PyTorch**
- **Market-1501 Dataset**

---

## Evaluation Metrics

| **Metric**   | **Score**  |
|--------------|------------|
| Accuracy     | 97.00%     |
| Precision    | 100.00%    |
| Recall       | 86.36%     |
| F1 Score     | 92.68%     |

---

## Installation

To set up the project, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/gkarapetyan20/person_reid
   cd person_reid
   ```

2. **Install Dependencies**:
   Install the required packages from the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Market-1501 Dataset**:
   - Download the dataset from [this link](https://www.kaggle.com/datasets/ljlbarb/market1501) or the official source.
   - Extract it into the appropriate directory within the project.

---

## Usage

1. **Train the Model**:
   ```bash
   python train.py --dataset_path ... --epoch ... --backbone ... --learning_rate ... --batch_size ...
   ```

2. **Evaluate the Model**:
   ```bash
   python test.py
   ```

---

## Contributing

Contributions are welcome! If you'd like to improve this project or fix any issues, please open a pull request.

---

## Contact

For any questions or further information, feel free to reach out:

- **Email**: [gevorgkarapetyan229@gmail.com](mailto:gevorgkarapetyan229@gmail.com)
- **LinkedIn**: [Gevorg Karapetyan](https://www.linkedin.com/in/gevorg-karapetyan-21b013229/)

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

