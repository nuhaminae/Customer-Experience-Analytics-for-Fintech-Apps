# Customer Experience Analytics for Fintech Apps

This repository provides a comprehensive analytical framework for understanding and enhancing customer experience in fintech applications. Through a combination of data science techniques and interactive Jupyter Notebooks, the project aims to extract actionable insights from user data, feedback, and behavioural patterns from three Ethiopian banks –Bank of Abyssinia, Commerical Bank of Ethiopia, and Dashen Bank – mobile application. 
---

## Overview

Customer experience (CX) is critical to the success of any fintech product. By systematically analysing user sentiment and feedback organisations can identify pain points, improve retention, and tailor services to user needs.

This project is designed for data analysts, product managers, and researchers seeking to leverage data-driven approaches for CX improvement in fintech apps.

---

## Methodology

The analysis in this repository follows a structured methodology:

1. **Data Acquisition and Preparation**
   - Collect user rating, feedback, sentiment data.
   - Clean and preprocess data, ensuring consistency and handling missing and duplicated values.

2. **Exploratory Data Analysis (EDA)**
   - Explore key metrics like ratings and reviews.

3. **Sentiment and Text Analytics**
   - Apply Natural Language Processing (NLP) to feedback and support data.
   - Use sentiment analysis to gauge customer satisfaction and identify recurrent themes.

4. **Visualisation and Reporting**
   - Present findings through compelling plots, dashboards, and summary statistics.
   - Visualise key metrics like positive and negative sentiments. 

---

## Repository Structure

```
.
├── notebooks/          # Jupyter Notebooks for each analysis step
├── data/               # (gitignored) Raw and processed data files
├── scripts/            # Reusable Python modules for analysis
├── requirements.txt    # Python package dependencies
└── README.md           # Project documentation
```

---

## Getting Started

### Prerequisites

- Python 3.7 or above
- Jupyter Notebook or JupyterLab
- Standard data science libraries: `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `nltk` or `spaCy`, etc.

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/nuhaminae/Customer-Experience-Analytics-for-Fintech-Apps.git
   cd Customer-Experience-Analytics-for-Fintech-Apps
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```
   Open the desired notebook in the `notebooks/` directory.

---

## Usage

1. **Place your data** in the `data/` folder, ensuring paths in notebooks match your data files.
2. **Run notebooks** sequentially to perform web scrapping and then analysis.
3. **Interpret outputs** to inform product improvements and customer support strategies.

---

## Contributing

Contributions are welcome. Please open an issue or submit a pull request for improvements or new analytics modules.

---

### Project Status
The project is not completed. Check [commit history](https://github.com/nuhaminae/Customer-Experience-Analytics-for-Fintech-Apps/commits/main/) for full detail.