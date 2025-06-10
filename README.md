# Customer Experience Analytics for Fintech Apps

The project is a data science project for analysing customer reviews of majour Ethiopian Banks' mobile applications. The repository provides tools for automated review scraping, preprocessing, sentiment analysis, thematic mining, and visualisation of user feedback from Google Play Store.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Directory Structure](#directory-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
  - [Usage](#usage)
- [Analysis Workflow](#analysis-workflow)
- [Results and Visualisations](#results--visualisations)
- [Extending the Project](#extending-the-project)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project automates the extraction and analysis of user reviews from Google Play Store for top Ethiopian banks' mobile banking applications. It applies NLP techniques to uncover sentiment, detect recurring themes, and visualise feedback trends, providing actionable insights for product improvement and customer satisfaction.

**Targeted Banks:**
- Bank of Abyssinia (BOA)
- Commercial Bank of Ethiopia (CBE)
- Dashen Bank

**Targeted Apps:**
- BOAMobile
- Commercial Mobile Banking
- Dashen Super App

---

## Features

- **Automated Scraping:** Collects user reviews using custom Python scripts.
- **Data Cleaning and Preprocessing:** Removes duplicates, handles missing values, and standardises data.
- **Sentiment Analysis:** Classifies reviews as positive or negative and computes sentiment scores.
- **Thematic Analysis:** Extracts key topics and themes using TF-IDF and spaCy.
- **Visualisation:** Generates sentiment distribution plots and word clouds.
- **Jupyter Notebooks:** Reproducible analysis and visualisation workflows.

---

## Directory Structure

```
Customer-Experience-Analytics-for-Fintech-Apps/
│
├── notebooks/
│   ├── web_scrapping.ipynb        # Review scraping and preprocessing
│   └── analysis.ipynb             # Sentiment and thematic analysis
│
├── scripts/
│   ├── scraper_preprocessor.py    # Custom scraping and preprocessing logic
│   └── sentiment_thematic_analysis.py  # Sentiment and thematic analysis logic
│
├── data/                          # Processed review data (CSV files)
│
├── plot_images/                   # Generated plots and word clouds (PNG files)
│
├── README.md
└── ...
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package manager)
- Recommended: [virtualenv](https://virtualenv.pypa.io/)

**Python packages:**  
Install dependencies listed in `requirements.txt`

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/nuhaminae/Customer-Experience-Analytics-for-Fintech-Apps.git
   cd Customer-Experience-Analytics-for-Fintech-Apps
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # On Unix or Mac
   .venv\Scripts\activate         # On Windows
   ```

3. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

- **Data Scraping and Preprocessing:**  
  Open and run `notebooks/web_scrapping.ipynb` to collect and prepare review data.

- **Analysis and Visualisation:**  
  Open and run `notebooks/analysis.ipynb` to perform sentiment, thematic analysis, and generate plots.

- **Database Handling:**
  Open and run `notebooks/banks_schema`, `notebooks/reviews_schema`, `notebooks/banks_values.sql`, and `notebooks/oracle.ipynb` chronologically (with caution) to save cleaned CSV to database.

---

## Analysis Workflow

1. **Scrape app reviews** using custom scripts.
2. **Preprocess the data**: remove duplicates, handle missing values, and save to CSV.
3. **Sentiment analysis**: classify each review and compute sentiment scores.
4. **Thematic analysis**: extract top keywords and assign themes.
5. **Visualise results**: generate sentiment distributions and word clouds.
6. **Save outputs**: processed data in `data/`, visualisations in `plot_images/`.

---

## Results and Visualisations

- **Processed Data:**  
  Cleaned and enriched review datasets for each bank, stored as CSV files in `data/`.

- **Plots and Word Clouds:**  
  Sentiment distributions and keyword clouds, stored as PNGs in `plot_images/`.

- **Sample Visualisations:**
  - Sentiment distribution by rating
  - Top keywords in positive/negative reviews
  - Thematic breakdowns

---

## Extending the Project

- Add more banks or fintech apps by updating the app IDs in the scraping script.
- Integrate additional data sources (e.g., Apple App Store, social media).
- Enhance NLP analysis with advanced models.
- Automate regular scraping and reporting.

---

## Contribution

Contributions, suggestions, and bug reports are welcome!  
Feel free to open an issue or submit a pull request.


## Project Status
The project is completed. Check [commit history](https://github.com/nuhaminae/Customer-Experience-Analytics-for-Fintech-Apps/commits/main/) for full detail.