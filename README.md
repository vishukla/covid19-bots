# COVID19 bots

This repository contains supplementary code for the following paper:
>Social media bots as active disseminators of COVID-19 news (currently under review)<br>
>DOI: [10.2196/preprints.22292](https://doi.org/10.2196/preprints.22292)
>
>Ahmed Al-Rawi, Vishal Shukla

This code is being used to perform analyses on COVID19 Twitter dataset collected by Twitter Capture and Analysis Toolset (TCAT). It generates files for the identified active users, Botometer scores for the user handles, terms, hashtags, emojis and sentiment for the tweets.

### Requirements

This code has the following listed requirements to run successfully.
- Python 3.x
- Apache Spark

To install Python package dependencies, use `pip install -r requirements.txt` command.

### Structure

The collected Twitter data is expected to be in `data` directory. Same directory will be used by the script to write generated output files. The entire dataset will be made available upon request.

```{bash}
.
├── data
│   ├── tcat_COVID-20200403-20200405------------fullExport--a04e82a9e6.csv # Collected data files
│   ├── tcat_COVID-20200404-20200405------------fullExport--a04e82a9e6.csv  
│   ├── tcat_COVID-20200412-20200414------------fullExport--ecc23817f6.csv
.   .
.   .
.   .
│   └── tcat_COVID-20200414-20200416------------fullExport--ecc23817f6.csv
├── LICENSE
├── README.md
├── requirements.txt
└── src
    ├── bots.py
    ├── main.py
    └── udfs.py
```

### Usage

```{bash}
python ./src/main.py
```
