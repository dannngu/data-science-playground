



# 🧹 Data Cleaning with Object-Oriented Programming (OOP) in Python

## 👁️ Overview

This tutorial aims to illustrate a logical **data-cleaning** workflow using **OOP** in Python, highlighting modularization techniques to enhance the creation of reusable **data pipelines**.

## 1️⃣ Prerequisites

Before running the tutorial, follow these steps:

>[!NOTE]
> **Download the dataset**
> [🔗data_employes_to_clean.csv](https://drive.google.com/file/d/10pHzZj_40CzF5GxTOlEXgGE3Gp0OzS4r/view?usp=drive_link)


1. **Save the script:** Ensure your Python script is named `data_cleaner.py`.
2. **Prepare the dataset:** Make sure the **CSV** file `data-employes-to-clean.csv` (downloaded from Google Drive) is in the same directory as `data_cleaner.py` or a correct Path of your choice.
3. **Install dependencies:** If you haven't installed them yet, run:
   ```bash
   pip install requirements.txt
   ```
4. Execute the script: Run the following command in your terminal:
   ```bash
   python data_cleaner.py
   ```

## 📁 Project Structure
- `name.md`: -> Description of the project tutorial.
- `notebooks.ipynb` or `quarto`: -> Are used for expermiental and more easy way to perfom **EDA**.
- `script.py`: -> It is used for best practices (modularization, reusability, scalability, performance, etc.) and also to create pipelines to prevent data leaks.


## 💻 What does this code do at each step?

- `load_data()`: The class is initialized with the file path and a default delimiter (comma). The `load_data()` method attempts to load the **CSV**.
    
- `observe_data()`: Use `info()`, `sample()`, and `value_counts()` to give you an overview of the dataset, including which columns have problems.
    
- `handle_missing_values()`:
  - First, we replace the **-1s** that the dataset uses as **nulls** in **'Rating'** and **'Established'** with `np.nan` so that Pandas handles them as such.
  - **'Age'** is imputed with the **median**.
  - **'Rating'** is imputed with `KNNImputer`, a more advanced method that uses the 5 closest columns (based on **'Age'** and a numerical version of **'Salary'**) to estimate the missing value, as we discussed. This demonstrates a more robust imputation.
  - **'Established'** (the year of founding) is also imputed with the **median**.

- `convert_datatypes`:
  - **'Established'** is converted to integer **(int)** after imputation.
  - **'Easy Apply'** is converted to **(category)** type for optimization.
  - **'Salary'** column, which comes as a text range (e.g. "44k-99k"), is a challenge. We create a new column **'Salary_Numeric'** by extracting the numbers with regular expressions and calculating the midpoint of the range. If there are ranges with a single number (e.g. '99k') it also handles them. The **NaN's** that may arise from this conversion (e.g. if the text was **"unknown"**) are filled with the **median**.
    
- `clean_text_column()`: 
  - Used to clear the **'Location'** column. Here, the regular expression `r',([A-Za-z]{2,})$'` looks for a comma followed by two or more letters (which would be the state/country code, such as 'In' or 'Ny') at the end of the string, and replaces them with empty to standardize the city/country name. We also eliminate extra spaces.

- `save_data_frame()`: 
  - This function saves the processed **DataFrame** to an output file while ensuring a structured and safe approach:
  - `output_directory="cleaned_data"`: Specifies the output folder, defaulting to `cleaned_data()`. If it doesn’t exist, it will be created automatically.
  - `file_name="cleaned_dataset.csv"`: Defines the output file name, which can be changed to `.xlsx` for an **Excel** format.
  - `index=False`: Prevents saving the DataFrame index as an extra column in the file.
  - Using `Path()` for safe, **cross-platform** file handling:
  - `Path(output_directory)`: Defines the output directory path.
  - `mkdir(parents=True, exist_ok=True)`: Creates the directory if it doesn’t exist, preventing errors in repeated executions.
  - `full_file_path` = `output_path / file_name`: Uses clean syntax to construct the file path.
  - `self.df.to_csv()` or `self.df.to_excel()`: Pandas methods that write the **DataFrame** to a file.

 