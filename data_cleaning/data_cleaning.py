"""
@author: Daniel Ramirez Guitron
Date: 29/05/2025

Linkdin: https://www.linkedin.com/in/danielguitron/

Github: https://github.com/dannngu

E-mail: contactguitron@gmail.com
"""


"""
Required imports
"""

import pandas as pd
import numpy as np
import re  # For regular expressions
from pathlib import Path  # A modern way to manage routes
from sklearn.impute import KNNImputer  # Advanced imputation


"""
Step: 1 Create the parent class
In this case it will be to manage the data preprocessing
"""


class DataManager:
    """
    This will be used to encapsulate loading(), observing() and cleaning() operations.
    in a DataFrame in Pandas.
    """

    def __init__(self, file_path, sep=','):
        self.file_path = file_path
        self.sep = sep
        self.df = None
        print(f"[+]DataManager initialize for the file: {self.file_path}")

    def load_data(self):
        """
        Load the CSV file in a DataFrame in Pandas.
        Handling delimiters sep=',', ';'.
        """
        try:
            self.df = pd.read_csv(self.file_path, sep=self.sep)

            # Make a copy of the original DataFrame for integrity purposes and assign it to self.df.
            self.df = self.df.copy()

            print("\n--- [✅] Data load successfully ---")
            print(
                f"[+] First 5 files of the original data set({self.file_path}):")
            print(self.df.head())
        except FileNotFoundError:
            print(f"[❗] Error: Could not find file {self.file_path}")
            self.df = None
        except Exception as e:
            print(f"[❌] An error occurred while uploading the file.: {e}")
            self.df = None
        return self.df is not None

    def observe_data(self, sample_n=5):
        """
        Displays the summary  of the data using info(), sample() and value_counts().
        """
        if self.df is None:
            print("[+] There is no data loaded to observe.")
            return

        print("\n---  [ℹ️] General Data Observation (df.info()) ---")
        self.df.info()

        print(
            f"\n--- [ℹ️] Aleatory sample of {sample_n} files (df.sample() ---)")
        self.df.sample(5)

        print(
            "\n--- [ℹ️] Counting of unique values in categoriacal/key columns (df['col'].value_counts()) ---")
        # Identify some columns that can be categorical for value_counts().
        # 'Location' and 'Easy Apply' are good candidates.
        for col in ['Location', 'Easy Apply', 'Established']:
            if col in self.df.columns and self.df[col].dtype == 'object':
                print(f"\n[+] value counts for '{col}'")
                print(self.df[col].value_counts())

    def identify_missing_values(self):
        """
        Identify and display the count of missing NaN null values.
        """
        if self.df is None:
            print(f"[+] No data loaded to display or identify missing values")
            return

        print(f"\n--- [ℹ️] Identify Missing Values ---")
        missing_counts = self.df.isnull().sum()
        missing_percentages = (self.df.isnull().sum() / len(self.df)) * 100
        missing_info = pd.DataFrame({
            'Count': missing_counts,
            'Percentage': missing_percentages
        })
        # Only shows columns with nulls.
        print(missing_info[missing_info['Count'] > 0])

    def handle_missing_values(self):
        """
        Apply strategies to handle missing values. 
        Shows the mean/median of imputation and KNN.
        """
        if self.df is None:
            print(f"[+] No data loaded to display or identify missing values")
            return

        print("\n--- [ℹ️] Handling Missing Values ---")
        # The data set has -1 as a missing value in some columns.
        # First. replace this -1 with Nan so that Pandas recognizes it.
        print(f"[+] Replazing -1 per Nan in 'Rating' and 'Established'...")
        self.df['Rating'] = self.df['Rating'].replace(-1, np.nan)
        self.df['Established'] = self.df['Established'].replace(-1, np.nan)

        # Now, we'll impute or delete according to the analysis:

        # Column 'Age': Have missing values. Is `numeric`.
        # We can use the mean or median. Median is more robust to atipical values.
        if 'Age' in self.df.columns:
            median_age = self.df['Age'].median()
            self.df['Age'] = self.df['Age'].fillna(median_age)
            print(f"Column 'Age' inputated with the mean: {median_age}")

        # Column 'Rating' have mising values (original -1, now NaN). Is numeric.
        # We'll use KNNImputer for more sophisticated imputation.
        if 'Rating' in self.df.columns and self.df['Rating'].isnull().any():
            print("Impute 'Rating' using KNNImputer...")
            # We need other numeric columns to KNN. Usage of 'Age' and 'Salary' if they are numeric.
            # First we ensure that 'Salary' is numeric.
            self.df['Salary_Lower'] = self.df['Salary'].astype(
                str).str.extract(r'(\d+)k').astype(float) * 1000
            self.df['Salary_Upper'] = self.df['Salary'].astype(
                str).str.extract(r'-(\d+)k').astype(float) * 1000
            self.df['Salary_Mid'] = (
                self.df['Salary_Lower'] + self.df['Salary_Upper']) / 2

            # We select numeric columns for KNN
            cols_for_knn = ['Age', 'Salary_Mid', 'Rating']
            temp_df_knn = self.df[cols_for_knn].copy()

            # Make sure there are no NaN in the column used fo the distance or handle them
            # To simplify the example, fill in the NaN of Salary_Mid if there are any before KNN
            if temp_df_knn['Salary_Mid'].isnull().any():
                temp_df_knn['Salary_Mid'] = temp_df_knn['Salary_Mid'].fillna(
                    temp_df_knn['Salary_Mid'].mean())

            imputer = KNNImputer(n_neighbors=5)
            # Only inputate 'Rating' keeping the other columns.
            imputed_rating = imputer.fit_transform(temp_df_knn[['Rating', 'Age', 'Salary_Mid']])[
                :, 0]  # Only for the first column = Rating

            self.df['Rating'] = imputed_rating  # Upadte the column Rating.
            print(f"[+] Column 'Rating' imputed with KNNImputer.")

            # Delete the temporal comlumns of 'Salary'
            self.df = self.df.drop(
                columns=['Salary_Lower', 'Salary_Upper', 'Salary_Mid'])

        # 'Established' column: Year founded. It is numerical.
        # We can impute with the mode if there is a very common year, or the median.
        if 'Established' in self.df.columns and self.df['Established'].isnull().any():
            median_established = self.df['Established'].median()
            self.df['Established'] = self.df['Established'].fillna(
                median_established)
            print(
                f"[+] Column 'Established' imputed by the median: {median_established}")

        print(
            "\n--- [ℹ️] Checking for Missing Values ​​After Handling ---")
        print(self.df.isnull().sum())

    def convert_datatypes(self):
        """
        Convert data types to specific columns
        """
        if self.df is None:
            print(f"[+] There is no data loaded to convert in data-types")
            return

        print("\n--- [ℹ️] Data Type Conversion---")

        # Convertir 'Established' to int (after imputation)
        if 'Established' in self.df.columns:
            # Makign sure that ther are no decimal values that don't allow the conversion to int.
            self.df['Established'] = self.df['Established'].astype(int)
            print(f"[+] Column 'Established' converted into int.")

        if 'Easy Apply' in self.df.columns:
            self.df['Easy Apply'] = self.df['Easy Apply'].astype('category')
            print(f"[+] 'Easy Apply' convert to category")

        # Convertir 'Salary' (which is a text range) to numeric (e.g: mean of the range)
        if 'Salary' in self.df.columns:
            print(
                f"[+] Converting 'Salary' (text) into a numeric value (mean of the range)...")
            # Extract the numbers using regex and calculate the average
            # '44k-99k' -> 44000, 99000 -> (44000+99000)/2
            # '55k-66k'
            # '99k' -> 99000

            def parse_salary_range(salary_str):
                if pd.isna(salary_str):
                    return np.nan
                salary_str = str(salary_str).lower().replace(
                    'k', '000').replace('$', '')
                numbers = re.findall(r'\d+', salary_str)
                if len(numbers) == 2:
                    return (float(numbers[0]) + float(numbers[1])) / 2
                elif len(numbers) == 1:
                    return float(numbers[0])
                return np.nan

            self.df['Salary_Numeric'] = self.df['Salary'].apply(
                parse_salary_range)
            # Impuate any NaN resulsatn of this convertior (e.g: 'unknown')
            if self.df['Salary_Numeric'].isnull().any():
                self.df['Salary_Numeric'] = self.df['Salary_Numeric'].fillna(
                    self.df['Salary_Numeric'].median())
            print("[+] Column 'Salary_Numeric' created and 'Salary' processed.")
            # Optional: You could remove the original 'Salary' column if you no longer need it
            # self.df.drop(columns=['Salary'], inplace=True)

        print("\n--- [ℹ️] Data Types After Conversion ---")
        print(self.df.dtypes)

    def clean_text_column(self, column_name):
        """
        Clean a column of text using regular expressions.
        For example: a 'Location' (remove extra characters or standardize)
        """
        if self.df is None:
            print("[+] No data loaded to clear text.")
            return
        if column_name not in self.df.columns:
            print(
                f"[+] The column '{column_name}' doesn't extist in the DataFrame.")
            return
        if self.df[column_name].dtype != 'object':
            print(
                f"[+] The column '{column_name}' is not of type text (object), no cleaning with regex will be applied.")

        print(
            f"\n--- [ℹ️] Cleaning Text for Column '{column_name}' with RegEx ---")

        # Cleanup example for 'Location' column
        # We want to standardize 'India,In' to 'India' or 'New York,Ny' to 'New York'
        # Pattern: ",[A-Za-z]+" at the end of string
        # Replace with empty string
        self.df[column_name] = self.df[column_name].astype(
            str).str.replace(r',([A-Za-z]{2,})$', '', regex=True)
        # Also remove extra spaces at the beginning/end.
        self.df[column_name] = self.df[column_name].astype(str).str.strip()
        print(
            f"[+] Top 5 unique values ​​in '{column_name}' after cleaning:")
        print(self.df[column_name].value_counts().head(5))

    def get_cleaned_data(self):
        """
        Return the DataFrame after all the cleaning operations.
        """
        return self.df

    def save_data(self, output_directory="preproccesed", output_file_name="cleaned_dataset.csv", index=False):
        """
        Save the clean DataFrame to a CSV or Excel file at a specific path.
        Create the directory if it does not exist.

        Args:
            output_directory (str): Name of the directory where the file will be saved (relative).
            output_file_name (str): Name of the output file.
            index (bool): Whether to include the index of the DataFrame in the file.
        """
        if self.df is None:
            print(f"[+] No DataFrame to save.")
            return

        # Using pathlib.Path (recommended).
        output_path = Path(output_directory)
        # Create the directory and its parent if it does not exist.
        output_path.mkdir(parents=True, exist_ok=True)

        # Merge the directory and the name of the file.
        full_file_path = output_path / output_file_name

        try:
            if output_file_name.endswith('.csv'):
                self.df.to_csv(full_file_path, index=index)
            elif output_file_name.endswith('.xlsx'):
                self.df.to_excel(full_file_path, index=index)
            else:
                print(
                    f"[+] Format of the file is not supported to be saved: {output_file_name}.")
                return
            print(
                f"\n--- [✅]DataFrame saved successfully in: {full_file_path} ---")
        except Exception as e:
            print(f"[+] Error saving the DataFrame: {e}")


# --- Usage of the Class DataManager ---
if __name__ == "__main__":
    data_file = Path("../data/raw/data_employes_to_clean.csv")

    # Create an instance of the class DataManager
    manager = DataManager(data_file)

    # 1. Load the data
    if manager.load_data():
        # 2. Obervate the data
        manager.observe_data()

        # 3. Idenitfy and handle missing null/NaN values.
        manager.identify_missing_values()
        manager.handle_missing_values()
        # Verify again after the manager
        print("\n--- Final Verification of missing values ---")
        print(manager.df.isnull().sum())

        # 4. Convert datatypes
        manager.convert_datatypes()

        # 5. Clean one Column of text with regexp
        # Applaying the cleaning to the 'Location' column.
        manager.clean_text_column('Location')

        # Get the clean DataFrame
        final_df = manager.get_cleaned_data()
        print("\n--- Final DataFrame Cleaned and Transformed ---")
        print(final_df.head())
        print("\nFinal datatypes:")
        print(final_df.dtypes)

        # Little final verification
        print("\n[+] Verifying the 'Salary_Numeric' before the cleaning:")
        print(final_df['Salary_Numeric'].describe())

        # Save the clean Dataset
        manager.save_data(output_directory='preproccesed',
                          output_file_name="clean_employes.csv")
        # You could also save it as Excel:
        # manager.save_data(output_directory="clean_data", file_name="clean_employees.xlsx")
