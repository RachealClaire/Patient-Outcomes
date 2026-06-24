import pandas as pd
import numpy as np

# Load your dataset
df = ml_data.copy()

# Function to create codebook
def generate_codebook(data):

    codebook = []

    for col in data.columns:
        
        dtype = data[col].dtype
        missing = data[col].isna().sum()
        unique = data[col].nunique()

        entry = {
            "Variable_Name": col,
            "Data_Type": str(dtype),
            "Missing_Values": missing,
            "Percent_Missing": round((missing / len(data)) * 100, 2),
            "Unique_Values": unique
        }

        # Numeric variables summary
        if pd.api.types.is_numeric_dtype(data[col]):
            entry["Mean"] = data[col].mean()
            entry["Std"] = data[col].std()
            entry["Min"] = data[col].min()
            entry["Max"] = data[col].max()
            entry["Example_Values"] = np.nan

        # Categorical variables summary
        else:
            entry["Mean"] = np.nan
            entry["Std"] = np.nan
            entry["Min"] = np.nan
            entry["Max"] = np.nan
            entry["Example_Values"] = ", ".join(map(str, data[col].dropna().unique()[:5]))

        codebook.append(entry)

    codebook_df = pd.DataFrame(codebook)

    return codebook_df


# Generate codebook
codebook = generate_codebook(df)

# Save to Excel
codebook.to_excel("Outputs/data_codebook.xlsx", index=False)

print("Codebook generated successfully!")