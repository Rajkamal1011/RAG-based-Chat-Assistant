import pandas as pd

# File paths
file1 = "1_102_responses_lamini.xlsx"
file2 = "103_204_responses_lamini.xlsx"

# Load the two Excel files into DataFrames
df1 = pd.read_excel(file1)
df2 = pd.read_excel(file2)

# Combine the two DataFrames using concat
combined_df = pd.concat([df1, df2], ignore_index=True)

# Save the combined DataFrame to a new Excel file
output_file = "textual_responses_lamini.xlsx"
combined_df.to_excel(output_file, index=False)

print(f"Files combined successfully into {output_file}")
