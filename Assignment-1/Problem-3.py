import pandas as pd
import numpy as np

# Step 1: Create the DataFrame
np.random.seed(42)
names = ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace', 'Helen', 'Ivan', 'Judy']
subjects = ['Math', 'Science', 'English', 'Math', 'Science', 'English', 'Math', 'Science', 'English', 'Math']
scores = np.random.randint(50, 101, size=10)

df = pd.DataFrame({
    'Name': names,
    'Subject': subjects,
    'Score': scores,
    'Grade': ''  # initially empty
})

# Step 2: Assign grades based on score
def assign_grade(score):
    if score >= 90:
        return 'A'
    elif score >= 80:
        return 'B'
    elif score >= 70:
        return 'C'
    elif score >= 60:
        return 'D'
    else:
        return 'F'

df['Grade'] = df['Score'].apply(assign_grade)

# Step 3: Print DataFrame sorted by Score (descending)
sorted_df = df.sort_values(by='Score', ascending=False)
print("DataFrame sorted by Score (descending):\n", sorted_df)

# Step 4: Average score for each subject
avg_scores = df.groupby('Subject')['Score'].mean()
print("\nAverage score per subject:\n", avg_scores)

# Step 5: Function to filter students with Grade A or B
def pandas_filter_pass(dataframe):
    return dataframe[dataframe['Grade'].isin(['A', 'B'])]

# Apply filter
passed_students = pandas_filter_pass(df)
print("\nStudents with Grade A or B:\n", passed_students)
