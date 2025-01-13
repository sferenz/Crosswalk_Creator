
import pandas as pd
import sys

from argparse import ArgumentParser
from difflib import SequenceMatcher

def read_csv_with_three_columns(file_path):
    # Read CSV file
    df = pd.read_csv(file_path)
    
    # Check if the CSV file has exactly 3 columns
    if df.shape[1] != 3:
        raise ValueError("Die CSV-Datei muss genau 3 Spalten haben")
    
    return df

def create_comparison_matrix(schemaA, schemaB):
    # Create a DataFrame with the length of schemaA as rows and length of schemaB as columns
    matrix = pd.DataFrame(index=range(len(schemaA)), columns=range(len(schemaB)))
    for i in range(len(schemaA)):
        for j in range(len(schemaB)):
            # Calculate the similarity between the two rows
            similarity = calculate_similarity(schemaA.iloc[i,0], schemaB.iloc[j,0]) + calculate_similarity(schemaA.iloc[i,2], schemaB.iloc[j,2])*0.5
            matrix.iloc[i, j] = similarity
            matrix.iloc[j, i] = similarity

    return matrix

def calculate_similarity(stringA, stringB):
    # Calculate the similarity between two strings
    return SequenceMatcher(None, stringA, stringB).ratio()

def create_decision_list(comparison_matrix):
    # Create a DataFrame with the same size as comparison_matrix
    decision_matrix_1 = pd.DataFrame(index=comparison_matrix.index, columns=comparison_matrix.columns)
    decision_matrix_2 = pd.DataFrame(index=comparison_matrix.index, columns=comparison_matrix.columns)
    
    # Set all values in the decision_matrix to 0
    decision_matrix_1[:] = 0
    decision_matrix_2[:] = 0

    # Create a DataFrame for non_matched_elements in B
    non_machted_elements = pd.DataFrame(index=range(
        len(comparison_matrix.columns)), columns=1)
    non_machted_elements[:] = 0

    # Create a DataFrame for matched_elements from A
    decision_list = pd.DataFrame(index=range(len(comparison_matrix.index)), columns=1)

    # Iterate over all columns in the comparison_matrix (each element in schemaB should be used once)
    for i in range(len(comparison_matrix.columns)):
        best_match_position = -1
        best_match_value = 0
        for j in range(len(comparison_matrix.index)):
            # Check if the similarity between the two rows is higher than the current best match
            if comparison_matrix.iloc[i, j] > best_match_value:
                best_match_position = j
                best_match_value = comparison_matrix.iloc[i, j]

        if best_match_position != -1:
            # Set the best match to 1
            decision_matrix_1.iloc[i, best_match_position] = best_match_value

    # Iterate over all rows in the comparison_matrix (each element in schemaA should only get one element from schemaB)
    for i in range(len(comparison_matrix.index)):
        best_match_position = -1
        best_match_value = 0
        for j in range(len(comparison_matrix.columns)):
            # Check if the similarity between the two rows is higher than the current best match
            if decision_matrix_1.iloc[i, j] > best_match_value:
                best_match_position = j
                best_match_value = decision_matrix_1.iloc[i, j]

        if best_match_position != -1:
            # Set the best match to 1
            decision_matrix_2.iloc[i, best_match_position] = 1
        decision_list.iloc[i, 0] = best_match_position

    for cols in decision_matrix_2:
        if (decision_matrix_2[cols] == 0).all():
            non_machted_elements.iloc[cols, 0] = 1

       
    return decision_list, non_machted_elements


def create_output_list(decision_list, non_matched_elements, comparison_matrix, schemaA, schemaB):
    zero_columns = non_matched_elements.sum().sum()

    output_list = pd.DataFrame(index=range(len(schemaA)+zero_columns), columns=7)
   
    for i in range(len(schemaA)):
        # Copy schemaA to output_list
        output_list.iloc[i, :3] = schemaA.iloc[i, :3]

        # Add best matched schemaB to output_list
        best_match_position = decision_list.iloc[i, 0]
        if best_match_position != -1:
            output_list.iloc[i, 3:6] = schemaB.iloc[best_match_position, :3]
            output_list.iloc[i, 6] = comparison_matrix.iloc[i,
                                                            best_match_position]

    # Add non-matched schemaB to output_list
    i = 0  # Counter for the output_list
    for j in range(len(non_matched_elements)):
         if non_matched_elements.iloc[j] == 1:
            output_list.iloc[i, 3:6] = schemaB.iloc[j, :3]
            output_list.iloc[i, 6] = 0
            i += 1
             
    return output_list
      
def main():
    parser = ArgumentParser()
    parser.add_argument("-A", "--schenmaA", dest="schemaA")
    parser.add_argument("-B", "--schenmaB", dest="schemaB")
    parser.add_argument("-o", "--output", dest="output")
    args = parser.parse_args()

    # Step 0: Read in both CSV files
    schemaA = read_csv_with_three_columns(args.schemaA)
    schemaB = read_csv_with_three_columns(args.schemaB)

    # Step 1: Create comparison matrix
    comparison_matrix = create_comparison_matrix(schemaA, schemaB)

    # Step 2: Reduce comparison matrix to decision matrix
    decision_list, non_matched_elements = create_decision_list(comparison_matrix)

    # Step 3: Create output list
    output_list = create_output_list(decision_list, non_matched_elements, comparison_matrix, schemaA, schemaB)

    # Step 4: Save output list to CSV
    output_list.to_csv(args.output, index=False)
    print(f"Output list saved to {args.output}")


if __name__ == "__main__":
    sys.exit(main())