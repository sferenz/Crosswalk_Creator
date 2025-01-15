
import pandas as pd
import sys

from argparse import ArgumentParser
from difflib import SequenceMatcher
from scipy.optimize import linear_sum_assignment

def read_csv_with_three_columns(file_path):
    # Read CSV file
    print(f"Reading CSV file {file_path}...")
    df = pd.read_csv(file_path, delimiter=';')
    
    # Check if the CSV file has more than 3 columns
    if df.shape[1] < 3:
        raise ValueError("The csv-file should have at least three columns, but it has {df.shape[1]}")
    
    return df

def create_comparison_matrix(schemaA, schemaB):
    # Create a DataFrame with the length of schemaA as rows and length of schemaB as columns
    matrix = pd.DataFrame(index=range(len(schemaA)), columns=range(len(schemaB)))
    for i in range(len(schemaA)):
        for j in range(len(schemaB)):
            # Calculate the similarity between the two rows
            similarity = calculate_similarity(schemaA.iloc[i,0], schemaB.iloc[j,0])*0.5 + calculate_similarity(schemaA.iloc[i,2], schemaB.iloc[j,2])*0.25
            matrix.iloc[i, j] = similarity

    return matrix

def calculate_similarity(stringA, stringB):
    # Calculate the similarity between two strings
    return SequenceMatcher(None, stringA, stringB).ratio()

def find_best_matching(comparison_matrix):
    # Convert the similarity matrix to a cost matrix
    cost_matrix = comparison_matrix.max().max() - comparison_matrix

    # Use the Hungarian algorithm to find the best matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    return row_ind, col_ind

def create_decision_list(comparison_matrix):
    row_ind, col_ind = find_best_matching(comparison_matrix)

    # Create a DataFrame for non_matched_elements in B
    non_machted_elements = pd.DataFrame(index=comparison_matrix.columns, columns=[0])
    non_machted_elements[:] = 1

    # Create a DataFrame for matched_elements from A
    decision_list = pd.DataFrame(index=comparison_matrix.index, columns=[0])
    decision_list[:] = -1
    
    for i in range(len(row_ind)):
        if comparison_matrix.iloc[row_ind[i], col_ind[i]] > 0.5:
            decision_list.iloc[row_ind[i], 0] = col_ind[i]
            non_machted_elements.iloc[col_ind[i], 0] = 0

    return decision_list, non_machted_elements


def create_output_list(decision_list, non_matched_elements, comparison_matrix, schemaA, schemaB):
    zero_columns = non_matched_elements.sum().sum()

    output_list = pd.DataFrame(index=range(len(schemaA)+zero_columns), columns=range(7))
   
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
    i = len(schemaA)  # Counter for the output_list
    for j in range(len(non_matched_elements)):
         if non_matched_elements.iloc[j, 0] == 1:
            output_list.iloc[i, 3:6] = schemaB.iloc[j, :3]
            output_list.iloc[i, 6] = 0
            i += 1
             
    return output_list
      
def main():
    parser = ArgumentParser()
    parser.add_argument("-A", "--schenmaA", dest="schemaA")
    parser.add_argument("-B", "--schenmaB", dest="schemaB")
    parser.add_argument("-O", "--output", dest="output")
    args = parser.parse_args()

    # Step 0: Read in both CSV files
    schemaA = read_csv_with_three_columns(args.schemaA)
    schemaB = read_csv_with_three_columns(args.schemaB)
    print(f"Read both input files")

    # Step 1: Create comparison matrix
    comparison_matrix = create_comparison_matrix(schemaA, schemaB)
    print(f"Created comparison matrix")

    # Step 2: Reduce comparison matrix to decision list
    decision_list, non_matched_elements = create_decision_list(comparison_matrix)
    print(f"Created decision list")

    # Step 3: Create output list
    output_list = create_output_list(decision_list, non_matched_elements, comparison_matrix, schemaA, schemaB)
    print(f"Created output list")

    # Step 4: Save output list to CSV
    output_list.to_csv(args.output, index=False, sep = ";")
    print(f"Output list saved to {args.output}")


if __name__ == "__main__":
    sys.exit(main())