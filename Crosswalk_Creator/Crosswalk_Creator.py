
import pandas as pd
import sys
import json

from argparse import ArgumentParser
from difflib import SequenceMatcher
from scipy.optimize import linear_sum_assignment
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF, RDFS, OWL, SKOS, DC

def read_csv_with_three_columns(file_path):
    # Read CSV file
    print(f"Reading CSV file {file_path}...")
    df = pd.read_csv(file_path, delimiter=';')
    
    # Check if the CSV file has more than 3 columns
    if df.shape[1] < 3:
        raise ValueError("The csv-file should have at least three columns, but it has {df.shape[1]}")

    df_cleaned = df.dropna(how='all')

    return df_cleaned

def get_english_thing(graph, subject, thing_type):
    # Get the English version of thing type
    for thing in graph.objects(subject, thing_type):
         if isinstance(thing, Literal) and thing.language == 'en':
            return str(thing)
    return None


def read_ontology(file_path):
    g = Graph()

    # Try to parse the file with rdflib
    if file_path.lower().endswith('.jsonld'):
        g.parse(file_path, format='json-ld')
    elif file_path.lower().endswith('.xml') or file_path.lower().endswith('.rdf') or file_path.lower().endswith('.owl'): 
        g.parse(file_path, format='xml')
    else:
        raise ValueError("Unsupported file format. Please provide a .jsonld, .xml, .rdf, or .owl file.")
    
    rows = []
    
    # Iterate over all triples in the graph
    for s, p, o in g:
        if p == RDF.type and o in [OWL.Class, RDFS.Class, RDF.Property, OWL.ObjectProperty, OWL.DatatypeProperty]:
            # Find the label and description of the class
            label = get_english_thing(g, s, SKOS.prefLabel)
            if label == None:
                label = get_english_thing(g, s, RDFS.label)
            if label == None:
                 label = str(s)

            if label != None:
                description_list = []
                description_list.append(g.value(s, RDFS.comment))
                description_list.append(get_english_thing(g, s, SKOS.definition))
                description_list.append(g.value(s, RDFS.isDefinedBy))
                description_list.append(get_english_thing(g, s, DC.description))

                description = ""
                for i in description_list:
                    if i != None:
                        description = description + i

                rows.append((label, str(s), description))
    
    schema_list = pd.DataFrame(rows)

    print(f"{schema_list}")

    return schema_list

def read_json_schema(file_path):
    """
    Reads a metadata schema from a JSON file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The metadata schema as a dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not a valid JSON.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            schema = json.load(file)

    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file {file_path} is not a valid JSON.")

    rows = []

    for key, value in schema['definitions'].items():
        if 'properties' in value:
            for i, j in value['properties'].items():
                if 'items' in j:
                    element = j['items']
                    if 'properties' in element:
                        for n, m in value['properties'].items():
                            if 'items' in m:
                                element = m['items']
                            else:
                                element = m
                            name = element.get('title', 'No title')
                            description = element.get('description', 'No description')
                            type_description = element.get('type', 'No type')
                            rows.append((name, description, type_description))
                else:
                    element = j
                    
                name = element.get('title', 'No title')
                description = element.get('description', 'No description')
                type_description = element.get('type', 'No type')
                rows.append((name, description, type_description))

    df = pd.DataFrame(rows, columns=['Name', 'Description', 'Type'])
    print(f"{df}")

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
        if comparison_matrix.iloc[row_ind[i], col_ind[i]] > 0.3:
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
    parser.add_argument("-A", "--schenmaA", dest="schemaA", help="Schema to compare to (as csv)")
    parser.add_argument("-B", "--schenmaB", dest="schemaB", help="Other schema which will be compared to schema A (as csv, jsonld or xml)")
    parser.add_argument("-O", "--output", dest="output")
    args = parser.parse_args()

    # Step 0: Read in both CSV files
    if args.schemaA.lower().endswith('.csv'):
        schemaA = read_csv_with_three_columns(args.schemaA)
    else:
        raise ValueError("File for schemaA is not a csv: {args.schemaA}")

    if args.schemaB.lower().endswith('.jsonld') or args.schemaB.lower().endswith('.xml') or args.schemaB.lower().endswith('.owl'):
        schemaB = read_ontology(args.schemaB)
    elif args.schemaB.lower().endswith('.csv'):
        schemaB = read_csv_with_three_columns(args.schemaB) 
    elif args.schemaB.lower().endswith('.json'):
        schemaB = read_json_schema(args.schemaB)
    else: 
        raise ValueError("File for schemaA is not a csv: {args.schemaA}")

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