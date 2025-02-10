import numpy as np
from biothings_client import get_client
import requests
from itertools import combinations


def one_hot_encode_dna(sequence, dtype=np.float32):
    """Efficient one-hot encoding of a DNA sequence.

    Args:
        sequence (str): A DNA sequence (uppercase, e.g., 'ACGTN').

    Returns:
        np.ndarray: A 2D array of shape (len(sequence), 4), where each row is a one-hot vector.
    """
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    seq_array = np.array(
        [mapping.get(base, -1) for base in sequence]
    )  # Map bases to indices (-1 for unknown bases)

    one_hot = np.zeros((len(sequence), 4), dtype=dtype)
    valid_idx = seq_array >= 0  # Ignore 'N' or unknown bases
    one_hot[np.arange(len(sequence))[valid_idx], seq_array[valid_idx]] = 1

    return one_hot


def get_ensembl_gene_id_biothings(gene_name):
    mg = get_client("gene")  # Create a MyGene.info gene client

    # Query with alias support
    result = mg.query(gene_name, species="human", fields="ensembl.gene,alias,symbol")

    if "hits" in result and result["hits"]:
        for hit in result["hits"]:
            # Check if the Ensembl ID is available
            if "ensembl" in hit:
                if isinstance(hit["ensembl"], list):
                    return [i["gene"] for i in hit["ensembl"]]  # Handle multiple IDs
                else:
                    return hit["ensembl"]["gene"]

    return None  # Return None if not found


def get_ensembl_gene_id_hgnc_with_alias(gene_name):
    url = f"https://rest.genenames.org/search/{gene_name}"
    headers = {"Accept": "application/json"}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        if data["response"]["numFound"] > 0:
            all_symbols = []
            all_hgnc_ids = []
            all_ensembl_ids = []
            all_scores = []
            for doc in data["response"]["docs"]:
                symbol = doc.get("symbol")
                hgnc_id = doc.get("hgnc_id")
                score = doc.get("score")
                all_symbols.append(symbol)
                all_hgnc_ids.append(hgnc_id)
                all_scores.append(score)

            # get the ensemble gene id for the match with the highest score
            max_score_index = all_scores.index(max(all_scores))
            hgnc_id = all_hgnc_ids[max_score_index]

            url = f"https://rest.genenames.org/fetch/hgnc_id/{hgnc_id}"
            headers = {"Accept": "application/json"}
            response2 = requests.get(url, headers=headers)

            if response2.status_code == 200:
                data2 = response2.json()
                for doc2 in data2["response"]["docs"]:
                    if "ensembl_gene_id" in doc2:
                        ensembl_id = doc2["ensembl_gene_id"]
                        all_ensembl_ids.append(ensembl_id)
            else:
                print("Error in fetching Ensembl ID for HGNC ID:", hgnc_id)

            if len(all_ensembl_ids) == 1:
                return all_ensembl_ids[0]
            elif len(all_ensembl_ids) > 1:
                return all_ensembl_ids
            else:
                print("No Ensembl ID found for gene:", gene_name)
                return None
    return None


def chromosome_split(chromosome_data, split_ratio=(0.7, 0.1, 0.2)):
    # Sort chromosomes by number of points (descending order for better greedy choices)
    sorted_chromosomes = sorted(chromosome_data.items(), key=lambda x: -x[1])
    total_points = sum(chromosome_data.values())
    target_train, target_val, target_test = [r * total_points for r in split_ratio]

    # Initialize sets and current totals
    train_set, val_set, test_set = set(), set(), set()
    train_total, val_total, test_total = 0, 0, 0

    # Greedy assignment
    for chrom, points in sorted_chromosomes:
        # Calculate current deltas from the target
        train_delta = (train_total + points) - target_train
        val_delta = (val_total + points) - target_val
        test_delta = (test_total + points) - target_test

        # Assign to the split that minimizes the difference from target
        if train_delta <= val_delta and train_delta <= test_delta:
            train_set.add(chrom)
            train_total += points
        elif val_delta <= test_delta:
            val_set.add(chrom)
            val_total += points
        else:
            test_set.add(chrom)
            test_total += points

    print(f"Train proportion = {train_total*100.0/total_points}")
    print(f"Val proportion = {val_total*100.0/total_points}")
    print(f"Test proportion = {test_total*100.0/total_points}")

    return {"train": train_set, "val": val_set, "test": test_set}
