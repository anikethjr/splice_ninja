from biothings_client import get_client
import requests


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
            for doc in data["response"]["docs"]:
                symbol = doc.get("symbol")
                hgnc_id = doc.get("hgnc_id")
                all_symbols.append(symbol)
                all_hgnc_ids.append(hgnc_id)

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
