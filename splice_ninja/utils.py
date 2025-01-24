from biothings_client import get_client


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
