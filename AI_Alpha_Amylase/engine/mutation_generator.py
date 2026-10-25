AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def generate_single_mutations(sequence):
    mutations = []

    for position, original in enumerate(sequence, start=1):
        for new_amino_acid in AMINO_ACIDS:

            if new_amino_acid != original:
                mutations.append({
                    "mutation": f"{original}{position}{new_amino_acid}",
                    "position": position,
                    "wt_aa": original,
                    "mut_aa": new_amino_acid
                })

    return mutations