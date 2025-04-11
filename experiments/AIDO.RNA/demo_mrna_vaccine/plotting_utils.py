import os
import torch
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from umap import UMAP
import py3Dmol
from Bio import PDB
import numpy as np

def prediction_histogram(background_pred: str, gene_pred: str = None, mutated_pred: str = None, mutated_ids: str = None, background_pred_var: str = 'labels'):
    # test_predictions = pd.read_csv('test_predictions/test_predictions.tsv', sep='\t')
    # test_predictions = pd.read_csv('efficiency_test_predictions/test_predictions.tsv', sep='\t')
    test_predictions = pd.read_csv(background_pred, sep='\t')
    fig = px.histogram(test_predictions, x=background_pred_var, nbins=50)

    if gene_pred:
        gene_predictions = pd.read_csv(gene_pred, sep='\t')
        fig.add_vline(x=gene_predictions['predictions'].values[0], line_dash="dash", line_color="red")

    if mutated_pred:
        mutant_predictions = pd.read_csv(mutated_pred, sep='\t')
        mutant_ids = pd.read_csv(mutated_ids, sep='\t')
        mutant_predictions = mutant_predictions.merge(mutant_ids, right_on='sequence', left_on='sequences')
        mutant_predictions.drop_duplicates(subset='sequences', inplace=True)
        mutant_predictions['height'] = 0
        fig.add_trace(px.scatter(mutant_predictions, x='predictions', y='height', color='predictions', hover_name='id').data[0])
    fig.show()

def prediction_embeddings(
        gene_pred: str,
        mutated_pred,
        mutated_ids,
        mutated_embeddings,
        gene_embedding = None,
    ):
    # test_predictions = pd.read_csv('test_predictions/test_predictions.tsv', sep='\t')
    # test_predictions = pd.read_csv('efficiency_test_predictions/test_predictions.tsv', sep='\t')
    gene_predictions = pd.read_csv(gene_pred, sep='\t')
    mutant_predictions = pd.read_csv(mutated_pred, sep='\t')
    mutant_ids = pd.read_csv(mutated_ids, sep='\t')
    mutant_predictions = mutant_predictions.merge(mutant_ids, right_on='sequence', left_on='sequences')
    mutant_predictions.drop_duplicates(subset='sequences', inplace=True)
    mutant_predictions['height'] = -1

    # 4
    gene_embeddings = torch.load(gene_embedding)
    gene_embeddings['mean_embeddings'] = gene_embeddings['mean_embeddings'].numpy().tolist()
    gene_embeddings = pd.DataFrame(gene_embeddings)
    gene_df = gene_predictions.merge(gene_embeddings, on='sequences', how='left')
    gene_df.drop_duplicates(subset='sequences', inplace=True)

    mutated_embeddings = torch.load(mutated_embeddings)
    mutated_embeddings['mean_embeddings'] = mutated_embeddings['mean_embeddings'].numpy().tolist()
    mutated_embeddings = pd.DataFrame(mutated_embeddings)
    mutated_df = mutant_predictions.merge(mutated_embeddings, on='sequences', how='left')
    mutated_df.drop_duplicates(subset='sequences', inplace=True)

    # UMAP of mean_embeddings colored by predictions
    umap = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean')
    stacked_df = pd.concat([gene_df, mutated_df], axis=0)
    umap_embeddings = umap.fit_transform(stacked_df['mean_embeddings'].tolist())
    gene_df['umap1'] = umap_embeddings[:len(gene_df)][:, 0]
    gene_df['umap2'] = umap_embeddings[:len(gene_df)][:, 1]
    mutated_df['umap1'] = umap_embeddings[len(gene_df):][:, 0]
    mutated_df['umap2'] = umap_embeddings[len(gene_df):][:, 1]

    mutated_df['deviations'] = np.abs(mutated_df['predictions'] - gene_predictions['predictions'].values[0])
    mutated_df = mutated_df.sort_values('deviations', ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mutated_df['umap1'], y=mutated_df['umap2'], mode='markers', marker=dict(color=mutated_df['predictions'], colorscale='Viridis'), text=mutated_df['id'], name='Mutated'))
    # fig.add_trace(go.Scatter(x=mutated_df['umap1'], y=mutated_df['umap2'], mode='markers', marker=dict(color=mutated_df['predictions'], colorscale='Viridis', colorbar=dict(thickness=20)), text=mutated_df['id'], name='Mutated'))
    fig.add_trace(go.Scatter(x=gene_df['umap1'], y=gene_df['umap2'], mode='markers', marker=dict(color='red', size=10), name='Wild Type'))
    fig.update_layout(xaxis_title='UMAP1', yaxis_title='UMAP2')
    # make square
    fig.update_layout(
        autosize=False,
        width=600,
        height=600,
    )
    fig.show()


def show_protein(file):
    view = py3Dmol.view(query='pdb')
    with open(file, 'r') as f:
        view.addModel(f.read(), 'pdb')
    view.setStyle({'cartoon': {'color': 'spectrum'}})
    view.zoomTo()
    view.show()


def align_proteins(file1, file2):
    """Aligns two PDB structures based on C-alpha atoms."""
    parser = PDB.PDBParser(QUIET=True)
    structure1 = parser.get_structure("struct1", file1)
    structure2 = parser.get_structure("struct2", file2)

    super_imposer = PDB.Superimposer()

    # Extract C-alpha atoms for alignment
    atoms1 = [atom for atom in structure1.get_atoms() if atom.get_id() == 'CA']
    atoms2 = [atom for atom in structure2.get_atoms() if atom.get_id() == 'CA']

    min_atoms = min(len(atoms1), len(atoms2))
    if min_atoms == 0:
        raise ValueError("No C-alpha atoms found in one of the structures.")

    # Align the second structure to the first
    super_imposer.set_atoms(atoms2[:min_atoms],atoms1[:min_atoms])
    super_imposer.apply(structure2)  # **Apply rotation and translation**

    return structure1, structure2

def translate_structure(structure, offset):
    """Move the structure along the x-axis after alignment."""
    for atom in structure.get_atoms():
        coord = atom.get_coord()
        atom.set_coord(coord + np.array([offset, 0, 0]))  # Move along x-axis

def find_backbone_differences(structure1, structure2, threshold=2.0):
    """Find backbone differences based on atom coordinates."""
    backbone_atoms = {'N', 'C', 'O', 'CA'}
    diff_residues = []

    res1 = {res.get_id()[1]: res for res in structure1.get_residues()}
    res2 = {res.get_id()[1]: res for res in structure2.get_residues()}

    for res_id in res1.keys() & res2.keys():
        atoms1 = {atom.get_id(): atom.get_coord() for atom in res1[res_id] if atom.get_id() in backbone_atoms}
        atoms2 = {atom.get_id(): atom.get_coord() for atom in res2[res_id] if atom.get_id() in backbone_atoms}

        if len(atoms1) == len(atoms2):  # Ensure same number of backbone atoms
            for atom_id in atoms1:
                distance = np.linalg.norm(atoms1[atom_id] - atoms2[atom_id])
                if distance > threshold:
                    diff_residues.append(res_id)
                    break  # Only need to mark the residue once

    return diff_residues

def show_protein_comparison(file1, file2):
    """Visualizes two aligned protein structures side by side, highlighting backbone differences."""
    structure1, structure2 = align_proteins(file1, file2)

    # **Align first, then move second structure**
    translate_structure(structure2, offset=50.0)

    view = py3Dmol.view(width=1000, height=600)

    # Load first structure (left, blue)
    with open(file1, 'r') as f:
        view.addModel(f.read(), 'pdb')
    view.setStyle({'model': 0}, {'cartoon': {'color': 'blue'}})

    # Load second structure (right, red)
    with open(file2, 'r') as f:
        view.addModel(f.read(), 'pdb')
    view.setStyle({'model': 1}, {'cartoon': {'color': 'red'}})

    # Highlight backbone differences
    diff_residues = find_backbone_differences(structure1, structure2)
    for res_id in diff_residues:
        # view.addStyle({'resi': res_id, 'model': 0}, {'cartoon': {'color': 'yellow'}})  # Left differences
        view.addStyle({'resi': res_id, 'model': 1}, {'cartoon': {'color': 'orange'}})  # Right differences

    view.zoomTo()
    return view.show()


def generate_ct_files(rna_sequence, output_prefix, chunk_size=1_000):
    num_chunks = (len(rna_sequence) + chunk_size - 1) // chunk_size
    
    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min((chunk_idx + 1) * chunk_size, len(rna_sequence))
        chunk_seq = rna_sequence[start:end]
        output_file = f"{output_prefix}_chunk{chunk_idx + 1}.ct"
        
        with open(output_file, 'w') as f:
            f.write(f"{len(chunk_seq)} {output_prefix}_chunk{chunk_idx + 1}\n")
            for i, nucleotide in enumerate(chunk_seq):
                index = i + 1
                prev_index = index - 1 if index > 1 else 0
                next_index = index + 1 if index < len(chunk_seq) else 0
                pair_index = 0  # No base pairs in this dummy structure
                f.write(f"{index}\t{nucleotide}\t{prev_index}\t{next_index}\t{pair_index}\t{index}\n")
        print(f"CT file saved as {output_file}")


def ct_to_dot_bracket(directory, output_file, seq_len, chunk_size=1_000):
    input_files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".ct")])
    npy_files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".npy")])
    ss_matrix_full = np.zeros([seq_len, seq_len], dtype=np.float32)
    merged_nucleotides = ""
    merged_ss = ""
    
    for idx, ct_file in enumerate(input_files):
        ss_matrix = np.load(npy_files[idx]).astype(np.float32)
        ss_matrix_full[idx*chunk_size:(idx+1)*chunk_size, idx*chunk_size:(idx+1)*chunk_size] = ss_matrix
        
        with open(ct_file, 'r') as f:
            lines = f.readlines()[1:]  # Skip header
        
        sequence_length = len(lines)
        structure = ["."] * sequence_length
        nucleotides = ""
        pairings = {}
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 6:
                # index, _, _, _, pair, _ = map(int, parts)
                index = int(parts[0])
                pair = int(parts[4])
                merged_nucleotides += str(parts[1])
                if pair > 0:
                    pairings[index - 1] = pair - 1
        
        for i, j in pairings.items():
            if j > i:
                structure[i] = "("
                structure[j] = ")"
        
        merged_ss += "".join(structure)

    with open(output_file, 'w') as f:
        f.write(">infered\n" + merged_nucleotides + "\n" + merged_ss + "\n")
    
    print(f"Dot-bracket notation saved as {output_file}")

    return merged_nucleotides, merged_ss, ss_matrix_full


def dot_bracket_to_matrix(dot_bracket):
    """Convert dot-bracket notation to a binary pairing matrix."""
    length = len(dot_bracket)
    matrix = np.zeros((length, length), dtype=int)
    stack = []
    
    for i, char in enumerate(dot_bracket):
        if char == '(':
            stack.append(i)  # Push index of opening bracket
        elif char == ')':
            if stack:
                j = stack.pop()  # Get the matching opening bracket index
                matrix[i, j] = 1
                matrix[j, i] = 1  # Symmetric matrix

                # for k in range(10):
                #     matrix[max(0,i-k), max(0,j-k)] = 1
                #     matrix[max(0,j-k), max(0,i-k)] = 1  # Symmetric matrix

                #     matrix[min(length, i+k), min(length, j+k)] = 1
                #     matrix[min(length, j+k), min(length, i+k)] = 1  # Symmetric matrix
    
    return matrix


def display_complex_cif(file_path, width=800, height=600):
    # Read the CIF file content
    with open(file_path, 'r') as f:
        cif_data = f.read()
    
    # Create a py3Dmol view with the specified dimensions
    view = py3Dmol.view(width=width, height=height)
    
    # Add the model from the CIF data
    view.addModel(cif_data, "cif")

    # Style 
    view.setStyle({'chain': 'A0'}, {'cartoon': {'color': '#f5ae4c'}})
    view.setStyle({'chain': 'B0'}, {'cartoon': {'color': '#58db9a'}})
    view.setStyle({'chain': 'C0'}, {'cartoon': {'color': '#a887de'}})
    
    # Adjust the view to fit the model
    view.zoomTo()
    
    # Display the viewer in the notebook
    return view.show()