

from collections import Counter
from typing import Optional, Union

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray, get_chain_starts, get_residue_starts
from biotite.structure.io.pdbx import convert as pdbx_convert
from biotite.structure.molecules import get_molecule_indices

from fold.data import ccd
from fold.data.ccd import get_ccd_ref_info
from fold.data.constants import (
    DNA_STD_RESIDUES,
    PROT_STD_RESIDUES_ONE_TO_THREE,
    RES_ATOMS_DICT,
    RNA_STD_RESIDUES,
    STD_RESIDUES,
)
from fold.data.utils import get_starts_by

from fold.utils.logger import Logger

logger = Logger.logger

# Ignore inter residue metal coordinate bonds in mmcif _struct_conn
if "metalc" in pdbx_convert.PDBX_COVALENT_TYPES:  # for reload
    pdbx_convert.PDBX_COVALENT_TYPES.remove("metalc")

class AddAtomArrayAnnot(object):
    """
    The methods in this class are all designed to add annotations to an AtomArray
    without altering the information in the original AtomArray.
    """

    @staticmethod
    def add_token_mol_type(
        atom_array: AtomArray, sequences: dict[str, str]=None,
    ) -> AtomArray:
        """
        Add molecule types in atom_arry.mol_type based on ccd pdbx_type.

        Args:
            atom_array (AtomArray): Biotite AtomArray object.
            sequences (dict[str, str]): A dict of label_entity_id --> canonical_sequence

        Return
            AtomArray: add atom_arry.mol_type = "protein" | "rna" | "dna" | "ligand"
        """
        mol_types = np.zeros(len(atom_array), dtype="U7")
        starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
        for start, stop in zip(starts[:-1], starts[1:]):
            entity_id = atom_array.label_entity_id[start]
            if (sequences is not None) and (entity_id not in sequences):
                # non-poly is ligand
                mol_types[start:stop] = "ligand"
                continue
            res_name = atom_array.res_name[start]

            mol_types[start:stop] = ccd.get_mol_type(res_name)

        atom_array.set_annotation("mol_type", mol_types)
        return atom_array

    @staticmethod
    def add_atom_mol_type_mask(atom_array: AtomArray) -> AtomArray:
        """
        Mask indicates is_protein / rna / dna / ligand.
        It is atom-level which is different with paper (token-level).
        The type of each atom is determined based on the most frequently
        occurring type in the chain to which it belongs.

        Args:
            atom_array (AtomArray): Biotite AtomArray object

        Returns:
            AtomArray: Biotite AtomArray object with
                       "is_ligand", "is_dna", "is_rna", "is_protein" annotation added.
        """
        # it should be called after mmcif_parser.add_token_mol_type
        chain_starts = struc.get_chain_starts(atom_array, add_exclusive_stop=True)
        chain_mol_type = []
        for start, end in zip(chain_starts[:-1], chain_starts[1:]):
            mol_types = atom_array.mol_type[start:end]
            mol_type_count = Counter(mol_types)
            most_freq_mol_type = max(mol_type_count, key=mol_type_count.get)
            chain_mol_type.extend([most_freq_mol_type] * (end - start))
        atom_array.set_annotation("chain_mol_type", chain_mol_type)

        for type_str in ["ligand", "dna", "rna", "protein"]:
            mask = (atom_array.chain_mol_type == type_str).astype(int)
            atom_array.set_annotation(f"is_{type_str}", mask)
        return atom_array

    @staticmethod
    def add_modified_res_mask(atom_array: AtomArray) -> AtomArray:
        """
        Ref: AlphaFold3 SI Chapter 5.9.3

        Determine if an atom belongs to a modified residue,
        which is used to calculate the Modified Residue Scores in sample ranking:
        Modified residue scores are ranked according to the average pLDDT of the modified residue.

        Args:
            atom_array (AtomArray): Biotite AtomArray object

        Returns:
            AtomArray: Biotite AtomArray object with
                       "modified_res_mask" annotation added.
        """
        modified_res_mask = []
        starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
        for start, stop in zip(starts[:-1], starts[1:]):
            res_name = atom_array.res_name[start]
            mol_type = atom_array.mol_type[start]
            res_atom_nums = stop - start
            if res_name not in STD_RESIDUES and mol_type != "ligand":
                modified_res_mask.extend([1] * res_atom_nums)
            else:
                modified_res_mask.extend([0] * res_atom_nums)
        atom_array.set_annotation("modified_res_mask", modified_res_mask)
        return atom_array

    @staticmethod
    def add_centre_atom_mask(atom_array: AtomArray) -> AtomArray:
        """
        Ref: AlphaFold3 SI Chapter 2.6
            • A standard amino acid residue (Table 13) is represented as a single token.
            • A standard nucleotide residue (Table 13) is represented as a single token.
            • A modified amino acid or nucleotide residue is tokenized per-atom (i.e. N tokens for an N-atom residue)
            • All ligands are tokenized per-atom
        For each token we also designate a token centre atom, used in various places below:
            • Cα for standard amino acids
            • C1′ for standard nucleotides
            • For other cases take the first and only atom as they are tokenized per-atom.

        Args:
            atom_array (AtomArray): Biotite AtomArray object

        Returns:
            AtomArray: Biotite AtomArray object with "centre_atom_mask" annotation added.
        """
        res_name = list(STD_RESIDUES.keys())
        std_res = np.isin(atom_array.res_name, res_name) & (
            atom_array.mol_type != "ligand"
        )
        prot_res = np.char.str_len(atom_array.res_name) == 3
        prot_centre_atom = prot_res & (atom_array.atom_name == "CA")
        nuc_centre_atom = (~prot_res) & (atom_array.atom_name == r"C1'")
        not_std_res = ~std_res
        centre_atom_mask = (
            std_res & (prot_centre_atom | nuc_centre_atom)
        ) | not_std_res
        centre_atom_mask = centre_atom_mask.astype(int)
        atom_array.set_annotation("centre_atom_mask", centre_atom_mask)
        return atom_array

    @staticmethod
    def add_distogram_rep_atom_mask(atom_array: AtomArray) -> AtomArray:
        """
        Ref: AlphaFold3 SI Chapter 4.4
        the representative atom mask for each token for distogram head
        • Cβ for protein residues (Cα for glycine),
        • C4 for purines and C2 for pyrimidines.
        • All ligands already have a single atom per token.

        Due to the lack of explanation regarding the handling of "N" and "DN" in the article,
        it is impossible to determine the representative atom based on whether it is a purine or pyrimidine.
        Therefore, C1' is chosen as the representative atom for both "N" and "DN".

        Args:
            atom_array (AtomArray): Biotite AtomArray object

        Returns:
            AtomArray: Biotite AtomArray object with "distogram_rep_atom_mask" annotation added.
        """
        std_res = np.isin(atom_array.res_name, list(STD_RESIDUES.keys())) & (
            atom_array.mol_type != "ligand"
        )

        # for protein std res
        std_prot_res = std_res & (np.char.str_len(atom_array.res_name) == 3)
        gly = atom_array.res_name == "GLY"
        prot_cb = std_prot_res & (~gly) & (atom_array.atom_name == "CB")
        prot_gly_ca = gly & (atom_array.atom_name == "CA")

        # for nucleotide std res
        purines_c4 = np.isin(atom_array.res_name, ["DA", "DG", "A", "G"]) & (
            atom_array.atom_name == "C4"
        )
        pyrimidines_c2 = np.isin(atom_array.res_name, ["DC", "DT", "C", "U"]) & (
            atom_array.atom_name == "C2"
        )

        # for nucleotide unk res
        unk_nuc = np.isin(atom_array.res_name, ["DN", "N"]) & (
            atom_array.atom_name == r"C1'"
        )

        distogram_rep_atom_mask = (
            prot_cb | prot_gly_ca | purines_c4 | pyrimidines_c2 | unk_nuc
        ) | (~std_res)
        distogram_rep_atom_mask = distogram_rep_atom_mask.astype(int)

        atom_array.set_annotation("distogram_rep_atom_mask", distogram_rep_atom_mask)

        assert np.sum(atom_array.distogram_rep_atom_mask) == np.sum(
            atom_array.centre_atom_mask
        )

        return atom_array

    @staticmethod
    def add_plddt_m_rep_atom_mask(atom_array: AtomArray) -> AtomArray:
        """
        Ref: AlphaFold3 SI Chapter 4.3.1
        the representative atom for plddt loss
        • Atoms such that the distance in the ground truth between atom l and atom m is less than 15 Å
            if m is a protein atom or less than 30 Å if m is a nucleic acid atom.
        • Only atoms in polymer chains.
        • One atom per token - Cα for standard protein residues
            and C1′ for standard nucleic acid residues.

        Args:
            atom_array (AtomArray): Biotite AtomArray object

        Returns:
            AtomArray: Biotite AtomArray object with "plddt_m_rep_atom_mask" annotation added.
        """
        std_res = np.isin(atom_array.res_name, list(STD_RESIDUES.keys())) & (
            atom_array.mol_type != "ligand"
        )
        ca_or_c1 = (atom_array.atom_name == "CA") | (atom_array.atom_name == r"C1'")
        plddt_m_rep_atom_mask = (std_res & ca_or_c1).astype(int)
        atom_array.set_annotation("plddt_m_rep_atom_mask", plddt_m_rep_atom_mask)
        return atom_array

    @staticmethod
    def add_ref_space_uid(atom_array: AtomArray) -> AtomArray:
        """
        Ref: AlphaFold3 SI Chapter 2.8 Table 5
        Numerical encoding of the chain id and residue index associated with this reference conformer.
        Each (chain id, residue index) tuple is assigned an integer on first appearance.

        Args:
            atom_array (AtomArray): Biotite AtomArray object

        Returns:
            AtomArray: Biotite AtomArray object with "ref_space_uid" annotation added.
        """
        # [N_atom, 2]
        chain_res_id = np.vstack((atom_array.asym_id_int, atom_array.res_id)).T
        unique_id = np.unique(chain_res_id, axis=0)

        mapping_dict = {}
        for idx, chain_res_id_pair in enumerate(unique_id):
            asym_id_int, res_id = chain_res_id_pair
            mapping_dict[(asym_id_int, res_id)] = idx

        ref_space_uid = [
            mapping_dict[(asym_id_int, res_id)] for asym_id_int, res_id in chain_res_id
        ]
        atom_array.set_annotation("ref_space_uid", ref_space_uid)
        return atom_array

    @staticmethod
    def add_cano_seq_resname(atom_array: AtomArray) -> AtomArray:
        """
        Assign to each atom the three-letter residue name (resname)
        corresponding to its place in the canonical sequences.
        Non-standard residues are mapped to standard ones.
        Residues that cannot be mapped to standard residues and ligands are all labeled as "UNK".

        Note: Some CCD Codes in the canonical sequence are mapped to three letters. It is labeled as one "UNK".

        Args:
            atom_array (AtomArray): Biotite AtomArray object

        Returns:
            AtomArray: Biotite AtomArray object with "cano_seq_resname" annotation added.
        """
        cano_seq_resname = []
        starts = struc.get_residue_starts(atom_array, add_exclusive_stop=True)
        for start, stop in zip(starts[:-1], starts[1:]):
            res_atom_nums = stop - start
            mol_type = atom_array.mol_type[start]
            resname = atom_array.res_name[start]

            one_letter_code = ccd.get_one_letter_code(resname)
            if one_letter_code is None or len(one_letter_code) != 1:
                # Some non-standard residues cannot be mapped back to one standard residue.
                one_letter_code = "X" if mol_type == "protein" else "N"

            if mol_type == "protein":
                res_name_in_cano_seq = PROT_STD_RESIDUES_ONE_TO_THREE.get(
                    one_letter_code, "UNK"
                )
            elif mol_type == "dna":
                res_name_in_cano_seq = "D" + one_letter_code
                if res_name_in_cano_seq not in DNA_STD_RESIDUES:
                    res_name_in_cano_seq = "DN"
            elif mol_type == "rna":
                res_name_in_cano_seq = one_letter_code
                if res_name_in_cano_seq not in RNA_STD_RESIDUES:
                    res_name_in_cano_seq = "N"
            else:
                # some molecules attached to a polymer like ATP-RNA. e.g.
                res_name_in_cano_seq = "UNK"

            cano_seq_resname.extend([res_name_in_cano_seq] * res_atom_nums)

        atom_array.set_annotation("cano_seq_resname", cano_seq_resname)
        return atom_array

    @staticmethod
    def remove_bonds_between_polymer_chains(
        atom_array: AtomArray, entity_poly_type: dict[str, str]
    ) -> struc.BondList:
        """
        Remove bonds between polymer chains based on entity_poly_type

        Args:
            atom_array (AtomArray): Biotite AtomArray object
            entity_poly_type (dict[str, str]): entity_id to poly_type

        Returns:
            BondList: Biotite BondList object (copy) with bonds between polymer chains removed
        """
        copy = atom_array.bonds.copy()
        polymer_mask = np.isin(
            atom_array.label_entity_id, list(entity_poly_type.keys())
        )
        i = copy._bonds[:, 0]
        j = copy._bonds[:, 1]
        pp_bond_mask = polymer_mask[i] & polymer_mask[j]
        diff_chain_mask = atom_array.chain_id[i] != atom_array.chain_id[j]
        pp_bond_mask = pp_bond_mask & diff_chain_mask
        copy._bonds = copy._bonds[~pp_bond_mask]

        # post-process after modified bonds manually
        # due to the extraction of bonds using a mask, the lower one of the two atom indices is still in the first
        copy._remove_redundant_bonds()
        copy._max_bonds_per_atom = copy._get_max_bonds_per_atom()
        return copy

    @staticmethod
    def find_equiv_mol_and_assign_ids(
        atom_array: AtomArray,
        entity_poly_type: Optional[dict[str, str]] = None,
        check_final_equiv: bool = True,
    ) -> AtomArray:
        """
        Assign a unique integer to each molecule in the structure.
        All atoms connected by covalent bonds are considered as a molecule, with unique mol_id (int).
        different copies of same molecule will assign same entity_mol_id (int).
        for each mol, assign mol_atom_index starting from 0.

        Args:
            atom_array (AtomArray): Biotite AtomArray object
            entity_poly_type (Optional[dict[str, str]]): label_entity_id to entity.poly_type.
                              Defaults to None.
            check_final_equiv (bool, optional): check if the final mol_ids of same entity_mol_id are all equivalent.

        Returns:
            AtomArray: Biotite AtomArray object with new annotations
            - mol_id: atoms with covalent bonds connected, 0-based int
            - entity_mol_id: equivalent molecules will assign same entity_mol_id, 0-based int
            - mol_residue_index: mol_atom_index for each mol, 0-based int
        """
        # Re-assign mol_id to AtomArray after break asym bonds
        if entity_poly_type is None:
            mol_indices: list[np.ndarray] = get_molecule_indices(atom_array)
        else:
            bonds_filtered = AddAtomArrayAnnot.remove_bonds_between_polymer_chains(
                atom_array, entity_poly_type
            )
            mol_indices: list[np.ndarray] = get_molecule_indices(bonds_filtered)

        # assign mol_id
        mol_ids = np.array([-1] * len(atom_array), dtype=np.int32)
        for mol_id, atom_indices in enumerate(mol_indices):
            mol_ids[atom_indices] = mol_id
        atom_array.set_annotation("mol_id", mol_ids)

        assert ~np.isin(-1, atom_array.mol_id), "Some mol_id is not assigned."
        assert len(np.unique(atom_array.mol_id)) == len(
            mol_indices
        ), "Some mol_id is duplicated."

        # assign entity_mol_id
        # --------------------
        # first atom of mol with infos in attrubites, eg: info.num_atoms, info.bonds, ...
        ref_mol_infos = []
        # perm for keep multiple chains in one mol are together and in same chain order
        new_atom_perm = []
        chain_starts = struc.get_chain_starts(atom_array, add_exclusive_stop=False)
        entity_mol_ids = np.zeros_like(mol_ids)
        for mol_id, atom_indices in enumerate(mol_indices):
            atom_indices = np.sort(atom_indices)
            # keep multiple chains-mol has same chain order in different copies
            chain_perm = np.argsort(
                atom_array.label_entity_id[atom_indices], kind="stable"
            )
            atom_indices = atom_indices[chain_perm]
            # save indices for finally re-ordering atom_array
            new_atom_perm.extend(atom_indices)

            # check mol equal, keep chain order consistent with atom_indices
            mol_chain_mask = np.isin(atom_indices, chain_starts)
            entity_ids = atom_array.label_entity_id[atom_indices][
                mol_chain_mask
            ].tolist()

            match_entity_mol_id = None
            for entity_mol_id, mol_info in enumerate(ref_mol_infos):
                # check mol equal
                # same entity_ids and same atom name will assign same entity_mol_id
                if entity_ids != mol_info.entity_ids:
                    continue

                if len(atom_indices) != len(mol_info.atom_name):
                    continue

                atom_name_not_equal = (
                    atom_array.atom_name[atom_indices] != mol_info.atom_name
                )
                if np.any(atom_name_not_equal):
                    diff_indices = np.where(atom_name_not_equal)[0]
                    query_atom = atom_array[atom_indices[diff_indices[0]]]
                    ref_atom = atom_array[mol_info.atom_indices[diff_indices[0]]]
                    logger.warning(
                        f"Two mols have entity_ids and same number of atoms, but diff atom name:\n{query_atom=}\n{  ref_atom=}"
                    )
                    continue

                # pass all checks, it is a match
                match_entity_mol_id = entity_mol_id
                break

            if match_entity_mol_id is None:  # not found match mol
                # use first atom as a placeholder for mol info.
                mol_info = atom_array[atom_indices[0]]
                mol_info.atom_indices = atom_indices
                mol_info.entity_ids = entity_ids
                mol_info.atom_name = atom_array.atom_name[atom_indices]
                mol_info.entity_mol_id = len(ref_mol_infos)
                ref_mol_infos.append(mol_info)
                match_entity_mol_id = mol_info.entity_mol_id

            entity_mol_ids[atom_indices] = match_entity_mol_id

        atom_array.set_annotation("entity_mol_id", entity_mol_ids)

        # re-order atom_array to make atoms with same mol_id together.
        atom_array = atom_array[new_atom_perm]

        # assign mol_atom_index
        mol_starts = get_starts_by(
            atom_array, by_annot="mol_id", add_exclusive_stop=True
        )
        mol_atom_index = np.zeros_like(atom_array.mol_id, dtype=np.int32)
        for start, stop in zip(mol_starts[:-1], mol_starts[1:]):
            mol_atom_index[start:stop] = np.arange(stop - start)
        atom_array.set_annotation("mol_atom_index", mol_atom_index)

        # check mol equivalence again
        if check_final_equiv:
            num_mols = len(mol_starts) - 1
            for i in range(num_mols):
                for j in range(i + 1, num_mols):
                    start_i, stop_i = mol_starts[i], mol_starts[i + 1]
                    start_j, stop_j = mol_starts[j], mol_starts[j + 1]
                    if (
                        atom_array.entity_mol_id[start_i]
                        != atom_array.entity_mol_id[start_j]
                    ):
                        continue
                    for key in ["res_name", "atom_name", "mol_atom_index"]:
                        # not check res_id for ligand may have different res_id
                        annot = getattr(atom_array, key)
                        assert np.all(
                            annot[start_i:stop_i] == annot[start_j:stop_j]
                        ), f"not equal {key} when find_equiv_mol_and_assign_ids()"

        return atom_array

    @staticmethod
    def add_tokatom_idx(atom_array: AtomArray) -> AtomArray:
        """
        Add a tokatom_idx corresponding to the residue and atom name for each atom.
        For non-standard residues or ligands, the tokatom_idx should be set to 0.

        Parameters:
        atom_array (AtomArray): The AtomArray object to which the annotation will be added.

        Returns:
        AtomArray: The AtomArray object with the 'tokatom_idx' annotation added.
        """
        # pre-defined atom name order for tokatom_idx
        tokatom_idx_list = []
        for atom in atom_array:
            atom_name_position = RES_ATOMS_DICT.get(atom.res_name, None)
            if atom.mol_type == "ligand" or atom_name_position is None:
                tokatom_idx = 0
            else:
                tokatom_idx = atom_name_position[atom.atom_name]
            tokatom_idx_list.append(tokatom_idx)
        atom_array.set_annotation("tokatom_idx", tokatom_idx_list)
        return atom_array

    @staticmethod
    def unique_chain_and_add_ids(atom_array: AtomArray) -> AtomArray:
        """
        Unique chain ID and add asym_id, entity_id, sym_id.
        Adds a number to the chain ID to make chain IDs in the assembly unique.
        Example: [A, B, A, B, C] ==> [A0, B0, A1, B1, C0]

        Args:
            atom_array (AtomArray): Biotite AtomArray object.

        Returns:
            AtomArray: Biotite AtomArray object with new annotations:
                - asym_id_int: np.array(int)
                - entity_id_int: np.array(int)
                - sym_id_int: np.array(int)
        """
        entity_id_uniq = np.sort(np.unique(atom_array.label_entity_id))
        entity_id_dict = {e: i for i, e in enumerate(entity_id_uniq)}
        asym_ids = np.zeros(len(atom_array), dtype=int)
        entity_ids = np.zeros(len(atom_array), dtype=int)
        sym_ids = np.zeros(len(atom_array), dtype=int)
        chain_ids = np.zeros(len(atom_array), dtype="U4")
        counter = Counter()
        start_indices = struc.get_chain_starts(atom_array, add_exclusive_stop=True)
        for i in range(len(start_indices) - 1):
            start_i = start_indices[i]
            stop_i = start_indices[i + 1]
            asym_ids[start_i:stop_i] = i

            entity_id = atom_array.label_entity_id[start_i]
            entity_ids[start_i:stop_i] = entity_id_dict[entity_id]

            sym_ids[start_i:stop_i] = counter[entity_id]
            counter[entity_id] += 1
            new_chain_id = f"{atom_array.chain_id[start_i]}{sym_ids[start_i]}"
            chain_ids[start_i:stop_i] = new_chain_id

        atom_array.set_annotation("asym_id_int", asym_ids)
        atom_array.set_annotation("entity_id_int", entity_ids)
        atom_array.set_annotation("sym_id_int", sym_ids)
        atom_array.chain_id = chain_ids
        return atom_array

    @staticmethod
    def add_ref_feat_info(
        atom_array: AtomArray,
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """
        Get info of reference structure of atoms based on the atom array.

        Args:
            atom_array (AtomArray): The atom array.

        Returns:
            tuple:
                ref_pos (numpy.ndarray): Atom positions in the reference conformer,
                                         with a random rotation and translation applied.
                                         Atom positions are given in Å. Shape=(num_atom, 3).
                ref_charge (numpy.ndarray): Charge for each atom in the reference conformer. Shape=(num_atom）
                ref_mask ((numpy.ndarray): Mask indicating which atom slots are used in the reference conformer. Shape=(num_atom）
        """
        info_dict = {}
        for ccd_id in np.unique(atom_array.res_name):
            # create ref conformer for each CCD ID
            ref_result = get_ccd_ref_info(ccd_id)
            if ref_result:
                for space_uid in np.unique(
                    atom_array[atom_array.res_name == ccd_id].ref_space_uid
                ):
                    if ref_result:
                        info_dict[space_uid] = [
                            ref_result["atom_map"],
                            ref_result["coord"],
                            ref_result["charge"],
                            ref_result["mask"],
                        ]
            else:
                # get conformer failed will result in an empty dictionary
                continue

        ref_mask = []  # [N_atom]
        ref_pos = []  # [N_atom, 3]
        ref_charge = []  # [N_atom]
        for atom in atom_array:
            ref_result = info_dict.get(atom.ref_space_uid)
            if ref_result is None:
                # get conformer failed
                ref_mask.append(0)
                ref_pos.append([0.0, 0.0, 0.0])
                ref_charge.append(0)

            else:
                atom_map, coord, charge, mask = ref_result
                atom_sub_idx = atom_map[atom.atom_name]
                ref_mask.append(mask[atom_sub_idx])
                ref_pos.append(coord[atom_sub_idx])
                ref_charge.append(charge[atom_sub_idx])

        ref_pos = np.array(ref_pos)
        ref_charge = np.array(ref_charge).astype(int)
        ref_mask = np.array(ref_mask).astype(int)
        return ref_pos, ref_charge, ref_mask

    @staticmethod
    def add_res_perm(
        atom_array: AtomArray,
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """
        Get permutations of each atom within the residue.

        Args:
            atom_array (AtomArray): biotite AtomArray object.

        Returns:
            list[list[int]]: 2D list of (N_atom, N_perm)
        """
        starts = get_residue_starts(atom_array, add_exclusive_stop=True)
        res_perm = []
        for start, stop in zip(starts[:-1], starts[1:]):
            res_atom = atom_array[start:stop]
            curr_res_atom_idx = list(range(len(res_atom)))

            res_dict = get_ccd_ref_info(ccd_code=res_atom.res_name[0])
            if not res_dict:
                res_perm.extend([[i] for i in curr_res_atom_idx])
                continue

            perm_array = res_dict["perm"]  # [N_atoms, N_perm]
            perm_atom_idx_in_res_order = [
                res_dict["atom_map"][i] for i in res_atom.atom_name
            ]
            perm_idx_to_present_atom_idx = dict(
                zip(perm_atom_idx_in_res_order, curr_res_atom_idx)
            )

            precent_row_mask = np.isin(perm_array[:, 0], perm_atom_idx_in_res_order)
            perm_array_row_filtered = perm_array[precent_row_mask]

            precent_col_mask = np.isin(
                perm_array_row_filtered, perm_atom_idx_in_res_order
            ).all(axis=0)
            perm_array_filtered = perm_array_row_filtered[:, precent_col_mask]

            # replace the elem in new_perm_array according to the perm_idx_to_present_atom_idx dict
            new_perm_array = np.vectorize(perm_idx_to_present_atom_idx.get)(
                perm_array_filtered
            )

            assert (
                new_perm_array.shape[1] <= 1000
                and new_perm_array.shape[1] <= perm_array.shape[1]
            )
            res_perm.extend(new_perm_array.tolist())
        return res_perm

    @staticmethod
    def add_ref_info_and_res_perm(atom_array: AtomArray) -> AtomArray:
        """
        Add info of reference structure of atoms to the atom array.

        Args:
            atom_array (AtomArray): The atom array.

        Returns:
            AtomArray: The atom array with the 'ref_pos', 'ref_charge', 'ref_mask', 'res_perm' annotations added.
        """
        ref_pos, ref_charge, ref_mask = AddAtomArrayAnnot.add_ref_feat_info(atom_array)
        res_perm = AddAtomArrayAnnot.add_res_perm(atom_array)

        str_res_perm = []  # encode [N_atom, N_perm] -> list[str]
        for i in res_perm:
            str_res_perm.append("_".join([str(j) for j in i]))

        assert (
            len(atom_array)
            == len(ref_pos)
            == len(ref_charge)
            == len(ref_mask)
            == len(res_perm)
        ), f"{len(atom_array)=}, {len(ref_pos)=}, {len(ref_charge)=}, {len(ref_mask)=}, {len(str_res_perm)=}"


        assert np.array_equal(atom_array.ref_pos, ref_pos)
        assert np.array_equal(atom_array.ref_charge, ref_charge)
        assert np.array_equal(atom_array.ref_mask, ref_mask)

        atom_array.set_annotation("ref_pos", ref_pos)
        atom_array.set_annotation("ref_charge", ref_charge)
        atom_array.set_annotation("ref_mask", ref_mask)
        atom_array.set_annotation("res_perm", str_res_perm)
        return atom_array
