import itertools
from copy import deepcopy
from math import sqrt

import numpy as np
from rdkit import Chem


class PiSystem:
    """
    Class to find the pi systems in a conjugated molecule and their size

    args:
        mol: the molecule

    attributes:
        mol: RDKit molecule
        pi_systems: list of pi systems in the molecule
        pi_systems_size: size of the largest pi system
        largest_pi_system: largest pi system in the molecule
        smallest_pi_system: largest pi system in the molecule
        mol_coords: coordinates of all atoms in the molecule
        pi_system_coords: coordinates of all atoms in the largest pi system
        pi_system_distances: distances between all atoms in the largest pi system
        max_distance: maximum distance between two atoms in the largest pi system
        ring_systems: list of ring systems in the molecule
        area: area of the largest pi system in the molecule
        areas: areas of all pi systems in the molecule
    """

    def __init__(self, mol: Chem.Mol):
        self.mol = self._clean_mol(mol)
        self.pi_systems = self.get_pi_systems()
        self.pi_systems_size = self.get_pi_system_size()
        self.largest_pi_system = self.get_largest_pi_system()
        self.smallest_pi_system = self.get_smallest_pi_system()
        self.mol_coords = self.get_mol_coords()
        self.pi_system_coords = self.get_pi_system_coords()
        self.pi_system_distances = self.get_pi_system_distances()
        self.max_distance = self.get_max_distance()
        self.ring_systems = self.get_ring_systems()
        self.area = self.get_largest_pi_area()
        self.areas = self.get_pi_areas()

    def _clean_mol(self, mol) -> Chem.Mol:
        """Remove hydrogens and kekulize the molecule"""
        mol_copy = deepcopy(mol)
        mol_copy = Chem.RemoveHs(mol_copy)
        Chem.Kekulize(mol_copy)
        return mol_copy

    def get_pi_systems(self) -> list[list[int]]:
        """Get the largest pi system in the molecule"""
        pi_systems = [
            self._find_pi_system(self.mol, x.GetIdx(), [x.GetIdx()])
            for x in self.mol.GetAtoms()
        ]  # find all pi systems
        pi_systems = [sorted(x) for x in pi_systems]  # sort the atoms in each pi system
        pi_systems = [x for x in pi_systems if len(x) > 1]  # remove single atoms
        pi_systems = list(
            s for s, _ in itertools.groupby(pi_systems)
        )  # remove duplicates
        return pi_systems

    def _find_pi_system(self, mol, current, seen):
        """Find the pi system of a given atom"""
        atom = mol.GetAtomWithIdx(current)
        for neighbor in atom.GetNeighbors():
            bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
            if (neighbor.GetIdx() not in seen) and (
                bond.GetIsConjugated() or bond.GetBondTypeAsDouble() > 1
            ):
                seen.append(neighbor.GetIdx())
                self._find_pi_system(mol, neighbor.GetIdx(), seen)
        return seen

    def get_largest_pi_system(self) -> list[int]:
        """Get the largest pi system in the molecule"""
        return max(self.pi_systems, key=len)

    def get_smallest_pi_system(self) -> list[int]:
        """Get the smallest pi system in the molecule"""
        return min(self.pi_systems, key=len)

    def get_pi_system_size(self) -> int:
        """Get the size of the largest pi system in the molecule"""
        return len(self.get_largest_pi_system())

    def get_mol_coords(self) -> dict:
        """Get the coordinates of the atoms in the largest pi system"""
        atom_list = []
        xyz_coords = {}
        for i, atom in enumerate(self.mol.GetAtoms()):
            positions = self.mol.GetConformer().GetAtomPosition(i)
            atom_list.append([positions.x, positions.y, positions.z])
            xyz_coords[i] = {
                "AtomNum": i,
                "AtomicNum": atom.GetAtomicNum(),
                "x": positions.x,
                "y": positions.y,
                "z": positions.z,
            }

        return xyz_coords

    def get_pi_system_coords(self) -> list[dict]:
        """Get the coordinates of the atoms in the largest pi system"""
        xyz_coords = self.get_mol_coords()
        system_coords = [xyz_coords[x] for x in self.largest_pi_system]
        return system_coords

    def get_pi_system_distances(self) -> list[list[float]]:
        """Get the distances between all atoms in the largest pi system"""
        system_coords = self.get_pi_system_coords()
        distances = np.zeros((len(system_coords), len(system_coords)))
        for i, atom1 in enumerate(system_coords):
            for j, atom2 in enumerate(system_coords):
                distances[i][j] = self._get_distance(atom1, atom2)
        return distances

    def _get_distance(self, atom1: dict, atom2: dict) -> float:
        """Get the distance between two atoms"""
        x_distance = float(atom1["x"]) - float(atom2["x"])
        y_distance = float(atom1["y"]) - float(atom2["y"])
        z_distance = float(atom1["z"]) - float(atom2["z"])
        distance = sqrt(x_distance**2 + y_distance**2 + z_distance**2)
        return distance

    def get_max_distance(self) -> float:
        """Get the maximum distance between two atoms in the largest pi system"""
        distances = self.get_pi_system_distances()
        return np.amax(distances)

    def get_ring_systems(self, include_spiro=False) -> list[list[int]]:
        """Get the ring systems in the molecule"""
        ring_info = self.mol.GetRingInfo()
        systems = []
        for ring in ring_info.AtomRings():
            ring_atoms = set(ring)
            n_systems = []
            for system in systems:
                n_in_common = len(ring_atoms.intersection(system))
                if n_in_common and include_spiro:
                    ring_atoms = ring_atoms.union(system)
                else:
                    n_systems.append(system)
            n_systems.append(sorted(list(ring_atoms)))
            systems = n_systems
        return systems

    def _normal(self, triangles) -> np.ndarray:
        # The cross product of two sides is a normal vector
        return np.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0], axis=1
        )

    def _area(self, triangles) -> np.ndarray:
        # The norm of the cross product of two sides is twice the area
        return np.linalg.norm(self._normal(triangles), axis=1) / 2

    def _get_points(self, coords_dict: dict) -> list[list[float]]:
        """Get the coordinates of the atoms"""
        points = []
        for atom in coords_dict.values():
            points.append([atom["x"], atom["y"], atom["z"]])
        return points

    def get_system_area(self, system) -> float:
        """Get the area of a pi system"""
        points = self._get_points(self.get_mol_coords())
        system_area = 0
        for ring in self.ring_systems:
            in_pi_system = all(elem in system for elem in ring)
            if in_pi_system:
                triangles = []
                for i in range(len(ring) - 2):
                    triangles.append(
                        [points[ring[0]], points[ring[i + 1]], points[ring[i + 2]]]
                    )
                triangles = np.array(triangles)
                ring_area = self._area(triangles).sum()
                system_area += ring_area
        return system_area

    def get_largest_pi_area(self) -> float:
        """Get the area of the largest pi system"""
        return self.get_system_area(self.largest_pi_system)

    def get_pi_areas(self) -> list[float]:
        """Get the area of all pi systems"""
        areas = []
        for system in self.pi_systems:
            areas.append(self.get_system_area(system))
        return areas
