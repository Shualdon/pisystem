# pisystem

A Python class with functions for describing &pi;-systems of conjugated molecules.


## Features

* `pi_systems`: Returns a list of lists with atom indecid of all seperate conjugated systems.
* `largest_pi_system`: Returns the list of atom indecis of the largest conjugated system.
* `smallest_pi_system`: Returns the list of atom indecis of the smallest conjugated system.
* `pi_systems_size`: Returns the number of atom in the largest conjugates system.
* `mol_coords`: Returns the atom coordinations for all atoms.
* `pi_system_coords`: Returns the atom coordinations for the atoms in the largest conjugated system.
* `max_distance`: Returns the longest distance between two atoms in the largest conjugates system.
* `area`: Returns the area of the largest conjugates system.
* `areas`: Returns a list of areas of all seperate conjugtaes systems.

see `pi_system_test.ipynb` notebook for examples.

### Notes
* In order to have better results, load a `.mol` file with an experimental or a QM-calculated geometry.  
* The area of the conjugates &pi;-system is calculated as a sum of ring areas in the system. Parts of the system that are not part of a ring (e.g. a vinyl group) will not count towards the area.
