# FEMCorrosionSimulation
The Finite Element method and utilities to simulate the corrosion process in parallel.

To execute this code you need to install different packages and then make the execution file with CMake. Find in the following an overview of list of requirements.
To keep it short usually a link to full description of installation or issue is attached.

## Instalation
You can find related details of all tools and packages. Indeed some of them supplied together and there is no need for installation but naming all is for having the big image of setups resulted to the respective publication. In case of installation keeping the order is recommended to avoid possible errors.

- **Operation System:** Ubuntu 20.04.2 LTS

  To assure all compilers are updated on essential packages are installd, in shell the following command can be run:
  ```shell
  sudo apt-get install build-essential software-properties-common
  ```
- **MPI:** The basic Message Passing Interface (MPI) can be applied through two most common implementations:
   1. OPEN MPI which is the one we used. It is required to build PETSc. Although PETSc would install it itself if it is not installed on the system, installing from shell in ubuntu 20.04.2 LTS by the following command installs `Open MPI 4.0.3`. Installation from source is not recommended.
        ```shell
        sudo apt install openmpi-bin
        ```
        For other linux systems one option here is to enlist all available versions:
        ```shell
        apt list -a openmpi-bin
        ```
        Then install an specific version:
        ```shell
        sudo apt install openmpi-bin=<specific version>(example: =4.0.3-0ubuntu1)
        ```
   2. MPICH havs similar steps for installation
- **PETSc:** In case PETSc is not installed on the system or the environmental variables are not correctly set, FEniCS has a default version of PETSc libraries. Since as a parallel computing library PETSc is optimized and updated continuously, installing the last version instead of relying on FEnics internal PETSc library is strongly recommended. Therefore step one is to install PETSc and step two is to set environmental variables.
   1. To install PETSc the full explanation can be found in [PETSc](https://www.mcs.anl.gov/petsc/documentation/installation.html). In short, three requirements are MPI, BLAS and LAPACK. Similar to MPI the other two can be installed if desired by passing the version. Ubuntu 20.04.2 LTS installs `BLAS 3.9.0` and `LAPACK 3.9.0`.
        ```shell
        sudo apt install libblas-dev
        sudo apt install liblapack-dev
        ```
        Some optional flags can be passed to PETSc while configuring to enhance the performance, among them `ParMETIS 4.0.3` and `METIS 5.1.0` should be installed manually from [ParMETIS](http://glaros.dtc.umn.edu/gkhome/metis/parmetis/download) and [METIS](http://glaros.dtc.umn.edu/gkhome/metis/metis/download). Other libraries can be downloaded through PETSc, then the configuring line of PETSc looks like the following:
        ```shell
        ./configure --with-packages-build-dir=/where/the/extracted/folder/is/petsc-3.13.0/myTemporaryBuild 
        --PETSC_ARCH=PETScForFEniCS --download-hypre=yes --with-metis-include=/usr/local/include/ 
        --with-metis-lib=/usr/local/lib/libmetis.so --with-parmetis-include=/usr/local/include/ 
        --with-parmetis-lib=/usr/local/lib/libparmetis.so --download-mumps=yes --download-scalapack=yes --download-suitesparse=yes
        ```
        (in case of installation on previous build pass also `--with-clean=1`)
   2. Two enviromental variales `PETSC_DIR` and `PETSC_ARCH` should be set. There is a guid for permanently setup based on respective [Linux system](https://unix.stackexchange.com/questions/117467/how-to-permanently-set-environmental-variables). The short setting for Ubuntu 20.04.2 LTS is first:
        ```shell
        sudo gedit /etc/profile
        ```
        and then adding the following lines to the end of file and reboot:
        ```shell
        PETSC_DIR=/where/it/is/installed/petsc-3.13.0 
        PETSC_ARCH=PETScForFEniCS 
        export PETSC_DIR PETSC_ARCH
        ```
- **FEniCS:** There are different ways for installation discussed in [FEniCS](https://fenics.readthedocs.io/en/latest/installation.html#id6). Since any part of code is not in python, the fastest way is to only install `fenics-ffc` component. FFC requires `fenics-fiat`, `fenics-ufl` and `fenics-dijitso` that should be installed automatically, however if did not they can be installed similarly.
  ```shell
  pip3 install fenics-ffc --upgrade
  ```
   The installation is complete by installing Dolfin module. The download and installation manual can be found in [Dolfin](https://bitbucket.org/fenics-project/dolfin/src/master/). For completeness, the following commands should be run in extracted folder after downloading the desired version.
  ```shell
  mkdir build
  cd build
  cmake ..
  make install
  ```
  It is necessary to set dolfin library directory permanently. To do this the steps are as follows:
  ```shell
  sudo gedit /etc/profile
  ```
  and then adding the following lines to the end of file and reboot:
  ```shell
  source /usr/local/share/dolfin/dolfin.conf
  ```
- **SUNDIALS:** It is a package for solving differential/algebraic equations. CVODE module for system of ordinary differential equations is integrated here. The download and installation guide can be found in [SUNDIALS](https://computing.llnl.gov/projects/sundials/sundials-software). After extracting into a folder (e.g. SourceDIR) and creating a folder (e.g. BuildDIR) in parallel to that. The following commands from inside the BuildDIR will install CVODE in default directory.
  ```shell
  cmake ../cvode-5.7.0
  make
  make install
  ```
## Implementation
By successful installation of requirements the current simulation can be run by commands:
```shell
cmake .
make
mpirun -np 1 demo_corrosion
```
Major changes and extensions to tailor this code to a problem are discussed at the following levels/tools:
- **Gmsh:** It can be downloaded from [Gmsh](https://gmsh.info/). By using it any complex geometry can be triangulated and transformed into suitable mesh for finite elements method. A mesh and different marked borders can be saved into several .msh file. Note to choose ASCII 2 option while saving to avoid issues in the next steps.
- **meshio:** 
