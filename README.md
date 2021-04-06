# FEMCorrosionSimulation
Provides both source code for the Finite Element method and prerequisites to efficiently simulate the corrosion process using parallelization.

In order to compile and run this code you need to install different packages and then make the execution file with CMake. In the following an overview of list of prerequisites is given.
To keep it short usually a link to full description of installation or issue is provided.

## Instalation
In the following details of all tools and packages are given. Indeed some of them are supplied together and thus there is no need for separate installation. However naming all is for having the big image of setups related to the underlying publication, see the licensing section at the end of this documentation. In case of installations, keeping the order is recommended to avoid possible errors.

- **Operating System:** Ubuntu 20.04.2 LTS

  To assure that all compilers are updated and essential packages are installed, in shell the following command can be run:
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
        Then install a specific version:
        ```shell
        sudo apt install openmpi-bin=<specific version>(example: =4.0.3-0ubuntu1)
        ```
   2. MPICH has similar steps for installation
- **PETSc:** In case PETSc is not installed on the system or the environmental variables are not correctly set, FEniCS has a default version of PETSc libraries. Since as a parallel computing library PETSc is optimized and updated continuously, installing the latest version instead of relying on FEnics internal PETSc library is strongly recommended. Therefore first step is to install PETSc and second is to set environmental variables.
   1. To install PETSc the full explanation can be found in [PETSc](https://www.mcs.anl.gov/petsc/documentation/installation.html). In short, three requirements are MPI, BLAS and LAPACK. Similar to MPI the other two can be installed if desired by passing the version. Ubuntu 20.04.2 LTS installs `BLAS 3.9.0` and `LAPACK 3.9.0`.
        ```shell
        sudo apt install libblas-dev
        sudo apt install liblapack-dev
        ```
        Some optional flags can be passed to PETSc while configuring to enhance the performance, among them `ParMETIS 4.0.3` and `METIS 5.1.0` should be installed manually from [ParMETIS](http://glaros.dtc.umn.edu/gkhome/metis/parmetis/download) and [METIS](http://glaros.dtc.umn.edu/gkhome/metis/metis/download). Other libraries can be downloaded through PETSc, then the configuring line of PETSc in terminal looks like the following:
        ```shell
        ./configure --with-packages-build-dir=/where/the/extracted/folder/is/petsc-3.13.0/myTemporaryBuild 
        --PETSC_ARCH=PETScForFEniCS --download-hypre=yes --with-metis-include=/usr/local/include/ 
        --with-metis-lib=/usr/local/lib/libmetis.so --with-parmetis-include=/usr/local/include/ 
        --with-parmetis-lib=/usr/local/lib/libparmetis.so --download-mumps=yes --download-scalapack=yes --download-suitesparse=yes
        ```
        (in case of installation on previous build pass also `--with-clean=1`)
   2. Two environmental variables `PETSC_DIR` and `PETSC_ARCH` should be set. There is a guide for permanently setup based on respective [Linux system](https://unix.stackexchange.com/questions/117467/how-to-permanently-set-environmental-variables). The short setting for Ubuntu 20.04.2 LTS is first:
        ```shell
        sudo gedit /etc/profile
        ```
        and then adding the following lines to the end of file and reboot:
        ```shell
        PETSC_DIR=/where/it/is/installed/petsc-3.13.0 
        PETSC_ARCH=PETScForFEniCS 
        export PETSC_DIR PETSC_ARCH
        ```
- **FEniCS:** There are different ways for installation discussed in [FEniCS](https://fenics.readthedocs.io/en/latest/installation.html#id6). Since any part of code is not in python, the fastest way is to only install the `fenics-ffc` component. FFC requires `fenics-fiat`, `fenics-ufl` and `fenics-dijitso` that should be installed automatically. However if it did not they can be installed similarly.
  ```shell
  pip3 install fenics-ffc --upgrade
  ```
   The installation is complete by installing the Dolfin module. The download and installation manual can be found in [Dolfin](https://bitbucket.org/fenics-project/dolfin/src/master/). For completeness, the following commands should be run in the extracted folder after downloading the desired version.
  ```shell
  mkdir build
  cd build
  cmake ..
  make install
  ```
  It is necessary to set the dolfin library directory permanently. To do this the steps are as follows:
  ```shell
  sudo gedit /etc/profile
  ```
  and then adding the following lines to the end of file and reboot:
  ```shell
  source /usr/local/share/dolfin/dolfin.conf
  ```
- **SUNDIALS:** It is a package for solving differential/algebraic equations. CVODE module for system of ordinary differential equations is integrated here. The download and installation guide can be found in [SUNDIALS](https://computing.llnl.gov/projects/sundials/sundials-software). After extracting into a directory (e.g. `~/SourceDIR`) and creating a directory (e.g. `~/BuildDIR`) in parallel to that. The following commands from inside the BuildDIR will install CVODE in the default directory (usually `/usr/local`).
  ```shell
  cmake ../cvode-5.7.0
  make
  make install
  ```
## Implementation
After successful installation of prerequisites the current simulation can be run by commands:
```shell
	@@ -86,27 +86,27 @@ mpirun -np 1 demo_corrosion
Major changes and extensions to tailor this code to a specific problem are discussed in the following levels/tools:
- **Gmsh:** It can be downloaded from [Gmsh](https://gmsh.info/). By using it any complex geometry can be triangulated and transformed into a suitable mesh for the finite element method. A mesh and different marked borders can be saved into several (.msh) files. Note to choose ASCII 2 option while saving to avoid issues in the next steps.
- **meshio:** The (.msh) files from Gmsh can not be imported directly by FEniCS, however (.xml) file format is an efficient choice and can be imported in parallel by FEniCS. For installation as python component read [meshio](https://pypi.org/project/meshio/) or run the command:
  ```shell
  pip3 install meshio
  ```
  after successful installation the following command will transform `mesh.msh` to `mesh.xml` suitable for FEniCS:
  ```shell
  meshio-convert mesh.msh mesh.xml -z
  ```
  passing `-z` flag is critical to eliminate third dimension that creates ambiguity for FEniCS. More information on flags is available by:
  ```shell
  meshio-convert -h
  ```
- **FEniCS:** In the current code only `/mesh/mesh.xml` is imported and since the geometry of mesh is a rectangle, boundaries are defined as straight lines. A good starting point on how to import boundaries as (.xml) file is explained in [FEniCS Q&A](https://fenicsproject.org/qa/2986/how-to-define-boundary-condition-for-mesh-generated-by-gmsh/) and [FEniCS Q&A](https://fenicsproject.org/qa/5337/importing-marked-mesh-for-parallel-use/) as well marking with Gmsh in [meshio GitHub](https://github.com/nschloe/meshio/issues/265) and migration from (.xml) to (.xdmf) in [FEniCS Q&A](https://fenicsproject.discourse.group/t/transitioning-from-mesh-xml-to-mesh-xdmf-from-dolfin-convert-to-meshio/412/49).
- **PETSc:** It is integrated by FEniCS or more precisely Dolfin. In class index page of [Dolfin](https://fenicsproject.org/olddocs/dolfin/2019.1.0/cpp/classes.html), the access to native pointers of PETSc via class members is explained. Good examples are `/src/FEMTools.cpp` and `main.cpp` of this directory; the finite element method is implemented in native PETSc environment while the input arguments are originally Dolfin objects. Other interesting aspect is the maneuver on mesh regions `Dolfin::Function` nodes and respective `Dolfin::PETScVector` objects, which can be found in `main.cpp` and `/src/GeneralTools.cpp`. Form headers such as `/src/Poisson.h` should be regenerated from `/ufl/Poisson.ufl` and replaced for every system. The command for generating `.h` heasers is:
  ```shell
  ffc -l dolfin Poisson.ufl
  ```
- **SUNDIALS:** Embedded in `/src/CorrosionTools.cpp` minor changes on the system `y'=f(y,t)` and Jacobian of `f(y,t)` can cover a wide range of problems.
- **ParaView:** The results are stored via Dolfin as (.pvd) files. The version `ParaView-4.4.0-Qt4-OpenGL2`, which can be downloaded from [ParaView](https://www.paraview.org/download/), is compatible for visualization.
- **CMake:** The use of widely applied CMake for installation of Dolfin, PETSc and SUNDIALS provides an efficient tool for binding such open-source libraries in the project. It can be downloaded and installed from [CMake](https://cmake.org/). The file `CMakeLists.txt` in this directory is an early example of how to bind different libraries and tools into a single project.
## Licensing
For Licenses of all libraries and tools please read their respective homepages. This directory with original developed codes is available for application, modification, and redistribution by citing as:
- **Citation:**

  This code is supplied with the submitted paper:
  Shariati, M., Weber, E.W., Höche, D., 2021. Parallel simulation of the POISSON-NERNST-PLANCK corrosion model with an algebraic flux correction scheme.
