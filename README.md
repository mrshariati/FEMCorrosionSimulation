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
    Some optional flags can pass to PETSc while configuring to enhance the performance, among them `ParMETIS 4.0.3` and `METIS 5.1.0` should be installed manually from [ParMETIS](http://glaros.dtc.umn.edu/gkhome/metis/parmetis/download) and [METIS](http://glaros.dtc.umn.edu/gkhome/metis/metis/download). Other libraries can be downloaded through PETSc, then the configuring line of PETSc looks like the following:
    ```shell
    ./configure --with-packages-build-dir=/where/the/extracted/folder/is/petsc-3.13.0/myTemporaryBuild 
    --PETSC_ARCH=PETScForFEniCS --download-hypre=yes --with-metis-include=/usr/local/include/ 
    --with-metis-lib=/usr/local/lib/libmetis.so --with-parmetis-include=/usr/local/include/ 
    --with-parmetis-lib=/usr/local/lib/libparmetis.so --download-mumps=yes --download-scalapack=yes --download-suitesparse=yes
    ```
    (in case of installation on previous build pass also `--with-clean=1`)
    2. By setting two enviromental variales `PETSC_DIR` and `PETSC_ARCH` permanently in respective [Linux system](https://unix.stackexchange.com/questions/117467/how-to-permanently-set-environmental-variables). The short setting for Ubuntu 20.04.2 LTS is:
    ```shell
    sudo gedit /etc/profile
    ```
    Add the following lines to the end of file and reboot:
    ```shell
    PETSC_DIR=/where/it/is/installed/petsc-3.13.0 
    PETSC_ARCH=PETScForFEniCS 
    export PETSC_DIR PETSC_ARCH
    ```
```mermaid
%% Example of sequence diagram
  sequenceDiagram
    Alice->>Bob: Hello Bob, how are you?
    alt is sick
    Bob->>Alice: Not so good :(
    else is well
    Bob->>Alice: Feeling fresh like a daisy
    end
    opt Extra response
    Bob->>Alice: Thanks for asking
    end
​```
