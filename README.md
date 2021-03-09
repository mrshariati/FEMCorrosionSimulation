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
    2. MPICH
