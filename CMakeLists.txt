# Require CMake
cmake_minimum_required(VERSION 3.16)
# Set CMake behavior
cmake_policy(SET CMP0004 NEW)

set(PROJECT_NAME demo_corrosion)
project(${PROJECT_NAME})

# Get DOLFIN configuration data (DOLFINConfig.cmake must be in DOLFIN_CMAKE_CONFIG_PATH)
find_package(DOLFIN REQUIRED)
include(${DOLFIN_USE_FILE})

#find_package(MPI REQUIRED)
include_directories(${MPI_CXX_INCLUDE_DIRS})

# Specify the path to SUNDIALS libraries (all enviromental variables such as SUNDIALS_SOLVER_LIB)- to refer to SUNDIALS examples at /usr/local/examples
#By default it installed in this directory
set(SUNDIALS_LIBRARY_DIR
  /usr/local/lib
  CACHE PATH "Location of SUNDIALS libraries")
# Find the SUNDIALS libraries
find_library(SUNDIALS_SOLVER_LIB
  sundials_cvode ${SUNDIALS_LIBRARY_DIR}
  DOC "CVODE library")

find_library(SUNDIALS_NVEC_LIB
  sundials_nvecserial ${SUNDIALS_LIBRARY_DIR}
  DOC "NVECTOR_SERIAL library")

find_library(SUNDIALS_MANYVEC_LIB
  sundials_nvecmanyvector ${SUNDIALS_LIBRARY_DIR}
  DOC "NVECTOR_MANYVECTOR library")

# List of SUNDIALS libraries
set(SUNDIALS_LIBRARIES
  -L${SUNDIALS_LIBRARY_DIR}
  ${SUNDIALS_SOLVER_LIB}
  ${SUNDIALS_NVEC_LIB}
  ${SUNDIALS_MANYVEC_LIB})

# Default build type (can be overridden by user)
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug MinSizeRel Release RelWithDebInfo." FORCE)
endif()

# Executable
add_executable(${PROJECT_NAME} main.cpp)

# Target libraries
target_link_libraries(${PROJECT_NAME} dolfin)
target_link_libraries(${PROJECT_NAME} ${MPI_CXX_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${SUNDIALS_LIBRARIES})

# Use mpi compilers
if(MPI_CXX_FOUND)
  target_compile_options(${PROJECT_NAME} PUBLIC ${MPI_CXX_COMPILER_FLAGS})
  set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
endif()
