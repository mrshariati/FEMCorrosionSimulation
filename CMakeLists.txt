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

# Use mpi compilers
if(MPI_CXX_FOUND)
  target_compile_options(${PROJECT_NAME} PUBLIC ${MPI_CXX_COMPILER_FLAGS})
  set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
endif()
