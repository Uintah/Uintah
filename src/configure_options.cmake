# General Uintah Build options
#
# NOTE: COMPONENTS SHOULD *NOT* ADD OPTIONS HERE. Rather, they should add options within their own scope to avoid confusion.
#

include( CMakeDependentOption )

# jcs may want to trigger this any time any components are activated?  Don't all need it?
option( ENABLE_HYPRE "Activates Hypre linking - may be required by some components" OFF )

option( ENABLE_KOKKOS "Enable the KOKKOS library - still in development" OFF )
option( ENABLE_SCI_MALLOC "Use SCI memory allocation/deallocation checks." OFF )
cmake_dependent_option( ENABLE_SCINEW_LINE_NUMBERS "Have SCI malloc store the file and line that calls 'scinew'" OFF
        "ENABLE_SCI_MALLOC" ON
    )

# jcs need to figure a different way to do this one
# see https://cmake.org/cmake/help/latest/module/CMakeDependentOption.html#module:CMakeDependentOption
option( ENABLE_MEMORY_INITIALIZATION "Force all new'd and malloc'd memory to be initialized a given value" )
if( ENABLE_MEMORY_INITIALIZATION )
    set( MEMORY_INIT_NUMBER 127 CACHE STRING "Value to initialize memory to" )
endif()

option( STATIC_BUILD "Link Uintah libraries statically" OFF )
if( STATIC_BUILD )
    set( BUILD_SHARED_LIBS OFF )
else()
    set( BUILD_SHARED_LIBS ON )
endif()

set( ASSERTION_LEVEL 0 CACHE STRING "Use assertion level N (0-3) (Where 0 turns off assertions, and 3 turns on all assertions.")
set( THROW_LEVEL 0 CACHE STRING "Use throw level N (0-3)" )
if( CMAKE_BUILD_TYPE EQUAL Release )
    set( ASSERTION_LEVEL 0 )
elseif( CMAKE_BUILD_TYPE EQUAL Debug )
    set( ASSERTION_LEVEL 3 )
endif()
option( ENABLE_UNIT_TESTS "Turn on the unit tests." ON )  # jcs check this
option( ENABLE_MINIMAL "Build a minimal set of libraries" OFF ) # jcs keep this?

# COMPONENT ACTIVATION
# jcs expand this.
option( ENABLE_CUDA     "Enable CUDA Bindings"               OFF )
option( ENABLE_HIP      "Enable HIP Bindings"                OFF )
option( ENABLE_SYCL     "Enable SYCL Bindings"               OFF )
option( ENABLE_NCCL     "Experimental: Enable NCCL Bindings" OFF )
option( ENABLE_RCCL     "Experimental: Enable RCCL Bindings" OFF )
option( ENABLE_OCCL     "Experimental: Enable OCCL Bindings" OFF )
option( ENABLE_EXAMPLES "Enable example component"           OFF )
option( ENABLE_ARCHES   "Enable the ARCHES component"        OFF )
option( ENABLE_ICE      "Enable the ICE component"           OFF )
option( ENABLE_MPM      "Enable the MPM component"           OFF )
option( ENABLE_WASATCH  "Enable the Wasatch component"       OFF )
option( ENABLE_RAY_SCATTER "Enable the ray scattering tools" OFF )

#----------------------------------------------------------
#               Set up GPU dependencies
#----------------------------------------------------------

# Die if two or more device backends are chosen
if( ENABLE_CUDA AND ENABLE_SYCL )
  message( FATAL_ERROR "CUDA / SYCL bindings are mutually exclusive" )
endif()
if( ENABLE_CUDA AND ENABLE_HIP )
  message( FATAL_ERROR "CUDA / HIP bindings are mutually exclusive" )
endif()
if( ENABLE_SYCL AND ENABLE_HIP )
  message( FATAL_ERROR "SYCL / HIP bindings are mutually exclusive" )
endif()

if( ENABLE_CUDA )
  enable_language( CUDA )
  find_dependency( CUDAToolkit REQUIRED )
  if( ENABLE_NCCL )
    find_dependency( NCCL )
  endif()
endif( ENABLE_CUDA )

if( ENABLE_HIP )
  # unsets previous CXX standard and sets CXX to std++17
  unset(CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_HIP_STANDARD 17)

  set(CMAKE_HIP_ARCHITECTURES "gfx90a" CACHE STRING "GPU targets to compile for")

  if(DEFINED ENV{ROCM_PATH})
    set(HIP_PATH "$ENV{ROCM_PATH}" CACHE PATH "Path to which HIP has been installed")
    set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})
  endif()

  #enable_language( HIP )
  find_package(hip)
  if(hip_FOUND)
     message(STATUS "Found HIP: ${HIP_VERSION}")
     message(STATUS "HIP: Runtime=${HIP_RUNTIME} Compiler=${HIP_COMPILER} Path=${HIP_PATH}")
     include_directories(${HIP_PATH}/include)
     add_compile_definitions(__HIP_PLATFORM_AMD__)
  else()
     message(FATAL_ERROR "Could not find HIP."
        " Ensure that HIP is either installed in /opt/rocm/hip or the variable HIP_PATH is set to point to the right location.")
  endif()

  if( ENABLE_RCCL )
    find_dependency( rccl REQUIRED )
  endif( ENABLE_RCCL )
endif( ENABLE_HIP )

if( ENABLE_SYCL )
  # unsets previous CXX standard and sets CXX to std++17
  unset(CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)

  # set SYCL flags for AoT compilation for ATS, PVC devices
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -sycl-std=2020 -fsycl-device-code-split=per_kernel -lze_loader")

  if( ENABLE_OCCL )
    find_dependency( OCCL )
  endif( ENABLE_OCCL )
endif( ENABLE_SYCL )

cmake_dependent_option( ENABLE_MPM_ICE "Enable MPM-ICE" ON
        "ENABLE_MPM; ENABLE_ICE" OFF
    )

# Fortran stuff
if( ENABLE_ARCHES ) # Arches requires fortran
    set( ENABLE_FORTRAN ON )
    include( CheckLanguage )
    check_language( Fortran )
    if( CMAKE_Fortran_COMPILER )
        set( ENABLE_FORTRAN ON )
    else()
        message( SEND_ERROR "No Fortran compiler was found" )
    endif()
    enable_language( Fortran )
    set( CMAKE_Fortran_FORMAT FIXED )
endif()

# jcs to figure out:
# HAVE_VISIT  VISIT_PATH
# HAVE_PIDX
# Hypre configuration...
# HAVE_TIFF
# gperftools

option( ENABLE_GPERFTOOLS "Search for google performance tools and link them if found. Consider setting `GPERFTOOLS_ROOT_DIR`" OFF )

# BUILD_MODELS_RADIATION