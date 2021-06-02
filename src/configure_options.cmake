# General Uintah Build options
#
# NOTE: COMPONENTS SHOULD *NOT* ADD OPTIONS HERE. Rather, they should add options within their own scope to avoid confusion.
#

include( CMakeDependentOption )

# jcs may want to trigger this any time any components are activated?  Don't all need it?
option( ENABLE_HYPRE "Activates Hypre linking - may be required by some components" OFF )

option( ENABLE_KOKKOS "Enable the KOKKOS library - still in development" OFF )
option( ENABLE_SCI_MALLOC "Use SCI memory allocation/deallocation checks." OFF )

# jcs need to figure a different way to do this one
# see https://cmake.org/cmake/help/latest/module/CMakeDependentOption.html#module:CMakeDependentOption
option( ENABLE_MEMORY_INITIALIZATION "Force all new'd and malloc'd memory to be initialized a given value" )
if( ENABLE_MEMORY_INITIALIZATION )
    set( MEMORY_INIT_NUMBER 127 CACHE STRING "Value to initialize memory to" )
endif()

option( STATIC_BUILD "Link Uintah libraries statically" OFF )
if( STATIC_BUILD )
    add_compile_definitions( STATIC_BUILD )
    set( BUILD_SHARED_LIBS OFF )
endif()

option( ENABLE_SCINEW_LINE_NUMBERS "Have SCI malloc store the file and line that calls 'scinew'" OFF )

set( ASSERTION_LEVEL 0 CACHE STRING "Use assertion level N (0-3) (Where 0 turns off assertions, and 3 turns on all assertions.")
set( THROW_LEVEL 0 CACHE STRING "Use throw level N (0-3)" )

option( ENABLE_UNIT_TESTS "Turn on the unit tests." ON )  # jcs check this
option( ENABLE_MINIMAL "Build a minimal set of libraries" OFF ) # jcs keep this?

# COMPONENT ACTIVATION
# jcs expand this.
option( ENABLE_EXAMPLES "Enable example component" OFF )
option( ENABLE_ARCHES "Enable the ARCHES component" OFF )
option( ENABLE_ICE "Enable the ICE component" OFF )
option( ENABLE_MPM "Enable the MPM component" OFF )
option( ENABLE_WASATCH "Enable the Wasatch component" OFF )

option( ENABLE_RAY_SCATTER "Enable the ray scattering tools" OFF )

cmake_dependent_option( ENABLE_MPM_ICE "Enable MPM-ICE" OFF
        "ENABLE_MPM;ENABLE_ICE" ON
    )

# jcs to figure out:
# HAVE_VISIT  VISIT_PATH
# HAVE_PIDX
# EXCEPTIONS_CRASH_DEF  -- see error_testdefs.h.in
# Hypre configuration...
# HAVE_TIFF
# gperftools

unset( HAVE_GPERFTOOLS )

# BUILD_MODELS_RADIATION