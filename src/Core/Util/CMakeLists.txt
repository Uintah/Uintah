configure_file( testEndian.h.in ${CMAKE_CURRENT_BINARY_DIR}/Endian.h )

include( CheckCXXSourceCompiles )
check_cxx_source_compiles(
        "int main(){ int x; __sync_add_and_fetch(&x,1); return 0; }"
        HAVE_GCCSYNC
    )
if( HAVE_GCCSYNC )
    set( REF_COUNTED "RefCounted_gcc.cc" )
else()
    set( REF_COUNTED "RefCounted.cc" )
endif()

add_library( Core_Util
        DebugStream.cc
        DOUT.cpp
        Endian.cc
        Environment.cc
        FileUtils.cc
        ProgressiveWarning.cc
        RWS.cc
        ${REF_COUNTED}
        SizeTypeConvert.cc
        soloader.cc
        StringUtil.cc
        SysUtils.cc
        Util.cc
        XMLUtils.cc
    )

target_link_libraries( Core_Util
        Core_Exceptions
        Core_Malloc
    )

target_include_directories( Core_Util
    PUBLIC
        ${CMAKE_SOURCE_DIR}
        ${MPI_CXX_INCLUDE_DIRS}
        ${CMAKE_BINARY_DIR}
    )
