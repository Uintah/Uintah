
macro( nebo_cuda_prep_dir )
  if( ENABLE_CUDA )
    include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )
  endif( ENABLE_CUDA )
endmacro( nebo_cuda_prep_dir )

macro( nebo_cuda_prep_file file )
  add_custom_command( OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${file}.cu
                      COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file} ${CMAKE_CURRENT_BINARY_DIR}
                      COMMAND ${CMAKE_COMMAND} -E rename ${CMAKE_CURRENT_BINARY_DIR}/${file} ${CMAKE_CURRENT_BINARY_DIR}/${file}.cu
                      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${file}
                     )
  set_source_files_properties( ${CMAKE_CURRENT_BINARY_DIR}/${file}.cu
                               PROPERTIES GENERATED TRUE
                              )
endmacro( nebo_cuda_prep_file )

macro( nebo_add_executable exe_name )
  set( files ${ARGN} )
  foreach( file ${files} )
    if( ENABLE_CUDA )
      nebo_cuda_prep_file( ${file} )
      set( cufiles ${cufiles} ${CMAKE_CURRENT_BINARY_DIR}/${file}.cu )
    endif( ENABLE_CUDA )
  endforeach()
  if( ENABLE_CUDA )
    cuda_add_executable( ${exe_name} ${cufiles} )
    unset( cufiles )
  else( ENABLE_CUDA )
    add_executable( ${exe_name} ${files} )
  endif( ENABLE_CUDA )
endmacro( nebo_add_executable )

macro( nebo_set_add_file var file )
  if( ENABLE_CUDA )
    nebo_cuda_prep_file( ${file} )
    set( ${var} ${${var}} ${CMAKE_CURRENT_BINARY_DIR}/${file}.cu )
  else( ENABLE_CUDA )
    set( ${var} ${${var}} ${file} )
  endif( ENABLE_CUDA )
endmacro( nebo_set_add_file )

macro( nebo_add_library lib_name )
  set( files ${ARGN} )
  if( ENABLE_CUDA )
    cuda_add_library( ${lib_name} STATIC ${files} )
  else( ENABLE_CUDA )
    add_library( ${lib_name} STATIC ${files} )
  endif( ENABLE_CUDA )
endmacro( nebo_add_library )
