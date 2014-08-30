get_filename_component( dir ${CMAKE_CURRENT_LIST_FILE} PATH )

set( mixmdl_SRC
  ${dir}/BetaMixMdl.cpp
  ${dir}/ClippedGaussMixMdl.cpp
  ${dir}/GaussKronrod.cpp
  ${dir}/Integrator.cpp
  ${dir}/MixMdlHelper.cpp
  )

add_library( mixmdl STATIC ${mixmdl_SRC} )