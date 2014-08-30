get_filename_component( dir ${CMAKE_CURRENT_LIST_FILE} PATH )

set( rxn_SRC
  ${dir}/EquilRxnMdl.cpp
  ${dir}/FastChemRxnMdl.cpp
  ${dir}/JCSFlamelets.cpp
  ${dir}/MixtureFraction.cpp
  ${dir}/StreamMixing.cpp
  )
	
add_subdirectory( philCoalTable )

add_library( rxnmdl STATIC ${rxn_SRC} )
