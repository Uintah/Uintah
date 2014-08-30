if( TabProps_PREPROCESSOR )

  set( suffix "tbl" )
  set( diffcmd "${CMAKE_BINARY_DIR}/tabprops/util/tablecompare" )

  set( test_dir ${PROJECT_SOURCE_DIR}/test )
  set( test_script ${test_dir}/run_test.cmake )

  set( exe ${CMAKE_BINARY_DIR}/tabprops/prepro/mixrxn )

  file( COPY ${test_dir}/elements.xml DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )

  add_test( test_h2_fastchem
    ${CMAKE_COMMAND}
    -D test_cmd:STRING=${exe}
    -D input_file:STRING=${test_dir}/h2_fastchem.i
    -D cantera_input:STRING=${test_dir}/h2o2.cti
    -D outfile=FastChem.${suffix}
    -D outfile_blessed:STRING=${test_dir}/h2_FastChem_blessed.${suffix}
    -D diff_cmd:STRING=${diffcmd}
    -P ${test_script}
    )

 add_test( test_h2_adiab_eq
   ${CMAKE_COMMAND}
   -D test_cmd:STRING=${exe}
   -D input_file:STRING=${test_dir}/h2_adeq.i
   -D cantera_input:STRING=${test_dir}/h2o2.cti
   -D outfile=AdiabaticEquil.${suffix}
   -D outfile_blessed:STRING=${test_dir}/h2_AdiabaticEquil_blessed.${suffix}
   -D diff_cmd:STRING=${diffcmd}
   -P ${test_script}
   )

 add_test( test_h2_eq
   ${CMAKE_COMMAND}
   -D test_cmd:STRING=${exe}
   -D input_file:STRING=${test_dir}/h2_eq.i
   -D cantera_input:STRING=${test_dir}/h2o2.cti
   -D outfile=Equilibrium.${suffix}
   -D outfile_blessed:STRING=${test_dir}/h2_Equilibrium_blessed.${suffix}
   -D diff_cmd:STRING=${diffcmd}
   -P ${test_script}
   )

 add_test( test_h2_nonreact
   ${CMAKE_COMMAND}
   -D test_cmd:STRING=${exe}
   -D input_file:STRING=${test_dir}/h2_nonreact.i
   -D cantera_input:STRING=${test_dir}/h2o2.cti
   -D outfile=NonReacting.${suffix}
   -D outfile_blessed:STRING=${test_dir}/h2_NonReacting_blessed.${suffix}
   -D diff_cmd:STRING=${diffcmd}
   -P ${test_script}
   )

 add_test( test_h2_eq_clipgauss
   ${CMAKE_COMMAND}
   -D test_cmd:STRING=${exe}
   -D input_file:STRING=${test_dir}/h2_cg.i
   -D cantera_input:STRING=${test_dir}/h2o2.cti
   -D outfile=MixingModel.${suffix}
   -D outfile_blessed:STRING=${test_dir}/h2_MixingModel_blessed.${suffix}
   -D diff_cmd:STRING=${diffcmd}
   -P ${test_script}
   )

endif( TabProps_PREPROCESSOR )