/* -----------------------------------------------------------------------  */
/*  Free the memory                                                         */
/* -----------------------------------------------------------------------  */
   free_darray_3d( x_CC,            0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
   free_darray_3d( y_CC,            0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
   free_darray_3d( z_CC,            0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
   free_darray_3d( Vol_CC,          0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);

   free_darray_4d( uvel_CC,         1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
   free_darray_4d( vvel_CC,         1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
   free_darray_4d( wvel_CC,         1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
   free_darray_4d( rho_CC,          1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
   free_darray_4d( press_CC,        1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
   free_darray_4d( delPress_CC,     1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);

   free_darray_4d( Temp_CC,         1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
   free_darray_4d( viscosity_CC,    1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
   free_darray_4d( thermalCond_CC,  1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
   free_darray_4d( cv_CC,           1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
   free_dvector_nr( R,                  1, N_MATERIAL);
   free_dvector_nr( gamma,                  1, N_MATERIAL);
   free_darray_4d( mass_CC,         1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
   free_darray_4d( xmom_CC,         1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
   free_darray_4d( ymom_CC,         1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
   free_darray_4d( zmom_CC,         1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
   free_darray_4d( int_eng_CC,      1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
   free_darray_4d( total_eng_CC,    1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
   free_darray_4d( div_velFC_CC,    1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);

 
   free_darray_4d( scalar1_CC,      1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
   free_darray_4d( scalar2_CC,      1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
   free_darray_4d( scalar3_CC,      1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
   free_dvector_nr( grav,              1, 3);
   free_darray_4d( speedSound,      1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);

                                        /* Lagrangian variables             */           
   free_darray_4d( xmom_L_CC,       1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
   free_darray_4d( ymom_L_CC,       1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
   free_darray_4d( zmom_L_CC,       1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
   free_darray_4d( int_eng_L_CC,    1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);

   
   free_darray_4d( press_L_CC,      1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
   free_darray_4d( mass_L_CC,       1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
   free_darray_4d( rho_L_CC,        1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
   free_darray_4d( Temp_L_CC,       1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
   free_darray_4d( Vol_L_CC,        1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
                                        /* source/sink terms                 */
   free_darray_4d( mass_source,     1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
   free_darray_4d( xmom_source,     1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
   free_darray_4d( ymom_source,     1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
   free_darray_4d( zmom_source,     1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
   free_darray_4d( int_eng_source,  1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);


                                        /*  BOUNDARY CONDITIONS              */
   free_imatrix(BC_inputs,          1, N_DOMAIN_WALLS,          1, N_MATERIAL); 

   free_iarray_3d(BC_types,         1, N_DOMAIN_WALLS,          1, N_VARIABLE_BC,   1, N_MATERIAL ); 
   free_darray_3d(BC_Values,        1, N_DOMAIN_WALLS,          1, N_VARIABLE_BC,   1, N_MATERIAL );
   free_iarray_3d(BC_float_or_fixed,1, N_DOMAIN_WALLS,          1, N_VARIABLE_BC,   1, N_MATERIAL );
                                        /*  OUTPUT TIMER                    */
   free_dvector_nr(t_output_vars,       1, 3);
   free_dvector_nr(delt_limits,         1, 3);   
                        
                                        /* Face-centered variables           */
   free_darray_5d( x_FC,            0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM, 1, 2,  1, 1);
   free_darray_5d( y_FC,            0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM, 1, 2,  1, 1);
   free_darray_5d( z_FC,            0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM, 1, 2,  1, 1);   
   free_darray_6d( uvel_FC,         0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM, 1, N_CELL_FACES, 1, N_MATERIAL, 1,1);
   free_darray_6d( vvel_FC,         0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM, 1, N_CELL_FACES, 1, N_MATERIAL, 1,1);
   free_darray_6d( wvel_FC,         0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM, 1, N_CELL_FACES, 1, N_MATERIAL, 1,1);
   free_darray_6d( press_FC,        0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM, 1, N_CELL_FACES, 1, N_MATERIAL, 1,1);
   free_darray_6d( tau_X_FC,        0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM, 1, N_CELL_FACES, 1, N_MATERIAL, 1,1);
   free_darray_6d( tau_Y_FC,        0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM, 1, N_CELL_FACES, 1, N_MATERIAL, 1,1);
   free_darray_6d( tau_Z_FC,        0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM, 1, N_CELL_FACES, 1, N_MATERIAL, 1,1);
