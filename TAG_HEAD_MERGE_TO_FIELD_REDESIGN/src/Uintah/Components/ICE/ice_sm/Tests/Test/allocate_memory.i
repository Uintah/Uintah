/*______________________________________________________________________
*   This section contains the memory allocations
*_______________________________________________________________________*/
    uvel_CC_old     = darray_3d(0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    vvel_CC_old     = darray_3d(0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    wvel_CC_old     = darray_3d(0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    vel_difference  = darray_3d(0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);



    x_CC            = darray_3d(0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    y_CC            = darray_3d(0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    z_CC            = darray_3d(0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    Vol_CC          = darray_3d(0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
                        
    uvel_CC         = darray_4d(1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    vvel_CC         = darray_4d(1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    wvel_CC         = darray_4d(1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    rho_CC          = darray_4d(1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    press_CC        = darray_4d(1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    delPress_CC     = darray_4d(1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);

    Temp_CC         = darray_4d(1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    viscosity_CC    = darray_4d(1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    thermalCond_CC  = darray_4d(1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    cv_CC           = darray_4d(1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    R               = dvector_nr(1, N_MATERIAL);
    gamma           = dvector_nr(1, N_MATERIAL);

    mass_CC         = darray_4d(1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    xmom_CC         = darray_4d(1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    ymom_CC         = darray_4d(1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    zmom_CC         = darray_4d(1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    int_eng_CC      = darray_4d(1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    total_eng_CC    = darray_4d(1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    div_velFC_CC    = darray_4d(1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);



    scalar1_CC      = darray_4d(1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    scalar2_CC      = darray_4d(1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    scalar3_CC      = darray_4d(1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    grav            = dvector_nr(1, 3);
    speedSound      = darray_4d(1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
       

                                        /* Lagrangian variables             */
    xmom_L_CC       = darray_4d(1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    ymom_L_CC       = darray_4d(1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    zmom_L_CC       = darray_4d(1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);    
    int_eng_L_CC    = darray_4d(1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);    

    press_L_CC      = darray_4d(1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    mass_L_CC       = darray_4d(1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    rho_L_CC        = darray_4d(1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    Temp_L_CC       = darray_4d(1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    Vol_L_CC        = darray_4d(1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
                                        /* source terms                     */
    mass_source     = darray_4d(1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    xmom_source     = darray_4d(1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    ymom_source     = darray_4d(1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    zmom_source     = darray_4d(1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    int_eng_source  = darray_4d(1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);


                                        /*  BOUNDARY CONDITIONS              */
    BC_inputs       = imatrix(      1, N_DOMAIN_WALLS,  1, N_MATERIAL);
    BC_types        = iarray_3d(    1, N_DOMAIN_WALLS,  1, N_VARIABLE_BC,   1, N_MATERIAL );
    BC_Values       = darray_3d(    1, N_DOMAIN_WALLS,  1, N_VARIABLE_BC,   1, N_MATERIAL ); 
    BC_float_or_fixed=iarray_3d(    1, N_DOMAIN_WALLS,  1, N_VARIABLE_BC,   1, N_MATERIAL ); 
                                        /*  OUTPUT TIMER                    */
    t_output_vars   = dvector_nr(  1,  3);
    delt_limits     = dvector_nr(  1,  3);    
                                        /* Face-centered variables           */
    x_FC            = darray_5d(0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM, 1, N_CELL_FACES, 1,  1);
    y_FC            = darray_5d(0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM, 1, N_CELL_FACES, 1,  1);
    z_FC            = darray_5d(0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM, 1, N_CELL_FACES, 1,  1);    
    uvel_FC         = darray_6d(0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM, 1, N_CELL_FACES, 1, N_MATERIAL, 1,1);
    vvel_FC         = darray_6d(0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM, 1, N_CELL_FACES, 1, N_MATERIAL, 1,1);    
    wvel_FC         = darray_6d(0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM, 1, N_CELL_FACES, 1, N_MATERIAL, 1,1);
    press_FC        = darray_6d(0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM, 1, N_CELL_FACES, 1, N_MATERIAL, 1,1);        
    tau_X_FC        = darray_6d(0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM, 1, N_CELL_FACES, 1, N_MATERIAL, 1,1);
    tau_Y_FC        = darray_6d(0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM, 1, N_CELL_FACES, 1, N_MATERIAL, 1,1);
    tau_Z_FC        = darray_6d(0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM, 1, N_CELL_FACES, 1, N_MATERIAL, 1,1);
