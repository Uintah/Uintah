#undef switch_explicit_implicit                                                                                
#undef switchOveride_Initial_Conditions                                                                                                                              
#undef switch_Compute_burgers_eq                       
#undef switch_step1_OnOff                              
#undef switch_step2_OnOff                              
#undef switch_step3_OnOff                              
#undef switch_step4_OnOff                              
#undef switch_step4_stress_source_OnOff                
#undef switch_step5_OnOff                              
#undef switch_step6_OnOff                                                                                      
#undef switch_step7_OnOff                              
#undef tecplot                                         
#undef switchDebug_main                                                                                        
#undef switchDebug_main_input                          
#undef switchDebug_find_delta_time                     
#undef switchDebug_readInputFile                       
#undef switchDebug_initializeVars                      
#undef switchDebug_grid                                
#undef switchDebug_calc_flux_or_primitive_vars         
#undef switchDebug_equation_of_state                   
#undef switchDebug_pcgmg_test                          
#undef switchDebug_pressure_PCG                        
#undef switchDebug_press_eq_residual                                                                          
#undef switchDebug_compute_face_centered_velocities    
#undef switchDebug_p_face                              
#undef switchDebug_accumulate_momentum_source_sinks    
#undef switchDebug_accumulate_energy_source_sinks      
#undef switchDebug_shear_stress_Xdir                   
#undef switchDebug_shear_stress_Ydir                   
#undef switchDebug_lagrangian_values                   
#undef switchDebug_lagrangian_vol                      
#undef switchDebug_q_exact_solution                    
#undef switchDebug_find_q_vertex_max                   
#undef switchDebug_Advect_q_out_flux                   
#undef switchDebug_Advect_q_in_flux                    
#undef switchDebug_Advect_influx_outflux_volume        
#undef switchDebug_advect_preprocess                   
#undef switchDebug_Advect_gradient_limiter             
#undef switchDebug_Advect_q                            
#undef switchDebug_advect_and_advance_in_time          
#undef switchDebug_update_CC_physical_boundary_conditions      
#undef switchDebug_update_CC_FC_physical_boundary_conditions   
#undef switchDebug_set_Dirichlet_BC                    
#undef switchDebug_set_Dirichlet_BC_FC                 
#undef switchDebug_set_wall_BC_FC                      
#undef switchDebug_set_Neumann_BC                      
#undef switchDebug_set_Neumann_BC_FC                   
#undef switchDebug_set_Periodic_BC                     
#undef switchDebug_output_CC                           
#undef switchDebug_output_CC_MM                        
#undef switchDebug_output_FC                           
#undef switchDebug_output_FC_MM                        
#undef switchDebug_printData_4d                        
                                                           


 


/*______________________________________________________________________
*    E  X  E  C  U  T  I  O  N     S  W  I  T  C  H  E  S 
* Switches for executing sections of code.
* if switch = 1 then compute
*_______________________________________________________________________*/
#define switch_explicit_implicit                        0       /* = 0 then run the code in explicit mode   */
                                                                /* = 1 then run in implicit mode            */
#define switch_Compute_burgers_eq                       0       /* This switch reduces the computation to   */
                                                                /* to computing Burgers eq.  by turning     */
                                                                /* off all of the sources and sinks of mass */
                                                                /* momentum and energy                      */
#define switchOveride_Initial_Conditions                0       /* if you want to include a file that       */
                                                                /* overides the initial conditions specified*/
                                                                /* by the input file                        */
#define switch_step1_OnOff                              1       /* Step 1 Evaluate equation of state        */
#define switch_step2_OnOff                              1       /* Step 2 face-centered vel. and delpress   */
#define switch_step3_OnOff                              1       /* Step 3 face-centered pressure            */
#define switch_step4_OnOff                              1       /* Step 4 source/sink of momentum and enery */
#define switch_step4_stress_source_OnOff                0       /*        shear stress terms                */
#define switch_step4_NIST_fire                          0       /*        NIST fire model                   */
#define switch_step5_OnOff                              1       /* Step 5 Accumulate sources and sinks      */
#define switch_step6_OnOff                              1       /* Step 6 Advection of mass,momentum and    */
                                                                /*        internal energy.                  */
#define switch_step7_OnOff                              1       /* Step 7 advance in time                   */
#define tecplot                                         0       /* used for dumping tecplot files           */





/*______________________________________________________________________
*      D  E  B  U  G     S  W  I  T  C  H  E  S 
* Switches for printing and plotting debugging information
*   MAIN CODE ROUTINES
*_______________________________________________________________________*/
#define switchDebug_main                                1     /* main program                             */
                                                              /* = 1 plot vars. =2 print to stderr        */
#define switchDebug_main_input                          0     /* visualize the inputs                     */

#define switchDebug_find_delta_time                     1       /* Finding the new time step with CFL cond.  */
/*__________________________________
*   Input and initialization functions
*   Problem setup
*___________________________________*/
#define switchDebug_readInputFile                       1
#define switchDebug_initializeVars                      0     /* switch for printing out initialized vars  */
#define switchDebug_grid                                0     /* grid generation                           */

/*__________________________________
*  Calc_momentum
*___________________________________*/
#define switchDebug_calc_flux_or_primitive_vars         0     /*  = 1 output to stderr, =2 plot data       */

/*__________________________________
*   Step 1:  Equation of state
*___________________________________*/
#define switchDebug_equation_of_state                   0     /* press_CC                                 */

/*__________________________________
*   Step 2:  PCGMG pressure switches
*___________________________________*/
#define switchDebug_pcgmg_test                          0     /* Use some test inside of the core routines*/
#define switchDebug_pressure_PCG                        0     /* Plot delPress_CC                         */
#define switchDebug_press_eq_residual                   0     /* =1 printout div_vel, dpress_dt, residual */
                                                              /* =2 plot of div_vel, dpress_dt, residual  */
#define switchDebug_compute_face_centered_velocities    0     /* Face-centered velocities                 */

/*__________________________________
*   Step 3: Face-centered pressure
*___________________________________*/
#define switchDebug_p_face                              0     /* face-centered pressure                   */

/*__________________________________
*   Step 4: Compute and accumulate the
*           source and sinks terms
*___________________________________*/
/*__________________________________
*   NIST Fire Model
*___________________________________*/
#define switchDebug_update_Q_TE_and_add_Q_cell          1     /* NIST transfer heat from TE to cells       */


#define switchDebug_accumulate_momentum_source_sinks    0     /* contour xmom_source                       */
#define switchDebug_accumulate_energy_source_sinks      0     /* contours of int. energy                   */
#define switchDebug_shear_stress_Xdir                   0     /* face-centered shear stress                */
#define switchDebug_shear_stress_Ydir                   0     /* face-centered shear stress                */

/*__________________________________
*   Step 5:  Lagrangian
*___________________________________*/
#define switchDebug_lagrangian_values                   0
#define switchDebug_lagrangian_vol                      0     /* Lagrangian_volume                         */

/*__________________________________
*   Step 6:  Advection
*___________________________________*/
#define switchDebug_q_exact_solution                    0     /* print/plot the exact solution to square   */
#define switchDebug_find_q_vertex_max                   0     /* plotting advection vertex max and min     */
#define switchDebug_Advect_q_out_flux                   0     /* plotting inside of q_out_flux             */
#define switchDebug_Advect_q_in_flux                    0     /* plotting inside of q_in_flux              */
#define switchDebug_Advect_influx_outflux_volume        0     /* plotting inside of influx_outflux volume  */
#define switchDebug_advect_preprocess                   0     /* plotting inside of advect_preprocess      */
#define switchDebug_Advect_gradient_limiter             0     /* plotting inside of gradiend_limter        */
#define switchDebug_Advect_q                            0     /* plotting inside of the functin Advect_q   */

/*__________________________________
*   Step7 7: Advect and advance in timee
*___________________________________*/
#define switchDebug_advect_and_advance_in_time          0     /*  = 1 output to stderr, =2 plot data       */
 
/*__________________________________
*   Boundary Conditions
*___________________________________*/
#define switchDebug_update_CC_physical_boundary_conditions      0
#define switchDebug_update_CC_FC_physical_boundary_conditions   0
#define switchDebug_set_Dirichlet_BC                    0     /* printout                                 */
#define switchDebug_set_Dirichlet_BC_FC                 0     /* printout                                 */
#define switchDebug_set_wall_BC_FC                      0     /* printout                                 */
#define switchDebug_set_Neumann_BC                      0     /* printout                                 */
#define switchDebug_set_Neumann_BC_FC                   0
#define switchDebug_set_Periodic_BC                     0     /* printout                                 */

/*__________________________________
*   Output to tecplot file
*___________________________________*/
#define switchDebug_output_CC                           0     /* switch for printing debug info            */
#define switchDebug_output_CC_MM                        0     /* switch for printing debug info            */
#define switchDebug_output_FC                           0     /* switch for printing debug info            */
#define switchDebug_output_FC_MM                        0     /* switch for printing debug info            */

/*__________________________________
* Misc debugging switches
*___________________________________*/
#define switchDebug_printData_4d                        2     /* When printing data to stderr using either */
                                                              /* (1) use %4.3lf;  (2) use %6.5lf           */     




/*______________________________________________________________________
* stopwatch switches for some of the functions  
*_______________________________________________________________________*/
#define sw_advect_and_advance_in_time                   0
#define sw_advection                                    0
#define sw_calc_flux_or_primitive_vars                  0
#define sw_interpolate_to_FC                            0
#define sw_lagrangian_mass                              0
#define sw_lagrangian_values                            0
#define sw_p_face                                       0
#define sw_vel_face                                     0


#define testproblem                                     2       /* = 1 initial shock discontinutiy          */
                                                                /* = 2 Sinusodial wave profile              */
                                                                /* = 3 Initial Linear Distribution          */
                                                                /* = 4 Expansion fan                        */
