/*______________________________________________________________________
* Switches for printing debugging information
*_______________________________________________________________________*/
#define switchDebug_readInputFile 1
#define switchDebug_output_CC 0        /* switch for printing debug info            */
#define switchDebug_output_CC_MF 0     /* switch for printing debug info            */
#define switchDebug_output_FC 0        /* switch for printing debug info            */
#define switchDebug_grid 1             /* grid generation                           */          
#define switchDebug_initializeVars 1
#define switchDebug_p_face 1            /* face-centered pressure                   */
#define switchDebug_vel_face 0          /* face centered velocity                   */
#define switchDebug_vol_L_FC 1          /* Lagrangian volume                        */
#define switchDebug_mass_L_CC 1         /* Lagrangian mass routine                  */
#define switchDebug_vel_L_CC 1          /* Lagrangian velocity routine              */
#define switchDebug_advection 0         /* advection routine                        */
/*______________________________________________________________________
* stopwatch switches for each function  
*_______________________________________________________________________*/
#define sw_interpolate_to_FC 0
#define sw_vel_face 0
#define sw_p_face 0
#define sw_lagrangian_mass 0
#define sw_lagrangian_vol 0
#define sw_lagrangian_values 0
#define sw_advection 0
#define sw_advance_in_time 0
/*______________________________________________________________________
* Switches for executing sections of code.
* if switch = 1 then compute
*_______________________________________________________________________*/
#define press_correction_vel_FC 0       /* pressure correction term in the face     */
                                        /* centered velocities                      */
#define advection_on 0                  /* advection operator in the time advanced  */
                                        /* equations                                */
    
#define tecplot 0                       /* used for dumping tecplot files           */
