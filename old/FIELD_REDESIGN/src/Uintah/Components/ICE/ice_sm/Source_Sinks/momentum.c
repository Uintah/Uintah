/* 
======================================================================*/
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include "functionDeclare.h"
#include "parameters.h"
#include "switches.h"
#include "macros.h"
#include "nrutil+.h"

/* ---------------------------------------------------------------------
 Function:  accumulate_momentum_source_sinks--SOURCE: Step 4, Accumulate all of the momentum source terms.
 Filename:  momentum.c
 Purpose:   This function accumulates all of the sources/sinks of momentum
            which is added to the current value for the momentum to form
            the Lagrangian momentum
            
 Computational Domain:
    For the pressure source term the face-centered pressure is needed from 
    all of the surrounding faces. 
    
 Ghostcell data dependency: 
    None    
   
 Version       Programmer         Date       Description                      
     -------   ----------         ----       -----------                 
        1.0     Todd Harman       10/13/99 
                                   
 ---------------------------------------------------------------------  */

void accumulate_momentum_source_sinks(
    int         xLoLimit,               /* x-array Lower Interior Nodes     */
    int         yLoLimit,               /* y-array Lower Interior Nodes     */
    int         zLoLimit,               /* z-array Lower Interior Nodes     */
    int         xHiLimit,               /* x-array Upper Interior Nodes     */
    int         yHiLimit,               /* y-array Upper Interior Nodes     */
    int         zHiLimit,               /* z-array Upper Interior Nodes     */
    double      delt,                   /* time increment                   (INPUT) */
    double      delX,                   /* cell spacing                     (INPUT) */
    double      delY,                   /*          ---//---                (INPUT) */
    double      delZ,                   /*          ---//---                (INPUT) */
    double      *grav,                  /* gravity (dir)              (INPUT) */
                                        /* 1 = x, 2 = y, 3 = z              */
    double      ****mass_CC,            /* cell-centered mass               (INPUT) */
    double      ****rho_CC,             /* Cell-centered density            (INPUT) */
    double      ******press_FC,         /* Face-center pressure             (INPUT) */
    double      ****Temp_CC,            /* Cell-centered Temperature        (INPUT) */
    double      ****cv_CC,              /* Extra variable for now           (INPUT) */
    double      ****uvel_CC,            /* cell-centered velocities         (INPUT) */
    double      ****vvel_CC,            /*          ---//---                (INPUT) */
    double      ****wvel_CC,            /*          ---//---                (INPUT) */
    double      ******tau_X_FC,         /* face-centered shear stress X-dir (OUTPUT)*/
    double      ******tau_Y_FC,         /* face-centered shear stress Y-dir (OUTPUT)*/
    double      ******tau_Z_FC,         /* face-centered shear stress Z-dir (OUTPUT)*/ 
    double      ****viscosity_CC,       /* cell centered viscosity          (INPUT) */         
    double      ****xmom_source,        /* accumlated source/sink of momentum(OUPUT)*/
    double      ****ymom_source,        /*          ---//---                (OUPUT) */
    double      ****zmom_source,        /*          ---//---                (OUPUT) */
    int         nMaterials   )
                                                          
{
    int     i, j, k, m;                 /* cell face locators               */

    double  mass,
            pressure_source,
            viscous_source,
            dummy,
            xmom_source_temp,           /* temporary variables that make    */
            ymom_source_temp,           /* debugging easier                 */
            zmom_source_temp,          
            switch_x,                   /* = 1 calculate, =0 Don't calc.    */
            switch_y,                   /*          ---//---                */
            switch_z;                   /*          ---//---                */
            
    char    should_I_write_output;
              
/*__________________________________
*   Plotting variables
*___________________________________*/
#if switchDebug_accumulate_momentum_source_sinks
    #include "plot_declare_vars.h"   
#endif
/*__________________________________
*   initialize variables
*___________________________________*/
    dummy   =0.0;
/*__________________________________
* double check inputs, 
*___________________________________*/
    assert ( xLoLimit >= 0 && xHiLimit <= X_MAX_LIM);
    assert ( yLoLimit >= 0 && yHiLimit <= Y_MAX_LIM);
    assert ( zLoLimit >= 0 && zHiLimit <= Z_MAX_LIM);
    
/*__________________________________
*  Depending on the dimensions of the
*   problem set some switches
*___________________________________*/
#if(N_DIMENSIONS == 1)
    delZ        = 1.0;
    switch_x    = 1.0;
    switch_y    = 0.0;
    switch_z    = 0.0;
#endif
#if(N_DIMENSIONS == 2)
    delZ        = 1.0;
    switch_x    = 1.0;
    switch_y    = 1.0;
    switch_z    = 0.0;
#endif
#if(N_DIMENSIONS == 3)
    switch_x    = 1.0;
    switch_y    = 1.0;
    switch_z    = 1.0;
#endif
/*__________________________________
*   Call the function that calculate
*   the viscous component
*   need to be written
*___________________________________*/
#if switch_step4_stress_source_OnOff
    shear_stress_Xdir(
                        xLoLimit,       yLoLimit,       zLoLimit,
                        xHiLimit,       yHiLimit,       zHiLimit,
                        delX,           delY,           delZ,
                        uvel_CC,        vvel_CC,        wvel_CC,
                        viscosity_CC,   tau_X_FC,       nMaterials   );
                        
   shear_stress_Ydir(
                        xLoLimit,       yLoLimit,       zLoLimit,
                        xHiLimit,       yHiLimit,       zHiLimit,
                        delX,           delY,           delZ,
                        uvel_CC,        vvel_CC,        wvel_CC,
                        viscosity_CC,   tau_Y_FC,       nMaterials   );
    #if(N_DIMENSIONS == 3)                        
   shear_stress_Zdir(
                        xLoLimit,       yLoLimit,       zLoLimit,
                        xHiLimit,       yHiLimit,       zHiLimit,
                        delX,           delY,           delZ,
                        uvel_CC,        vvel_CC,        wvel_CC,
                        viscosity_CC,   tau_Z_FC,       nMaterials   );
    #endif
#endif

/*__________________________________
*   Now accumulate the different contributions
*___________________________________*/
    for (m = 1; m <= nMaterials; m++)
    { 
        for ( i =  (xLoLimit); i <=  (xHiLimit); i++)
        {
            for ( j =  (yLoLimit); j <=  (yHiLimit); j++)
            {
                for ( k =  (zLoLimit); k <=  (zHiLimit); k++)
                {
                    mass = rho_CC[m][i][j][k] * delX * delY * delZ;
                    /*__________________________________
                    *   x-momentum
                    *___________________________________*/
                    pressure_source = *press_FC[i][j][k][RIGHT][m] - *press_FC[i][j][k][LEFT][m];

                    viscous_source  = *tau_X_FC[i][j][k][RIGHT][m] - *tau_X_FC[i][j][k][LEFT][m]
                                    + *tau_X_FC[i][j][k][TOP][m]   - *tau_X_FC[i][j][k][BOTTOM][m];
                    xmom_source_temp        =   delt * dummy - 
                                                delt * delY * delZ * pressure_source +
                                                delt * mass * grav[XDIR];
                    xmom_source[m][i][j][k] =   switch_x * xmom_source_temp;
                    /*__________________________________
                    *   y-momentum
                    *___________________________________*/                             
                    pressure_source = *press_FC[i][j][k][TOP][m] - *press_FC[i][j][k][BOTTOM][m];

                    viscous_source  = *tau_Y_FC[i][j][k][RIGHT][m] - *tau_Y_FC[i][j][k][LEFT][m]
                                    + *tau_Y_FC[i][j][k][TOP][m]   - *tau_Y_FC[i][j][k][BOTTOM][m];                          
                    ymom_source_temp        =   delt * dummy - 
                                                delt * delX * delZ * pressure_source +
                                                delt * mass *grav[YDIR];

                    ymom_source[m][i][j][k] =   switch_y * ymom_source_temp;
                    /*__________________________________
                    *   z-momentum
                    *___________________________________*/                            
                    pressure_source = *press_FC[i][j][k][FRONT][m] - *press_FC[i][j][k][BACK][m];  

                    viscous_source  = *tau_Z_FC[i][j][k][RIGHT][m] - *tau_Z_FC[i][j][k][LEFT][m]
                                    + *tau_Z_FC[i][j][k][TOP][m]   - *tau_Z_FC[i][j][k][BOTTOM][m]; 

                    zmom_source_temp        =   delt * dummy - 
                                                delt * delX * delY * pressure_source + 
                                                delt * mass *grav[ZDIR]; 

                    zmom_source[m][i][j][k] =   switch_z * zmom_source_temp;
                }
            }
        }
    }

/*STOP_DOC*/
/*______________________________________________________________________
*   DEBUGGING SECTION   
*_______________________________________________________________________*/
#if switchDebug_accumulate_momentum_source_sinks
    should_I_write_output = *getenv("SHOULD_I_WRITE_OUTPUT");
    if ( should_I_write_output == '1')
    {
         #define switchInclude_accumulate_momentum_source_sinks 1
         #include "debugcode.i"
         #undef switchInclude_accumulate_momentum_source_sinks
    }
#endif

/*__________________________________
*   Quite all fullwarn compiler remarks
*___________________________________*/
    QUITE_FULLWARN(delX);                   QUITE_FULLWARN(mass_CC[1][1][1][1]);
    QUITE_FULLWARN(rho_CC[1][0][0][0]);     QUITE_FULLWARN(Temp_CC[1][0][0][0]);
    QUITE_FULLWARN(cv_CC[1][0][0][0]);      QUITE_FULLWARN(viscosity_CC[1][0][0][0]);
    QUITE_FULLWARN(uvel_CC[1][0][0][0]);    QUITE_FULLWARN(vvel_CC[1][0][0][0]);
    QUITE_FULLWARN(wvel_CC[1][0][0][0]);    QUITE_FULLWARN(nMaterials);
    viscous_source = viscous_source;        should_I_write_output = should_I_write_output;    
}
