/* 
 ======================================================================*/
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include "nrutil+.h"
#include "parameters.h"
#include "functionDeclare.h"
#include "switches.h"
#include "macros.h"

/*---------------------------------------------------------------------  
 Function:  press_eq_residual--PRESS: compute the residual of the pressure eq.
 Filename:  pressure_residual.c
 Purpose:  
    This function sums the residual of the pressure equation for each
    interior cell.  Based on this residual the compute_delta_Press_Using_PCGMG
    may have to be called again to beat the residual down.
    
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       1/11/00
 ---------------------------------------------------------------------  */

 void   press_eq_residual(
        int     xLoLimit,                   /* x-array lower limit              */
        int     yLoLimit,                   /* y-array lower limit              */
        int     zLoLimit,                   /* z-array lower limit              */
        int     xHiLimit,                   /* x-array upper limit              */
        int     yHiLimit,                   /* y-array upper limit              */
        int     zHiLimit,                   /* z-array upper limit              */
        double  delX,                       /* Cell Width                       */
        double  delY,                       /* Cell width y dir.                */
        double  delZ,                       /* cell width z dir.                */
        double  delt,                       /* delta time                       */
        double  ****rho_CC,                 /* Cell-centered density            */
        double  ****speedSound,             /* speed of sound (x,y,z, material) */
            /*------to be treated as pointers---*/
                                            /*______(x,y,z,face, material)______*/
        double  ******uvel_FC,              /* u-face-centered velocity         */
        double  ******vvel_FC,              /* *v-face-centered velocity        */
        double  ******wvel_FC,              /* w face-centered velocity         */
        double  ****delPress_CC,            /* change in delta p                */
        double  ****press_CC,               /* cell-centered pressure           */
        double  *residual,
        int     nMaterials)
{
    int         i,j,k,m;
    double      RHS, vol,
                delPress_dt,                /* delPress/dt                      */
                ****div_vel_FC,             /* array containing the divergence  */
                sum, 
                coeff;     
    char    should_I_write_output;

/*__________________________________
*   Plotting variables
*___________________________________*/
#if (switchDebug_press_eq_residual == 1 || switchDebug_press_eq_residual == 2)
    double      ***plot_1,***plot_2, ***plot_3;             
    #include "plot_declare_vars.h"            
    plot_1  = darray_3d(0, X_MAX_LIM,   0, Y_MAX_LIM, 0, Z_MAX_LIM);    
    plot_2  = darray_3d(0, X_MAX_LIM,   0, Y_MAX_LIM, 0, Z_MAX_LIM);
    plot_3  = darray_3d(0, X_MAX_LIM,   0, Y_MAX_LIM, 0, Z_MAX_LIM);
    zero_arrays_3d(
                        xLoLimit,       yLoLimit,       zLoLimit,             
                        xHiLimit,       yHiLimit,       zHiLimit,
                         3,             
                        plot_1,         plot_2,         plot_3);
#endif

    div_vel_FC= darray_4d(1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);

/*__________________________________
*   Compute the divergence of the
*   face centered velocity field
*___________________________________*/                
    m = nMaterials;     /* HARDWIRE FOR NOW*/
    
    divergence_of_face_centered_velocity(  
                        xLoLimit,       yLoLimit,       zLoLimit,
                        xHiLimit,       yHiLimit,       zHiLimit,
                        delX,           delY,           delZ,
                        uvel_FC,        vvel_FC,        wvel_FC,
                        div_vel_FC,     nMaterials);
                        
    /*__________________________________
    *   Now compare the LHS to the RHS
    *___________________________________*/                        
    for ( i = xLoLimit; i <= xHiLimit; i++)
    {
        for ( j = yLoLimit; j <= yHiLimit; j++)
        {
            for ( k = zLoLimit; k <= zHiLimit; k++)
            { 
                vol             = delX * delY * delZ;
                coeff           = pow(speedSound[m][i][j][k],2) * rho_CC[m][i][j][k];
                RHS             = -(coeff * div_vel_FC[m][i][j][k]) * (delt/vol);
                
                delPress_dt     = delPress_CC[m][i][j][k];
                sum             = fabs(delPress_dt - RHS);
                *residual       = DMAX(*residual, sum);
           
                /*__________________________________
                * Now define some plotting stuff
                *___________________________________*/     
#if (switchDebug_press_eq_residual == 1 || switchDebug_press_eq_residual == 2)
                plot_1[i][j][k] = RHS;
                plot_2[i][j][k] = delPress_dt;
                plot_3[i][j][k] = delPress_dt - RHS; 
#endif               
            }
        }
    }
    fprintf(stderr,"delta press_CC residual %f\n",  *residual);

/*______________________________________________________________________
*   Plotting Section
*_______________________________________________________________________*/
#if (switchDebug_press_eq_residual == 1 || switchDebug_press_eq_residual == 2)
    should_I_write_output = *getenv("SHOULD_I_WRITE_OUTPUT");
    if ( should_I_write_output == '1')
    {
         #define switchInclude_press_eq_residual 1
         #include "debugcode.i"
         #undef switchInclude_press_eq_residual
    }
         free_darray_3d( plot_1,0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
         free_darray_3d( plot_2,0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
         free_darray_3d( plot_3,0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
#endif

/*__________________________________
*   Deallocate memory
*___________________________________*/
   free_darray_4d(div_vel_FC, 1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);

/*__________________________________
*   Quite fullwarn remarks in a way that
*   is compiler independent
*___________________________________*/
    QUITE_FULLWARN(press_CC[1][1][1][1]);       QUITE_FULLWARN(*wvel_FC[1][1][1][1][1]); 
    delZ = delZ;    
    should_I_write_output = should_I_write_output; 
}    
