/* 
 ======================================================================*/
#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "nrutil+.h"
#include "parameters.h"
#include "functionDeclare.h"
#include "switches.h"
#include "macros.h"
/*---------------------------------------------------------------------  
 Function:  pressure_iteration
 Filename:  pressure_iteration.c
 Purpose:  
    This function is desgined to solve the linear approximation of the
    time advanced pressure and the face centered velocity.
    
 Computational Domain:          Interior cells 
 Ghostcell data dependency:     

 References:
    Casulli, V. and Greenspan, D, Pressure Method for the Numerical Solution
    of Transient, Compressible Fluid Flows, International Journal for Numerical
    Methods in Fluids, Vol. 4, 1001-1012, (1984)
    
(2) Bulgarelli, U., Casulli, V. and Greenspan, D., "Pressure Methods for the 
    Numerical Solution of Free Surface Fluid Flows, Pineridge Press
    (1984)
            
 Steps for each cell:  
 --------------------       

    1)  Allocate memory for the intermediate velocities *half_FC and set 
        the appropriate face addresses equal to one another.
        
    2)  Set the values of the intermediate values equal to the velocity at the
        boundaries.
    3)  Set the ghostcell pressures equal to the pressures at the boundaries
    
    4)  Calculate the initial iteration for the pressure and the face-centered
        velocities.
    
  MAIN ITERATION LOOP
    5)  Now calculate the intermedate values of the velocity and the change in the
        cell-centered pressure.  Repeat the loop until the residual is below
        "CONVERGENCE_CRITERIA" 
             
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       08/03/99
 WARNINGS
    The convergence criteria is simple minded having the residual stagnate.
 ---------------------------------------------------------------------  */
void pressure_iteration(             
        int     xLoLimit,               /* x-array lower limit              */
        int     yLoLimit,               /* y-array lower limit              */
        int     zLoLimit,               /* z-array lower limit              */
        int     xHiLimit,               /* x-array upper limit              */
        int     yHiLimit,               /* y-array upper limit              */
        int     zHiLimit,               /* z-array upper limit              */
        double  delX,                   /* distance/cell, xdir              (INPUT) */
        double  delY,                   /* distance/cell, ydir              (INPUT) */
        double  delZ,                   /* distance/cell, zdir              (INPUT) */
                /*------to be treated as pointers---  (*)vel_FC(x,y,z,face,material */    
        double  ******uvel_FC,          /* u-face-centered velocity         (OUTPUT)*/
        double  ******vvel_FC,          /* v-face-centered velocity         (OUTPUT)*/
        double  ******wvel_FC,          /* w face-centered velocity         (OUTPUT)*/
                /*----------------------------------*/  
        double  ****uvel_CC,            /* u cell-centered velocity         (INPUT) */
        double  ****vvel_CC,            /* v cell-centered velocity         (INPUT) */
        double  ****wvel_CC,            /* w cell-centered velocity         (INPUT) */    
        double  ****press_CC,           /* Cell-center pressure(x,y,z,m)    (INPUT) */
        double  ****delPress_CC,        /* cell-centered change in pressure (OUTPUT)*/
        double  ****rho_CC,             /* Cell-centered density            (INPUT) */
        double  delt,                   /* delta t                          (INPUT) */
        double  *grav,                  /* Gravity(direction)               (INPUT) */
        int     ***BC_types,            /* defines which boundary conditions(INPUT) */
                                        /* have been set on each wall               */
        double  ***BC_Values,
        int     ***BC_float_or_fixed,   /* array that designates which variable is  */
                                        /* either fixed or floating on each wall of */
                                        /* the compuational domain                  */
                                              
        double  ****speedSound,         /* speed of sound (x,y,z, material) (INPUT) */

        int     nMaterials          )    
                                              
{

    int i, j, k, iter,                  /* cell indices                     */
        m;
            
    double  
            residual,                   /* residuals                        */
    /*__________________________________
    * Face centered velocities half way
    * through one iteration.
    *___________________________________*/
            ******uvel_half_FC,         /* u-face-centered velocity         */
                                        /* uvel_half_FC(x,y,z,face,material)*/
            ******vvel_half_FC,         /*  v-face-centered velocity        */
                                        /* vvel_half_FC(x,y,z,face,material)*/
            ******wvel_half_FC;         /* w face-centered velocity         */               
                                        /* wvel_half_FC(x,y,z,face,material)*/

/*__________________________________
*   Plotting variables
*___________________________________*/ 
#if switchDebug_pressure_interation 
    #include "plot_declare_vars.h"
#endif

/*_________________
* double check inputs
*__________________*/
    assert ( xLoLimit > 0 && xHiLimit < X_MAX_LIM);
    assert ( yLoLimit > 0 && yHiLimit < Y_MAX_LIM);
    assert ( zLoLimit > 0 && zHiLimit < Z_MAX_LIM); 
                                         
/* ----------------------------------------------------------------------- 
*  Define Variables and allocate memory                                                             
*-----------------------------------------------------------------------  */ 
    m                       = nMaterials;        /* HARDWIRED FOR NOW                  */                    
    iter                    = 0;
    residual                = 0.0;      
/*__________________________________
* Face-centered variables half way
* trough one iteration.
* NOTE THAT THESE ARE POINTERS.
*___________________________________*/
    uvel_half_FC    = darray_6d(0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM, 1, N_CELL_FACES, 1, 1, 1, 1);
    vvel_half_FC    = darray_6d(0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM, 1, N_CELL_FACES, 1, 1, 1, 1);
    wvel_half_FC    = darray_6d(0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM, 1, N_CELL_FACES, 1, 1, 1, 1);

/*START_DOC*/
/*__________________________________
* Now make sure that the face centered
* values know about each other.
* for example 
* [i][j][k][RIGHT] = [i-1][j][k][LEFT]
* include the ghost cells
*
*   BE CAREFUL OF THE LIMITS IN 3D
*___________________________________*/

 
    for ( k = 1; k <= Z_MAX_LIM-1; k++)
    {
        for ( j = 1; j <= Y_MAX_LIM; j++)
        {
            for ( i = 1; i <= X_MAX_LIM; i++)
            {
                uvel_half_FC[i][j][k][LEFT][1]    = uvel_half_FC[i-1][j][k][RIGHT][1];
                vvel_half_FC[i][j][k][BOTTOM][1]  = vvel_half_FC[i][j-1][k][TOP][1];
                wvel_half_FC[i][j][k][BACK][1]    = wvel_half_FC[i][j][k-1][FRONT][1];  
                *uvel_half_FC[i][j][k][LEFT][1]   = 0.0;
                *vvel_half_FC[i][j][k][BOTTOM][1] = 0.0;
                *wvel_half_FC[i][j][k][BACK][1]   = 0.0;  
            }
        }
    }
    
/*______________________________________________________________________
*           BOUNDARY CONDITIONS STUFF
*   Set the boundary conditons for the half iterations and set the ghost
*   cell pressures = edge pressure
*   
*_______________________________________________________________________*/
     
                        
                        
  update_CC_FC_physical_boundary_conditions( 
                        xLoLimit,           yLoLimit,       zLoLimit,
                        xHiLimit,           yHiLimit,       zHiLimit,
                        delX,               delY,           delZ,
                        BC_types,           BC_float_or_fixed,
                        BC_Values,
                        1,         3,
                        uvel_CC,            UVEL,           uvel_half_FC,
                        vvel_CC,            VVEL,           vvel_half_FC,
                        wvel_CC,            WVEL,           wvel_half_FC);
                        
      
/*______________________________________________________________________
*   Step 1 determine the face-centered
*   velocities for the initial interation
*_______________________________________________________________________*/                            
        vel_initial_iteration
                    (   xLoLimit,           yLoLimit,       zLoLimit,
                        xHiLimit,           yHiLimit,       zHiLimit,
                        BC_types,
                        uvel_FC,            vvel_FC,        wvel_FC,
                        press_CC,
                        rho_CC,             delt,           grav,
                        delX,               delY,           delZ,
                        1);   

/*__________________________________
* MAIN iteration loop
*___________________________________*/                                           
    iter       = 0;
    residual  = (double)BIG_NUM;
    while(  iter <= (int)MAX_ITERATION && residual >= (double)CONVERGENCE_CRITERIA  )
    {
      
        vel_Face_n_iteration(
                        xLoLimit,           yLoLimit,       zLoLimit,
                        xHiLimit,           yHiLimit,       zHiLimit,
                        delX,               delY,           delZ,
                        uvel_FC,            vvel_FC,        wvel_FC,
                        uvel_half_FC,       vvel_half_FC,   wvel_half_FC,
                        delPress_CC,        press_CC,
                        rho_CC,             delt,           grav,
                        speedSound,         m);
                        
        
        /*__________________________________
        *   Calculate the residual
        *   over cells inside of the compuational
        *   domain. Only compute the residule
        *   after the second iteration
        *___________________________________*/ 
        if(iter >= 2)
        { 
            residual   = 0.0; 
            for ( k = zLoLimit; k <= zHiLimit; k++)
            {
                for ( j = yLoLimit; j <= yHiLimit; j++)
                {
                    for ( i = xLoLimit; i <= xHiLimit; i++)
                    {   
                        residual = DMAX(residual,fabs( delPress_CC[m][i][j][k]/press_CC[m][i][j][k]));
                    }
                }
            }
        }
        /*__________________________________
        * Bullet proofing
        *___________________________________*/
        if (iter > MAX_ITERATION) 
        {
            Message(1, "Pressure iteration failure","Maximum iterations reached ","");
         }
        fprintf(stderr,"iteration %i, residual %f\n",iter, residual);         

        iter++;
    }  

/*______________________________________________________________________
*   DEBUGGING STUFF 
*_______________________________________________________________________*/  
    #if switchDebug_pressure_interation
       #define switchInclude_pressure_interation 1
       #include "debugcode.i"
       #undef switchInclude_pressure_interation
    #endif    

/*______________________________________________________________________
*   Deallocate memory
*_______________________________________________________________________*/
   free_darray_6d( uvel_half_FC, 0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM, 1, N_CELL_FACES, 1, 1, 1, 1);
   free_darray_6d( vvel_half_FC, 0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM, 1, N_CELL_FACES, 1, 1, 1, 1);
   free_darray_6d( wvel_half_FC, 0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM, 1, N_CELL_FACES, 1, 1, 1, 1);   
/*__________________________________
*   Quite fullwarn remarks is a way that
*   is compiler independent
*___________________________________*/
    QUITE_FULLWARN(BC_types[1][1][1]); 
    QUITE_FULLWARN(BC_Values[1][1][1]); 
}
/*STOP_DOC*/ 
