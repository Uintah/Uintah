/* 
 ======================================================================*/
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include "functionDeclare.h"
#include "switches.h"
#include "parameters.h"
#include "macros.h"
#include "nrutil+.h"
#define switch_include_grad_pressure 1  /* To include the gradient of the   */
                                        /* pressure in the eq               */
                                            
                                                
/*
 Function:  compute_face_centered_velocities--PRESSURE: Step 2, Calculates the face centered velocities from cell-centered data. 
 Filename:  interpolate_vel_CC_to_FC.c
 Purpose:
   This function calculates the face centered velocities.
 
 steps:
    1)  Compute the face-centered velocities inside the compuational domain
    2)  Update the face-centered boundary conditions

 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       02/22/99    
 
 NOTE:  I'm doing twice as much work by computing both the faces in each cell.
 By faces I mean top and bottom, left and right and front and back.
 
 Need to add the third dimension and multimaterial loop
 ---------------------------------------------------------------------  */
void compute_face_centered_velocities( 
        int     xLoLimit,               /* x-array lower limit              */
        int     yLoLimit,               /* y-array lower limit              */
        int     zLoLimit,               /* z-array lower limit              */
        int     xHiLimit,               /* x-array upper limit              */
        int     yHiLimit,               /* y-array upper limit              */
        int     zHiLimit,               /* z-array upper limit              */
        double  delX,                   /* distance/cell, xdir              */
        double  delY,                   /* distance/cell, ydir              */
        double  delZ,                   /* distance/cell, zdir              */
        double  delt,                   /* delta t                          */
        int     ***BC_types,            /* Dirichlet/Neuman [wall][var][m]  */   
        int     ***BC_float_or_fixed,   /* float or fixed [wall][var][m]    */
        double  ***BC_Values,           /* BC values BC_values[wall][var][m]*/ 
        double  ****rho_CC,             /* Cell-centered density            */
        double  *grav,                  /* gravity xdir=1, ydir=2, zdir=3   */      
        double  ****press_L_CC,         /* Cell-center pressure             */
                                        /* [*]vel_CC(x,y,z,material         */        
        double  ****uvel_CC,            /* u-cell-centered velocity         (INPUT) */
        double  ****vvel_CC,            /*  v-cell-centered velocity        (INPUT) */
        double  ****wvel_CC,            /* w cell-centered velocity         (INPUT) */
                                        /* [*]vel_FC(x,y,z,face,material)   */ 
        double  ******uvel_FC,          /* u-face-centered velocity         (OUTPUT)*/
        double  ******vvel_FC,          /* v-face-centered velocity         (OUTPUT)*/
        double  ******wvel_FC,          /* w-face-centered velocity         (OUTPUT)*/
        int     nMaterials          )

{
    int i, j, k, f,m,                   /* cell face locators               */            
        cell,                           /* variables that change in formula */
        xLo, yLo, zLo,
        xHi, yHi, zHi;
    
    double  term1,term2,term3,          /* temp symbols to represent terms  */
            rho_FC,                     /* face centered density            */
            *grad_P;                    /* gradient of the pressure at each */
                                        /* cell-face                        */
         
/*__________________________________
* PLOTTING VARIABLES
*___________________________________*/
#if switchDebug_compute_face_centered_velocities
    #include "plot_declare_vars.h" 
#endif 

#if switchsw_compute_face_centered_velocities      
    time_t start,secs;                  /* timing variables                */
    start = time(NULL); 
#endif
/*__________________________________
*  Allocate memory for grad array
*___________________________________*/
    grad_P = dvector_nr(1,N_CELL_FACES);

/* _______________________________________________________________________

                                 ________ 
                                /  1    /|
                               /_______/ |
                              |       | ______(3)
                       (4)____| I,J,K |  |     
                              |       | /      
                              |_______|/
                                  |               (6) = back face
                                 (2)              (5) = front face
                                 
    y-dir faces
_______________________________________________________________________ */

/*______________________________________________________________________
*   STEP 1)
*   Compute the face-centered velocities in the computational domain.
*_______________________________________________________________________*/
    
/*__________________________________
* double check inputs
*   WARNING: (*) can't = 0 or (*)_MAX_LIM
*___________________________________*/
    assert ( xLoLimit >= 0 && xHiLimit <= X_MAX_LIM);
    assert ( yLoLimit >= 0 && yHiLimit <= Y_MAX_LIM);
    assert ( zLoLimit >= 0 && zHiLimit <= Z_MAX_LIM);
    
    /*__________________________________
    *   TOP AND BOTTOM FACE VALUES  
    *   Extend the computations into the left
    *   and right ghost cells
    *___________________________________*/   
    xLo = xLoLimit;
    xHi = xHiLimit;

    yLo = yLoLimit;
    yHi = yHiLimit;
    zLo = zLoLimit;
    zHi = zHiLimit;

    term2 = 0.0;
    for ( m = 1; m <= nMaterials; m++)
    {
        for ( k = zLo; k <= zHi; k++)
        {
            for ( j = yLo; j <= yHi; j++)
            {
                for ( i = xLo; i <= xHi; i++)
                {   
                    cell   = j+1;            
                    grad_FC_Ydir(i, j, k, m, press_L_CC, delY, grad_P);

                    for(f = TOP; f <= BOTTOM; f++)
                    {
                    /*__________________________________
                    * Time n face centered velocity
                    *___________________________________*/

                        rho_FC = (rho_CC[m][i][cell][k]    + rho_CC[m][i][j][k])/2.0;

                        assert(rho_FC >= SMALL_NUM);     /* bullet proofing      */                
                        /*__________________________________
                        * interpolation to the face
                        *___________________________________*/
                         term1  = (rho_CC[m][i][cell][k]    * vvel_CC[m][i][cell][k] 
                                +   rho_CC[m][i][j][k]      * vvel_CC[m][i][j][k])/ (2.0 * rho_FC);

                        /*__________________________________
                        * pressure correction
                        *___________________________________*/
                        #if (switch_include_grad_pressure)
                        term2 =  (delt * grad_P[f]/ rho_FC);
                        #endif

                        /*__________________________________
                        * gravity term
                        *___________________________________*/

                        term3 =  delt * grav[2];

                        *uvel_FC[i][j][k][f][m] = 0.0;
                        *vvel_FC[i][j][k][f][m] = term1- term2 + term3;                    
                        *wvel_FC[i][j][k][f][m] = 0.0; 

                        /*__________________________________
                        * change cell index and signs on 
                        * pressure terms
                        *___________________________________*/
                        cell = j-1;
                    }
                }
            }
        }


    /*__________________________________
    * left and right faces
    * Extend the computations to the 
    * top and bottom ghostcells
    *___________________________________*/ 
        xLo = xLoLimit;
        xHi = xHiLimit;

        yLo = yLoLimit;
        yHi = yHiLimit;

        zLo = zLoLimit;
        zHi = zHiLimit;  
        term2 = 0.0;  
        for ( k = zLo; k <= zHi; k++)
        {
            for ( j = yLo; j <= yHi; j++)
            {
                for ( i = xLo; i <= xHi; i++)
                {   
                    cell   = i+1;       
                    grad_FC_Xdir(i, j, k, m, press_L_CC, delX, grad_P);
                    for(f = RIGHT; f <= LEFT; f++)
                    {
                    /*__________________________________
                    * Time n face centered velocity
                    *___________________________________*/ 
                        rho_FC = (rho_CC[m][cell][j][k]    + rho_CC[m][i][j][k])/2.0;
                        assert(rho_FC >= SMALL_NUM);     /* bullet proofing      */

                        /*__________________________________
                        * interpolation to the face
                        *___________________________________*/ ;           
                        term1   = (rho_CC[m][cell][j][k]    * uvel_CC[m][cell][j][k] 
                                +   rho_CC[m][i][j][k]      * uvel_CC[m][i][j][k])/ (2.0*rho_FC);

                        /*__________________________________
                        * pressure term
                        *___________________________________*/
                        #if (switch_include_grad_pressure)
                        term2 =  (delt * grad_P[f]/rho_FC); 
                        #endif
                        /*__________________________________
                        * gravity term
                        *___________________________________*/
                        term3 =  delt * grav[1]; 



                        *uvel_FC[i][j][k][f][m] = term1 - term2 + term3;
                        *vvel_FC[i][j][k][f][m] = 0.0;
                        *wvel_FC[i][j][k][f][m] = 0.0; 
                        /*__________________________________
                        * change cell index and signs on 
                        * pressure terms
                        *___________________________________*/
                        cell = i-1;

                    }
                }
            }
        }
    /*__________________________________
    * front and back faces
    * Extend the computations to the front
    * and back ghostcells
    *___________________________________*/
        xLo = xLoLimit;
        xHi = xHiLimit;    
        yLo = yLoLimit;
        yHi = yHiLimit;

        zLo = zLoLimit;
        zHi = zHiLimit;
        term2 = 0.0;
        for ( k = zLo; k <= zHi; k++)
        {
            for ( j = yLo; j <= yHi; j++)
            {
                for ( i = xLo; i <= xHi; i++)
                {   
                    cell   = k+1;            
                    grad_FC_Zdir(i, j, k, m, press_L_CC, delZ, grad_P); 

                    for(f = FRONT; f <= BACK; f++)
                    {
                    /*__________________________________
                    * Time n face centered velocity
                    *___________________________________*/   
                        rho_FC = (rho_CC[m][i][j][cell]    + rho_CC[m][i][j][k])/2.0;
                        assert(rho_FC >= SMALL_NUM);     /* bullet proofing          */ 

                        /*__________________________________
                        * interpolation to the face
                        *___________________________________*/
                        term1 = (rho_CC[m][i][j][cell]    * wvel_CC[m][i][j][cell] 
                             +   rho_CC[m][i][j][k]       * wvel_CC[m][i][j][k])/ (2.0 * rho_FC);

                        /*__________________________________
                        * pressure term
                        *___________________________________*/
                        #if (switch_include_grad_pressure)
                        term2 =  (delt * grad_P[f]/ rho_FC);
                        #endif
                        /*__________________________________
                        * gravity term
                        *___________________________________*/
 
                        term3 =  delt * grav[3]; 
                               
                        *uvel_FC[i][j][k][f][m] = 0.0;
                        *vvel_FC[i][j][k][f][m] = 0.0;

                        *wvel_FC[i][j][k][f][m] = 0.0;

                        #if (N_DIMENSIONS == 3) 
                            *wvel_FC[i][j][k][f][m] = term1 - term2 + term3; 
                        #endif
                        /*__________________________________
                        * change cell index and signs on 
                        * pressure terms
                        *___________________________________*/
                        cell = k-1;

                    }
                }
            }
        }
    }       /* material loop */

/*______________________________________________________________________
*   Step 2) Update any neumann boundary conditions
*_______________________________________________________________________*/ 
                         
    update_CC_FC_physical_boundary_conditions( 
                        xLoLimit,       yLoLimit,       zLoLimit,             
                        xHiLimit,       yHiLimit,       zHiLimit,             
                        delX,           delY,           delZ,
                        BC_types,       BC_float_or_fixed,
                        BC_Values, 
                        nMaterials,     3,                 
                        uvel_CC,        UVEL,           uvel_FC,
                        vvel_CC,        VVEL,           vvel_FC,
                        wvel_CC,        WVEL,           wvel_FC);

/*______________________________________________________________________
*   DEBUGGING AND STOP WATCH INFORMATION
*_______________________________________________________________________*/
#if switchDebug_compute_face_centered_velocities

    #define switchInclude_compute_face_centered_velocities 1
        #include "debugcode.i"
    #undef switchInclude_compute_face_centered_velocities
    
/*     for ( m = 1; m <= nMaterials; m++)
    {    
        printData_6d(       xLo,            yLo,      zLo,
                            xHi,            yHi,      zHi,
                            RIGHT,          LEFT,
                            m,              m,
                            "vel_Face_before_iterative_pressure_solver",     
                            "Uvel_FC",      uvel_FC,        0);

        printData_6d(       xLo,            yLo,      zLo,
                            xHi,            yHi,      zHi,
                            TOP,            BOTTOM,
                            m,              m,
                            "vel_Face_before_iterative_pressure_solver",     
                            "Vvel_FC",  vvel_FC,            0); 
    } */
#endif 
/*__________________________________
* Free up memory and stop the stopwatch
*___________________________________*/
    free_dvector_nr(grad_P, 1, N_CELL_FACES);
         
#if switchsw_compute_face_centered_velocities
    stopwatch("vel_face",start);
#endif

}
/*STOP_DOC*/
