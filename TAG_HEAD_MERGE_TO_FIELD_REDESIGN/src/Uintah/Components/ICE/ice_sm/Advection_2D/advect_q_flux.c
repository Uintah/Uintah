/* 
 ======================================================================*/
#include <sys/types.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include "nrutil+.h"
#include "functionDeclare.h"
#include "parameters.h"
#include "macros.h"
#include "switches.h"
/*---------------------------------------------------------------------
 Function:  q_out_flux--ADVECTION: Step 6.?, Compute the outflux of (q) 
    for each cell.  
 Filename:  advect_q_flux.c
 Purpose:
    Calculate the quantity \langle q \rangle for each outflux, including
    the corner flux terms

 References:
    "Compatible Fluxes for van Leer Advection" W.B VanderHeyden and 
    B.A. Kashiwa, Journal of Computational Physics, 
    146, 1-28, (1998) 
            
 Steps for each cell:  
 --------------------       
    1) calculate the gradients of q in the x, y, and z dir.        
    2) If the switch is turned on limit the gradients by the variable gradient_limiter 
    3) Calculate the quantity outflux of q for each of the outflowing volumes 
       
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       6/21/99
                                 4/19/00    rewritten so each face is 
                                            computed separately    
 
outflow volume notation in 2D
                 
                         q_outflux(TOP)
                        ______________________
                        |   |             |  |
  q_outflux_CF(TOP_L)   | + |      +      | +| q_outflux_CF(TOP_R)
                        |---|----------------|
                        |   |             |  |
  q_outflux(LEFT)       | + |     i,j,k   | +| q_outflux(RIGHT)
                        |   |             |  |
                        |---|----------------|
 q_outflux_CF(BOT_L)    | + |      +      | +| q_outflux_CF(BOT_R)
                        ----------------------
                         q_outflux(BOTTOM)
 
---------------------------------------------------------------------  */  
void q_out_flux( 
        int     xLoLimit,               /* x-array lower limit              */
        int     yLoLimit,               /* y-array lower limit              */
        int     zLoLimit,               /* z-array lower limit              */
        int     xHiLimit,               /* x-array upper limit              */
        int     yHiLimit,               /* y-array upper limit              */
        int     zHiLimit,               /* z-array upper limit              */
        double  delX,                   /* cell dimensions                  (INPUT) */
        double  delY,                   /*                                  (INPUT) */
        double  delZ,                   /*                                  (INPUT) */
        double  ***gradient_limiter,    /* vanleer type gradient limiter    (INPUT) */
        double  ****outflux_vol,        /* outflux volume                   (INPUT) */
        double  ****outflux_vol_CF,     /* corner flux contribution         (INPUT) */
        double  ****r_out_x,            /* x-dir centroid array (i,j,k,vol  (INPUT) */ 
        double  ****r_out_y,            /* y-dir centroid array             (INPUT) */                   
        double  ****r_out_z,            /* z-dir centroid array             (INPUT) */ 
        double  ****r_out_x_CF,         /* corner flux centroids            (INPUT) */
        double  ****r_out_y_CF,         /* -------//-------------                   */
        double  ****r_out_z_CF,         /* -------//-------------                   */ 
        double  ****q_outflux,          /* < q > outflux of q               (OUPUT) */
                                        /* (i, j, k, m, face)                       */
        double  ****q_outflux_CF,       /* corner flux contribution         (OUTPUT)*/
        double  ****q_CC,               /* primary data ( i, j, k, m)       (INPUT) */
        int     m               )       /* material                                 */
  
{
    int     i, j, k,                    /* cell indices                     */
            face, corner,                /* corner and face indices          */
            xLo, xHi,
            yLo, yHi,
            zLo, zHi;            
        
    double
            grad_x,                     /* Limited gradients               */
            grad_y,
            grad_z,
            ***grad_q_X,                /* gradients of q in the x,y,z dir  */
                                        /* temporary variables              */
            ***grad_q_Y,
            ***grad_q_Z;
/*__________________________________
*   Plotting variables
*___________________________________*/ 
#if switchDebug_Advect_q_out_flux
    double
        ***plot1,                        /* plot1ing array                    */       
        ***plot2;                        /* plot1ing array                    */ 

    #include "plot_declare_vars.h"
        plot1       = darray_3d(0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);    
        plot2       = darray_3d(0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM); 
#endif

/*__________________________________
* Double check inputs
*___________________________________*/                        
    assert ( xLoLimit >= 0 && xHiLimit < X_MAX_LIM);
    assert ( yLoLimit >= 0 && yHiLimit < Y_MAX_LIM);
    assert ( zLoLimit >= 0 && zHiLimit < Z_MAX_LIM);
/*______________________________________________________________________
*  Allocate Memory
*_______________________________________________________________________*/
    grad_q_X    = darray_3d(0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    grad_q_Y    = darray_3d(0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    grad_q_Z    = darray_3d(0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);

    zero_arrays_3d(
             xLoLimit,      yLoLimit,       zLoLimit,             
             xHiLimit,      yHiLimit,       zHiLimit,
             3,
             grad_q_X,      grad_q_Y,       grad_q_Z); 

/*START_DOC*/
/*______________________________________________________________________
*   Step 1 calculate the gradients of q in the x, y and z directions 
*   Include on layer of ghost cells in the computation
*_______________________________________________________________________*/
    #if SECOND_ORDER_ADVECTION == 1
    grad_q( 
             xLoLimit,      yLoLimit,       zLoLimit,
             xHiLimit,      yHiLimit,       zHiLimit,
             delX,          delY,           delZ,
             q_CC,           
             grad_q_X,      grad_q_Y,       grad_q_Z,
             m);
    #endif
  
/*______________________________________________________________________
*  Loop over all cells in the computational domain and one layer of 
*   ghostcells.
*_______________________________________________________________________*/
    xLo = GC_LO(xLoLimit);
    xHi = GC_HI(xHiLimit);
    yLo = GC_LO(yLoLimit);
    yHi = GC_HI(yHiLimit);
    zLo = GC_LO(zLoLimit);
    zHi = GC_HI(zHiLimit);                                  
    for ( i = xLo; i <= xHi; i++)
    {
        for ( j = yLo; j <= yHi; j++)
        {
            for ( k = zLo; k <= zHi; k++)
            { 
            
                /*__________________________________
                * Plotting Variables
                *___________________________________*/
                #if switchDebug_Advect_q_out_flux
                plot1[i][j][k]  = 0.0;
                plot2[i][j][k]  = 0.0;
                #endif
                /*__________________________________
                *  Step 2) limit the gradients if switch
                *  is on
                *___________________________________*/ 
                grad_x = 0.0;
                grad_y = 0.0;
                grad_z = 0.0;
                
                #if( LIMIT_GRADIENT_FLAG > 0 && SECOND_ORDER_ADVECTION == 1)
                
                    grad_x = grad_q_X[i][j][k] * gradient_limiter[i][j][k];
                    grad_y = grad_q_Y[i][j][k] * gradient_limiter[i][j][k];
                    grad_z = grad_q_Z[i][j][k] * gradient_limiter[i][j][k];
                #endif
                #if( LIMIT_GRADIENT_FLAG == 0 && SECOND_ORDER_ADVECTION == 1)
                
                    grad_x = grad_q_X[i][j][k];
                    grad_y = grad_q_Y[i][j][k];
                    grad_z = grad_q_Z[i][j][k];
                #endif

                /*__________________________________
                *  SLABS
                *___________________________________*/                       
                for ( face = TOP; face <= LEFT; face ++)
                {
                    q_outflux[i][j][k][face] = 0.0;
                    
                    if ( outflux_vol[i][j][k][face] > SMALL_NUM)
                    {
                        q_outflux[i][j][k][face] = 
                            q_CC[m][i][j][k] 
                         +  grad_x * r_out_x[i][j][k][face]
                         +  grad_y * r_out_y[i][j][k][face] 
                         +  grad_z * r_out_z[i][j][k][face];
                    }
                    #if switchDebug_Advect_q_out_flux
                    plot1[i][j][k] =  plot1[i][j][k] + q_outflux[i][j][k][face];
                    #endif
                }
                                
                /*__________________________________
                *   CORNER FLUX CONTRIBUTIONS   
                *___________________________________*/
                for ( corner = TOP_R; corner <= BOT_R; corner ++)
                {
                    q_outflux_CF[i][j][k][corner] = 0.0;
                    if(outflux_vol_CF[i][j][k][corner] > SMALL_NUM)
                    {
                        q_outflux_CF[i][j][k][corner] = 
                            q_CC[m][i][j][k] 
                         +  grad_x * r_out_x_CF[i][j][k][corner]
                         +  grad_y * r_out_y_CF[i][j][k][corner] 
                         +  grad_z * r_out_z_CF[i][j][k][corner];
                    }
                    #if switchDebug_Advect_q_out_flux
                    plot2[i][j][k] = plot2[i][j][k] + q_outflux_CF[i][j][k][corner];
                    #endif
                } 
   
            }
        }
    }

/*______________________________________________________________________
*   Section for Plotting
*_______________________________________________________________________*/ 
#if switchDebug_Advect_q_out_flux
    #define switchInclude_Advect_q_out_flux 1
    #include "debugcode.i"
    free_darray_3d( plot1,0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    free_darray_3d( plot2,0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    #undef switchInclude_Advect_q_out_flux

 #endif
 
/*______________________________________________________________________
*   Deallocate memory
*_______________________________________________________________________*/
   free_darray_3d( grad_q_X, 0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
   free_darray_3d( grad_q_Y, 0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
   free_darray_3d( grad_q_Z, 0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
/*__________________________________
*   Quite fullwarn messages
*___________________________________*/
    QUITE_FULLWARN(gradient_limiter);
}
/*STOP_DOC*/




/* 
 ======================================================================*/
#include <sys/types.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "nrutil+.h"
#include "functionDeclare.h"
#include "parameters.h"
#include "macros.h"
/*---------------------------------------------------------------------
 Function:  q_in_flux--ADVECTION: Step 6.?, Compute the influx of (q) for each cell.  
 Filename:  advect_q_flux.c
 Purpose:
    Calculate the influx contribution \langle q \rangle for each slab and corner
    flux.   
 
 References:
    "Compatible Fluxes for van Leer Advection" W.B VanderHeyden 
    and B.A. Kashiwa, Journal of Computational Physics, 146, 1-28, (1998) 
             
    
Implementation Notes:
---------------------
    The quantity q_outflux is needed from one layer of ghostcells surrounding
    the computational domain.  To avoid if-then statements inside of the
    nested for-loops is q_influx is computed out two layers of ghostcells.  The 
    array q_influx starts indexing from -1.
    
          
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       6/21/99  
       2.0          //           4/19/00    rewritten.  Allows outfluxes
                                            to come from any direction  
out/in flow volume notation in 2D
                 
                         q_outflux(TOP)
                        ______________________
                        |   |             |  |
  q_outflux_CF(TOP_L)   | + |      +      | +| q_outflux_CF(TOP_R)
                        |---|----------------|
                        |   |             |  |
  q_outflux(LEFT)       | + |     i,j,k   | +| q_outflux(RIGHT)
                        |   |             |  |
                        |---|----------------|
 q_outflux_CF(BOT_L)    | + |      +      | +| q_outflux_CF(BOT_R)
                        ----------------------
                         q_outflux(BOTTOM)
---------------------------------------------------------------------  */
void q_in_flux(
        int     xLoLimit,               /* x-array lower limit              */
        int     yLoLimit,               /* y-array lower limit              */
        int     zLoLimit,               /* z-array lower limit              */
        int     xHiLimit,               /* x-array upper limit              */
        int     yHiLimit,               /* y-array upper limit              */
        int     zHiLimit,               /* z-array upper limit              */
        double  ****q_influx,           /* <q> influx of q                  (OUPUT) */
                                        /* (i, j, k,face)                   */
        double  ****q_influx_CF,        /* corner flux contributions        (OUTPUT)*/
        double  ****q_outflux,          /* <q>  outflux of q                (INPUT) */
                                        /* (i, j, k, face)                  */ 
        double  ****q_outflux_CF,       /* corner flux contributions        (INPUT) */
        int     m               )       /* material                         */        
{
    int     i, j, k;                    /* cell indices                     */            
/*__________________________________
*   Plotting variables
*___________________________________*/ 
#if (switchDebug_Advect_q_in_flux)
    double  ***plot1,
            ***plot2,
            delX = 1.0;
    #include "plot_declare_vars.h"
    plot1   = darray_3d(0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);    
    plot2   = darray_3d(0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM); 
   
#endif
/*______________________________________________________________________
* Allocate memory,
*_______________________________________________________________________*/                       
    assert ( xLoLimit >= 0 && xHiLimit < X_MAX_LIM);
    assert ( yLoLimit >= 0 && yHiLimit < Y_MAX_LIM);
    assert ( zLoLimit >= 0 && zHiLimit < Z_MAX_LIM);
         
/*START_DOC*/ 

/*______________________________________________________________________
*  Loop over all of the cells in the computational domain and 
*   the ghostcells
*_______________________________________________________________________*/
                                  
    for ( i = xLoLimit; i <= xHiLimit; i++)
    {
        for ( j = yLoLimit; j <= yHiLimit; j++)
        {
            for ( k = zLoLimit; k <= zHiLimit; k++)
            { 
            
                 /*__________________________________
                *   Contributions from the slabs
                *___________________________________*/
                q_influx[i][j][k][TOP]        = q_outflux[i][j+1][k][BOTTOM];
                q_influx[i][j][k][BOTTOM]     = q_outflux[i][j-1][k][TOP];

                q_influx[i][j][k][RIGHT]      = q_outflux[i+1][j][k][LEFT];
                q_influx[i][j][k][LEFT]       = q_outflux[i-1][j][k][RIGHT];
                
                
                /*__________________________________
                *   ADD 3D HERE
                *___________________________________*/
                /*q_influx[i][j][k][FRONT]      = q_outflux[i][j][k+1][BACK];
                q_influx[i][j][k][BACK]       = q_outflux[i][j][k-1][FRONT];*/

                /*__________________________________
                *   Contributions from the corner flux volumes
                *___________________________________*/
                q_influx_CF[i][j][k][TOP_R]   = q_outflux_CF[i+1][j+1][k][BOT_L];
                q_influx_CF[i][j][k][BOT_R]   = q_outflux_CF[i+1][j-1][k][TOP_L];

                q_influx_CF[i][j][k][TOP_L]   = q_outflux_CF[i-1][j+1][k][BOT_R];
                q_influx_CF[i][j][k][BOT_L]   = q_outflux_CF[i-1][j-1][k][TOP_R]; 
                
                
                /*__________________________________
                *   Plotting variables
                *___________________________________*/ 
                #if (switchDebug_Advect_q_in_flux)
                plot1[i][j][k] =   q_influx[i][j][k][TOP]   + q_influx[i][j][k][BOTTOM]
                                 + q_influx[i][j][k][RIGHT] + q_influx[i][j][k][LEFT];   
                                 
                plot2[i][j][k] =   q_influx[i][j][k][TOP_R]   + q_influx[i][j][k][BOT_R]
                                 + q_influx[i][j][k][TOP_L] + q_influx[i][j][k][BOT_L]; 
                #endif            
            }
        }
    }    

 /*______________________________________________________________________
*   Section for Plotting
*_______________________________________________________________________*/ 
#if switchDebug_Advect_q_in_flux
    #define switchInclude_Advect_q_in_flux 1
   #include "debugcode.i"
   free_darray_3d( plot1, 0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
   free_darray_3d( plot2, 0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    #undef switchInclude_Advect_q_in_flux

 #endif
/*__________________________________
*   QUITEFULL WARN
*___________________________________*/ 
    m = m;
 
}
/*STOP_DOC*/
