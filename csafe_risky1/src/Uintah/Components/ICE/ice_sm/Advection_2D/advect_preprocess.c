/* 
 ======================================================================*/
#include <assert.h>
#include <stdlib.h>
#include "nrutil+.h"
#include "functionDeclare.h"
#include "parameters.h"
#include "switches.h"
#include "macros.h"
/* ---------------------------------------------------------------------
 Function:  advect_preprocess--Advection:Steps 6.? and 6.?, Compute influx and outflux of volume for each cell.
 Filename:  advect_q.c
 Purpose:   Calculate the influx, outflux of volume for each cell and the 
 outflux volume centroid.  Essentially, this function calculates stuff that
 only needs to be computed once but is used repeatedly in timeadvance.c
   
 References:
 --------------------
    "Compatible Fluxes for van Leer Advection" W.B VanderHeyden and 
    B.A. Kashiwa, Journal of Computational Physics, 146, 1-28, (1998) 
    and "Blueprint for the Uintah ICE-CFD code, Todd Harman
            
       
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       11/29/99  

 ---------------------------------------------------------------------  */
void advect_preprocess(
        int     xLoLimit,               /* x-array lower limit              */
        int     yLoLimit,               /* y-array lower limit              */
        int     zLoLimit,               /* z-array lower limit              */
        int     xHiLimit,               /* x-array upper limit              */
        int     yHiLimit,               /* y-array upper limit              */
        int     zHiLimit,               /* z-array upper limit              */
        double  delX,                   /* cell dimensions                  (INPUT) */
        double  delY,
        double  delZ,
        double  delt,                   /* time increment [sec]             (INPUT) */
                /* pointers             *(*)vel_FC(x,y,z,face,material              */
        double  ******uvel_FC,          /* *u-face-centered velocity        (INPUT) */
        double  ******vvel_FC,          /* *v-face-centered velocity        (INPUT) */
        double  ******wvel_FC,          /* *w face-centered velocity        (INPUT) */
        double  ****r_out_x,            /* x-dir centroid array (i,j,k,vol  (OUTPUT)*/ 
        double  ****r_out_y,            /* y-dir centroid array             (OUTPUT)*/                   
        double  ****r_out_z,            /* z-dir centroid array             (OUTPUT)*/                  
        double  ****r_out_x_CF,         /* Corner Flux Contributions        (OUTPUT)*/ 
        double  ****r_out_y_CF,         /*                                  (OUTPUT)*/                   
        double  ****r_out_z_CF,         /*                                  (OUTPUT)*/                  
        double  ****outflux_vol,        /* array containing the size of each(OUTPUT)*/
                                        /* of the outflux vols(i,j,k,vol)*/ 
        double  ****outflux_vol_CF,     /* corner flux contributions         */
        double  ****influx_vol,         /* array containing the size of each(OUTPUT)*/
                                        /* of the influx volumes            */ 
        double  ****influx_vol_CF,      /* corner flux contributions        */                                        
         int     m               )      /* material                         */  
{
/*__________________________________
*   Plotting variables
*___________________________________*/ 
#if switchDebug_advect_preprocess
    int     i, j, k, in, out;
    double
        ***plot1,                        /* plot1ing array                    */       
        ***plot2;                        /* plot1ing array                    */
  #include "plot_declare_vars.h" 
  plot1       = darray_3d(0, X_MAX_LIM,   0, Y_MAX_LIM, 0, Z_MAX_LIM);    
  plot2       = darray_3d(0, X_MAX_LIM,   0, Y_MAX_LIM, 0, Z_MAX_LIM);
#endif            
/*__________________________________
* double check inputs
*___________________________________*/   
    assert ( xLoLimit >= 0 && xHiLimit < X_MAX_LIM);
    assert ( yLoLimit >= 0 && yHiLimit < Y_MAX_LIM);
    assert ( zLoLimit >= 0 && zHiLimit < Z_MAX_LIM);
/*START_DOC*/
/*______________________________________________________________________
* Compute the influx, outflux vols and the outflux vol centroids.
*_______________________________________________________________________*/
    influx_outflux_volume(    
                        xLoLimit,       yLoLimit,       zLoLimit,
                        xHiLimit,       yHiLimit,       zHiLimit,
                        delX,           delY,           delZ,
                        delt,
                        uvel_FC,        vvel_FC,        wvel_FC,
                        influx_vol,     influx_vol_CF,
                        outflux_vol,    outflux_vol_CF,
                        m);

    outflow_vol_centroid(    
                        xLoLimit,       yLoLimit,       zLoLimit,
                        xHiLimit,       yHiLimit,       zHiLimit,
                        delX,           delY,           delZ,
                        delt,
                        uvel_FC,        vvel_FC,        wvel_FC,
                        r_out_x,        r_out_y,        r_out_z,
                        r_out_x_CF,     r_out_y_CF,     r_out_z_CF,
                        m);
/*______________________________________________________________________
*   Plotting stuff
*_______________________________________________________________________*/
#if switchDebug_advect_preprocess
    #define switchInclude_advect_preprocess 1 
    for ( i = xLoLimit; i <= xHiLimit; i++)
    {
        for ( j = yLoLimit; j <= yHiLimit; j++)
        {
            for ( k = zLoLimit; k <= zHiLimit; k++)
            { 
                plot1[i][j][k]  = 0.0;
                plot2[i][j][k] = 0.0;
                for( out = TOP; out <= LEFT; out++ )
                {
                    plot1[i][j][k]    = plot1[i][j][k] + outflux_vol[i][j][k][out];
                }
                
                for( in = TOP; in <= LEFT; in++ )
                {
                    plot2[i][j][k]  = plot2[i][j][k] +influx_vol[i][j][k][in];
                }
            }
        }
    }

    #include "debugcode.i"
    free_darray_3d( plot1, 0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    free_darray_3d( plot2,0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    
    #undef switchInclude_Advect_q 
#endif
/*STOP_DOC*/

}
