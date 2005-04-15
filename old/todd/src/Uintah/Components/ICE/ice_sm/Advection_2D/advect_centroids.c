
/* 
 ======================================================================*/
#include <sys/types.h>
#include <math.h>
#include <assert.h>
#include "nrutil+.h"
#include "functionDeclare.h"
#include "parameters.h"
#include "macros.h"
/* ---------------------------------------------------------------------
 Function:  outflow_vol_centroid--ADVECTION: Step 6.?, calculates the x, y, and z centroid components of each outflow flux of volume. 
 Filename:  advect_centroid.c
 Purpose:
 --------------------
   This routine calculates the x, y, and z centroid components of each outflow flux of volume,  
   both the slabs and the corner fluxes.
   
 References:
    "Compatible Fluxes for van Leer Advection" W.B VanderHeyden and 
    B.A. Kashiwa, Journal of Computational Physics, 146, 1-28, (1998) 
       
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       6/17/99    

Notation in 2D

                       
   ______________________          ______________________  _
   |   |             |  |          |   |             |  |  |  delY_top
   | + |      +      | +|          | + |      +      | +|  |
   |---|----------------|  --ytop  |---|----------------|  -
   |   |             |  |          |   |             |  |
   | + |     i,j,k   | +|          | + |     i,j,k   | +|
   |   |             |  |          |   |             |  |
   |---|----------------|  --y0    |---|----------------|  -
   | + |      +      | +|          | + |      +      | +|  | delY_bottom
   ----------------------          ----------------------  -
       |             |             |---|             |--|
       x0            xright          delX_left         delX_right
     
     
                          r_out_*(TOP)
                    ______________________
                    |   |             |  |
  r_out_*_CF(TOP_L) | + |      +      | +| r_out_*_CF(TOP_R)
                    |---|----------------|
                    |   |             |  |
  r_out_*(LEFT)     | + |     i,j,k   | +| r_out_*(RIGHT)
                    |   |             |  |
                    |---|----------------|
 r_out_*_CF(BOT_L)  | + |      +      | +| r_out_*_CF(BOT_R)
                    ----------------------
                         r_out_*(BOTTOM)
 ---------------------------------------------------------------------  */
void outflow_vol_centroid(    
        int     xLoLimit,               /* x-array lower limit              */
        int     yLoLimit,               /* y-array lower limit              */
        int     zLoLimit,               /* z-array lower limit              */
        int     xHiLimit,               /* x-array upper limit              */
        int     yHiLimit,               /* y-array upper limit              */
        int     zHiLimit,               /* z-array upper limit              */
        double  delX,                   /* cell dimensions                  (INPUT) */
        double  delY,                   /*                                  (INPUT) */
        double  delZ,                   /*                                  (INPUT) */
        double  delt,                   /* time increment sec.              (INPUT) */
        double  ******uvel_FC,          /* u-face-centered velocity         (INPUT) */
        double  ******vvel_FC,          /* v-face-centered velocity         (INPUT) */
        double  ******wvel_FC,          /* w-face-centered velocity         (INPUT) */
        double  ****r_out_x,            /* x-dir centroid array             (OUPUT) */
        double  ****r_out_y,            /* y-dir centroid array             (OUPUT) */
        double  ****r_out_z,            /* z-dir centroid array             (OUPUT) */
        double  ****r_out_x_CF,         /* Corner Flux contributions        */
        double  ****r_out_y_CF,         /* ----------//------------         */
        double  ****r_out_z_CF,         /* ----------//------------         */
        int     m )                     /* material                         (OUPUT) */

{
    int     i, j, k,                     /* cell indices                     */
            xLo, xHi,
            yLo, yHi,
            zLo, zHi;           
    double 
            delY_top,                   /* lengh of corner flux cells       */
            delY_bottom,
            delX_right,
            delX_left,
            y0, ytop,                    /* coordinates of the slabs         */
            x0, xright,
            r_y,                        /* centroids in the x and y dir     */
            r_x,
            rx_TOP_R, 
            rx_TOP_L, 
            ry_TOP_R, 
            rx_BOT_R,
            ry_BOT_R;          
     
/*_________________
* double check inputs
*__________________*/
    assert ( xLoLimit >= 0 && xHiLimit < X_MAX_LIM);
    assert ( yLoLimit >= 0 && yHiLimit < Y_MAX_LIM);
    assert ( zLoLimit >= 0 && zHiLimit < Z_MAX_LIM);

/*START_DOC*/  
    
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
                *   Find the length of each side of 
                *   a corner flux volume
                *___________________________________*/
                delY_top    = MAX(0.0, *vvel_FC[i][j][k][TOP][m]     * delt );
                delY_bottom = MAX(0.0,-*vvel_FC[i][j][k][BOTTOM][m]  * delt );
                delX_right  = MAX(0.0, *uvel_FC[i][j][k][RIGHT][m]   * delt );
                delX_left   = MAX(0.0,-*uvel_FC[i][j][k][LEFT][m]    * delt );
                
                /*__________________________________
                * x and y coordinates of the slabs
                *___________________________________*/
                y0          = delY_bottom;
                ytop        = delY - delY_top;
                x0          = delX_left;
                xright      = delX - delX_right;     
                   
                /*__________________________________
                *   For each slab
                *___________________________________*/
                r_y         = (ytop - y0)/2.0 + y0 - delY/2.0;
                r_out_x[i][j][k][RIGHT]      =  delX/2.0        - delX_right/2.0;
                r_out_y[i][j][k][RIGHT]      =  r_y;
                r_out_z[i][j][k][RIGHT]      =  0.0;

                r_out_x[i][j][k][LEFT]       =   delX_left/2.0   - delX/2.0;
                r_out_y[i][j][k][LEFT]       =   r_y;
                r_out_z[i][j][k][LEFT]       =   0.0;

                r_x         = (xright - x0)/2.0 + x0 - delX/2.0;
                r_out_x[i][j][k][TOP]        =   r_x;
                r_out_y[i][j][k][TOP]        =   delY/2.0        - delY_top/2.0;
                r_out_z[i][j][k][TOP]        =   0.0;
                
                r_out_x[i][j][k][BOTTOM]     =   r_x;
                r_out_y[i][j][k][BOTTOM]     =   delY_bottom/2.0 - delY/2.0;
                r_out_z[i][j][k][BOTTOM]     =   0.0;                

                /*__________________________________
                *   Compute the centroid of each outflux
                *   corner flux
                *___________________________________*/
                rx_TOP_R = delX/2.0          - delX_right/2.0;
                rx_TOP_L = delX_left/2.0     - delX/2.0;
                ry_TOP_R = delY/2.0          - delY_top/2.0;
                ry_BOT_R = delY_bottom/2.0   - delY/2.0;     

                r_out_x_CF[i][j][k][TOP_R]    =   rx_TOP_R;
                r_out_y_CF[i][j][k][TOP_R]    =   ry_TOP_R;
                r_out_z_CF[i][j][k][TOP_R]    =   0.0;

                r_out_x_CF[i][j][k][TOP_L]    =   rx_TOP_L;
                r_out_y_CF[i][j][k][TOP_L]    =   ry_TOP_R;
                r_out_z_CF[i][j][k][TOP_L]    =   0.0;

                r_out_x_CF[i][j][k][BOT_R]    =   rx_TOP_R;
                r_out_y_CF[i][j][k][BOT_R]    =   ry_BOT_R;
                r_out_z_CF[i][j][k][BOT_R]    =   0.0;

                r_out_x_CF[i][j][k][BOT_L]    =   rx_TOP_L;
                r_out_y_CF[i][j][k][BOT_L]    =   rx_BOT_R;
                r_out_z_CF[i][j][k][BOT_L]    =   0.0;
            }
        }
    }

/*__________________________________
*   Quite fullwarn remarks in a way that
*   is compiler independent
*___________________________________*/
    QUITE_FULLWARN(wvel_FC[1][1][1][1][1]);
    QUITE_FULLWARN(delZ);               
    QUITE_FULLWARN(r_out_z[1][1][1][1]);
    QUITE_FULLWARN(r_out_z_CF[1][1][1][1]);
}
/*STOP_DOC*/


