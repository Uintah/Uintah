/* 
 ======================================================================*/
#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "nrutil+.h"
#include "functionDeclare.h"
#include "parameters.h"
#include "switches.h"
#include "macros.h"
/* ---------------------------------------------------------------------
 Function:  advect_q--ADVECTION: Step 6, Main controller code for the advection operator.
 Filename:  advect_q.c
 Purpose:   Calculate the advection of q_CC 
   
 References:
 --------------------
    "Compatible Fluxes for van Leer Advection" W.B VanderHeyden and 
    B.A. Kashiwa, Journal of Computational Physics, 146, 1-28, (1998) 
    and Uintah-ICE CFD Multidimensional Compatible Advection Operator, Todd Harman
            
 Steps for each cell:  
 --------------------    
1)  Compute the gradient limiter for 2nd order advection
2)  Compute q outflux and q influx for each cell.
3)  Finally sum the influx and outflux portions
       
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       6/17/99    

The function advect_preprocess MUST be called prior to this function
 ---------------------------------------------------------------------  */
void advect_q(
        int     xLoLimit,               /* x-array lower limit              */
        int     yLoLimit,               /* y-array lower limit              */
        int     zLoLimit,               /* z-array lower limit              */
        int     xHiLimit,               /* x-array upper limit              */
        int     yHiLimit,               /* y-array upper limit              */
        int     zHiLimit,               /* z-array upper limit              */
        double  delX,                   /* cell dimensions                  (INPUT) */
        double  delY,
        double  delZ,
        double  ****q_CC,               /* q cell-center (i,j,k,m)          (INPUT) */
        double  ****r_out_x,            /* x-dir centroid array (i,j,k,vol  (INPUT) */ 
        double  ****r_out_y,            /* y-dir centroid array             (INPUT) */                   
        double  ****r_out_z,            /* z-dir centroid array             (INPUT) */ 
        double  ****r_out_x_CF,         /* corner flux centroids            (INPUT) */
        double  ****r_out_y_CF,         /* -------//-------------                   */
        double  ****r_out_z_CF,         /* -------//-------------                   */           
        double  ****outflux_vol,        /* array containing the size of each(INPUT) */
                                        /* of the outflux volumes(i,j,k,vol)*/ 
        double  ****outflux_vol_CF,     /* corner flux outflux volumes      (INPUT) */     
        double  ****influx_vol,         /* array containing the size of each(INPUT) */
        double  ****influx_vol_CF,      /* corner flux influx volumes       (INPUT) */

        double  ****advect_q_CC,        /* advected q_CC (i,j,k,m)          (OUPUT) */
         int     m               )      /* material                         */  
{
    int     i, j, k,                    /* cell indices                     */
            face, corner;               /* face and corner indices          */

    double
            sum_q_outflux,              /* sum of the contributions of the  */
                                        /* outflux of q                     */
            sum_q_outflux_CF,           /* corner flux contributions        */
            sum_q_influx,               /* sum of the contributions of the  */
            sum_q_influx_CF,            /* corner flux contributions        */
            ***grad_limiter,            /* limiter used to limit the gradient*/
                                        /* in a compatible fashion(i,j,k,vol*/
            ****q_influx,               /* influx of q for cell (i,j,k,vol) */
            ****q_influx_CF,            /* corner flux contributions        */
            ****q_outflux,              /* influx of q for cell (i,j,k,vol) */ 
            ****q_outflux_CF;           /* corner flux contributions        */
            
    char    should_I_write_output;                     
/*__________________________________
*   Plotting variables
*___________________________________*/ 
#if switchDebug_Advect_q
    double
        ***plot1,                        /* plot1ing array                    */       
        ***plot2,                       /* plot1ing array                    */
        ***plot3;
    #include "plot_declare_vars.h" 
    plot1 = darray_3d(0, X_MAX_LIM,   0, Y_MAX_LIM, 0, Z_MAX_LIM);    
    plot2 = darray_3d(0, X_MAX_LIM,   0, Y_MAX_LIM, 0, Z_MAX_LIM);
    plot3 = darray_3d(0, X_MAX_LIM,   0, Y_MAX_LIM, 0, Z_MAX_LIM);
    #endif                                       
/* -----------------------------------------------------------------------  */
/*  Allocate memory for the arrays                */
/* -----------------------------------------------------------------------  */
    grad_limiter    = darray_3d(0,  X_MAX_LIM,      0, Y_MAX_LIM,    0, Z_MAX_LIM);
    q_influx        = darray_4d(-1, X_MAX_LIM+1,   -1, Y_MAX_LIM+1, -1, Z_MAX_LIM, 1, N_CELL_FACES);
    q_influx_CF     = darray_4d(-1, X_MAX_LIM+1,   -1, Y_MAX_LIM+1, -1, Z_MAX_LIM, 1, N_CELL_VERTICES);
    q_outflux       = darray_4d(0,  X_MAX_LIM,      0, Y_MAX_LIM,    0, Z_MAX_LIM, 1, N_CELL_FACES);
    q_outflux_CF    = darray_4d(0,  X_MAX_LIM,      0, Y_MAX_LIM,    0, Z_MAX_LIM, 1, N_CELL_VERTICES);
/*__________________________________
* double check inputs
*___________________________________*/   
    assert ( xLoLimit >= 0 && xHiLimit < X_MAX_LIM);
    assert ( yLoLimit >= 0 && yHiLimit < Y_MAX_LIM);
    assert ( zLoLimit >= 0 && zHiLimit < Z_MAX_LIM);
    
    if(xLoLimit < 1 || yLoLimit < 1 || zLoLimit < 1)
    {
        Message(1,"ADVECTION:","The lower limits of the array must be = 1",
                "and there must be at least one layer of ghostcells");
    }   
/*___________________________________
* Step 1) Determine the gradient limiter, alpha
*____________________________________*/
                        
    gradient_limiter(
                        xLoLimit,       yLoLimit,       zLoLimit,
                        xHiLimit,       yHiLimit,       zHiLimit,
                        delX,           delY,           delZ,
                        q_CC,           grad_limiter,
                        m);
                              
/*__________________________________
*  Step 2) Determine the influx and 
*   outflux of q at each cell
*___________________________________*/
                            
       q_out_flux(                                              
                        xLoLimit,       yLoLimit,       zLoLimit,
                        xHiLimit,       yHiLimit,       zHiLimit,
                        delX,           delY,           delZ,
                        grad_limiter,                   
                        outflux_vol,    outflux_vol_CF,
                        r_out_x,        r_out_y,        r_out_z,
                        r_out_x_CF,     r_out_y_CF,     r_out_z_CF,
                        q_outflux,      q_outflux_CF,   q_CC,
                        m);
  
                               
        q_in_flux(                                              
                        xLoLimit,       yLoLimit,       zLoLimit,
                        xHiLimit,       yHiLimit,       zHiLimit,
                        q_influx,       q_influx_CF,
                        q_outflux,      q_outflux_CF,
                        m);
 
/*__________________________________
*  Step 3) Finally determine the 
* advection of q
*___________________________________*/                            
                                  
    for ( i = xLoLimit; i <= xHiLimit; i++)
    {
        for ( j = yLoLimit; j <= yHiLimit; j++)
        {
            for ( k = zLoLimit; k <= zHiLimit; k++)
            { 
            
                /*__________________________________
                *  OUTFLUX: SLAB 
                *___________________________________*/
                sum_q_outflux       = 0.0;
                sum_q_outflux_CF    = 0.0;
                sum_q_influx        = 0.0;
                sum_q_influx_CF     = 0.0;
                
          
                for( face = TOP; face <= LEFT; face++ )
                {
                    sum_q_outflux   = sum_q_outflux + 
                                    q_outflux[i][j][k][face] * outflux_vol[i][j][k][face];
                }
                /*__________________________________
                *  OUTFLUX: CORNER FLUX
                *___________________________________*/

                for (corner = TOP_R; corner <= BOT_R; corner ++)
                {
                    sum_q_outflux_CF = sum_q_outflux_CF + 
                                    q_outflux_CF[i][j][k][corner]   * outflux_vol_CF[i][j][k][corner];
                }
                /*__________________________________
                *  INFLUX: INFLUX
                *___________________________________*/
               for( face = TOP; face <= LEFT; face++ )
                {
                    sum_q_influx    = sum_q_influx +
                                     q_influx[i][j][k][face]*influx_vol[i][j][k][face];
                }
                /*__________________________________
                *  INFLUX: CORNER FLUX
                *___________________________________*/
                for (corner = TOP_R; corner <= BOT_R; corner ++)
                {
                    sum_q_influx_CF = sum_q_influx_CF + 
                                    q_influx_CF[i][j][k][corner]   * influx_vol_CF[i][j][k][corner];
                }
                /*__________________________________
                * Calculate the advected q at t + delta t
                *___________________________________*/ 
                
                advect_q_CC[m][i][j][k] = - sum_q_outflux - sum_q_outflux_CF 
                                          + sum_q_influx  + sum_q_influx_CF; 
             
                /*__________________________________
                *   PLOTTING VARIABLES
                *___________________________________*/
                #if switchDebug_Advect_q
                    plot1[i][j][k] = 0.0;
                    plot2[i][j][k] = 0.0;
                   
                    plot1[i][j][k]  = sum_q_outflux + sum_q_outflux_CF;
                    plot2[i][j][k]  = sum_q_influx + sum_q_influx_CF;
                    plot3[i][j][k]  = advect_q_CC[m][i][j][k];
                #endif

            }
        }
    }
/*______________________________________________________________________
*   Plot the results (MUST HARDWIRE WHAT YOU WANT TO VIEW)
*_______________________________________________________________________*/
#if switchDebug_Advect_q 
    should_I_write_output = *getenv("SHOULD_I_WRITE_OUTPUT");
    if ( should_I_write_output == '1')
    {
        #define switchInclude_Advect_q 1 
        #include "debugcode.i"
        free_darray_3d( plot1, 0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
        free_darray_3d( plot2, 0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
        free_darray_3d( plot3, 0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);    
        #undef switchInclude_Advect_q 
    }
 
#endif   

/*______________________________________________________________________
* DEALLOCATE MEMORY
*_______________________________________________________________________*/
   free_darray_3d( grad_limiter,0, X_MAX_LIM,    0, Y_MAX_LIM,   0, Z_MAX_LIM);
   free_darray_4d(q_influx,    -1, X_MAX_LIM+1, -1, Y_MAX_LIM+1, -1, Z_MAX_LIM+1, 1, N_CELL_FACES  );                   
   free_darray_4d(q_outflux ,   0, X_MAX_LIM,    0, Y_MAX_LIM,    0, Z_MAX_LIM,   1, N_CELL_FACES  );
   free_darray_4d(q_influx_CF, -1, X_MAX_LIM+1, -1, Y_MAX_LIM+1, -1, Z_MAX_LIM+1, 1, N_CELL_VERTICES  );                   
   free_darray_4d(q_outflux_CF ,0, X_MAX_LIM,    0, Y_MAX_LIM,    0, Z_MAX_LIM,   1, N_CELL_VERTICES  );

/*__________________________________
*   Quite fullwarn remarks in a way that
*   is compiler independent
*___________________________________*/ 
    should_I_write_output = should_I_write_output;
}   
/*STOP_DOC*/  


/* 
 ======================================================================*/
#include <sys/types.h>
#include <math.h>
#include <float.h>                      /* defines  DBL_MIN                 */ 
#include <assert.h>
#include "nrutil+.h"
#include "functionDeclare.h"
#include "parameters.h"


/* ---------------------------------------------------------------------
 Function:  influx_outflux_volume--ADVECTION: Step 6.?, Computes the fluxes of volume Delta_V_1 to Delta_V_6.
 Filename:  advect_q.c
 Purpose:   calculate the individual outfluxes and influxes for each cell.
            This includes the slabs and corner fluxes
 
 References:
 --------------------
    "Compatible Fluxes for van Leer Advection" W.B VanderHeyden and 
    B.A. Kashiwa, Journal of Computational Physics, 146, 1-28, (1998) 
    and Uintah-ICE CFD Multidimensional Compatible Advection Operator, Todd Harman
            
 Steps for each cell:  
 --------------------
 1) calculate the volume for each outflux
 3) set the influx_volume for the appropriate cell = to the q_outflux of the 
    adjacent cell. 

Implementation notes:
    The outflux of volume is calculated in each cell in the computational domain
    + one layer of ghostcells surrounding the domain. 
    
 History:
    Version   Programmer         Date       Descriptionø
    -------   ----------         ----       -----------
       1.0     Todd Harman       6/22/99
       2.0                       4/19/00     Completely rewritten


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
                         
                              
 CAVEAT:
    The face-centered velocity needs to be defined on all faces for each cell
    in the computational domain and a single ghostcell layer.   
 ---------------------------------------------------------------------  */
void influx_outflux_volume(    
        int     xLoLimit,               /* x-array lower limit              */
        int     yLoLimit,               /* y-array lower limit              */
        int     zLoLimit,               /* z-array lower limit              */
        int     xHiLimit,               /* x-array upper limit              */
        int     yHiLimit,               /* y-array upper limit              */
        int     zHiLimit,               /* z-array upper limit              */
        double  delX,                   /* cell dimensions                  (INPUT) */
        double  delY,                   /*                                  (INPUT) */
        double  delZ,                   /*                                  (INPUT) */
        double  delt,                   /* time increment [sec]             (INPUT) */
                /* pointers             *(*)vel_FC(x,y,z,face,material              */
        double  ******uvel_FC,          /* u-face-centered velocity         (INPUT) */
        double  ******vvel_FC,          /* *v-face-centered velocity        (INPUT) */
        double  ******wvel_FC,          /* w face-centered velocity         (INPUT) */
        double  ****influx_vol,         /* influx of vol(x, y, z, vol.)     (OUPUT) */
        double  ****influx_vol_CF,      /* corner flux contributions        (OUTPUT)*/
        double  ****outflux_vol,        /* outflux of vol(x, y, z, vol.)    (OUPUT) */
        double  ****outflux_vol_CF,     /* corner flux contributions        (OUTPUT)*/
        int     m           )           /* material                         */
        
                                       
{
    int     i, j, k,                    /* cell indices                     */
            xLo, xHi,
            yLo, yHi,
            zLo, zHi,
            face;
    double
            delY_top,                   /* see diagram above                */   
            delY_bottom,
            delX_right, 
            delX_left,  
            delX_tmp,
            delY_tmp,
            total_vol,                  /* total volume of the cell         */
            total_fluxvol,              /* sum of all of the fluxed volumes */
            bullet_proof_test;
/*__________________________________
*   Plotting variables
*___________________________________*/ 
#if switchDebug_Advect_influx_outflux_volume
    double
        ***plot1,                        /* plot1ing array                    */       
        ***plot2;                        /* plot1ing array                    */ 

        #include "plot_declare_vars.h"
        plot1       = darray_3d(0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);    
        plot2       = darray_3d(0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM); 
#endif
/*_________________
* double check inputs
*__________________*/
    assert ( xLoLimit >= 0 && xHiLimit < X_MAX_LIM);
    assert ( yLoLimit >= 0 && yHiLimit < Y_MAX_LIM);
    assert ( zLoLimit >= 0 && zHiLimit < Z_MAX_LIM);   

/*__________________________________
*   initialize the looping limits
*___________________________________*/  
    bullet_proof_test = 0.0;   
/*START_DOC*/
    xLo = GC_LO(xLoLimit);
    xHi = GC_HI(xHiLimit);
    yLo = GC_LO(yLoLimit);
    yHi = GC_HI(yHiLimit);
    zLo = GC_LO(zLoLimit);
    zHi = GC_HI(zHiLimit); 
/*______________________________________________________________________
*   Calculate the outflux of volume for everycell inside of the computational
*   domain + 1 layer of ghost cells. 
*_______________________________________________________________________*/     
    for ( i = xLo; i <= xHi; i++)
    {
        for ( j = yLo; j <= yHi; j++)
        {
            for ( k = zLo; k <= zHi; k++)
            {
                /*__________________________________
                *   SLAB outfluxes
                *___________________________________*/
                delY_top    = MAX(0.0, (*vvel_FC[i][j][k][TOP][m]     * delt) );
                delY_bottom = MAX(0.0,-(*vvel_FC[i][j][k][BOTTOM][m]  * delt) );
                delX_right  = MAX(0.0, (*uvel_FC[i][j][k][RIGHT][m]   * delt) );
                delX_left   = MAX(0.0,-(*uvel_FC[i][j][k][LEFT][m]    * delt) );

                delX_tmp    = delX - delX_right - delX_left;
                outflux_vol[i][j][k][TOP]        = MAX(0.0, *vvel_FC[i][j][k][TOP][m]     * delt * delX_tmp * delZ);
                outflux_vol[i][j][k][BOTTOM]     = MAX(0.0,-*vvel_FC[i][j][k][BOTTOM][m]  * delt * delX_tmp * delZ);

                delY_tmp    = delY - delY_top - delY_bottom;
                outflux_vol[i][j][k][RIGHT]      = MAX(0.0, *uvel_FC[i][j][k][RIGHT][m]   * delt * delY_tmp * delZ);
                outflux_vol[i][j][k][LEFT]       = MAX(0.0,-*uvel_FC[i][j][k][LEFT][m]    * delt * delY_tmp * delZ);

                #if 0   /* need for 3D  */
                outflux_vol[i][j][k][FRONT]      = MAX(0.0, *wvel_FC[i][j][k][FRONT][m]   * delt * delX * delY);
                outflux_vol[i][j][k][BACK]       = MAX(0.0,-*wvel_FC[i][j][k][BACK][m]    * delt * delX * delY);  
                #endif   
                /*__________________________________
                *   Corner flux terms
                *___________________________________*/
                outflux_vol_CF[i][j][k][TOP_R]   = delY_top      * delX_right;
                outflux_vol_CF[i][j][k][TOP_L]   = delY_top      * delX_left;
 
                outflux_vol_CF[i][j][k][BOT_R]   = delY_bottom   * delX_right;
                outflux_vol_CF[i][j][k][BOT_L]   = delY_bottom   * delX_left;                
            }
        }
    }
/*__________________________________
*   INFLUX TERMS
*___________________________________*/
    for ( i = xLoLimit; i <= xHiLimit; i++)
    {
        for ( j = yLoLimit; j <= yHiLimit; j++)
        {
            for ( k = zLoLimit; k <= zHiLimit; k++)
            {                
               /*__________________________________
               *   INFLUX SLABS
               *___________________________________*/
               influx_vol[i][j][k][TOP]        = outflux_vol[i][j+1][k][BOTTOM];
               influx_vol[i][j][k][BOTTOM]     = outflux_vol[i][j-1][k][TOP];

               influx_vol[i][j][k][RIGHT]      = outflux_vol[i+1][j][k][LEFT];
               influx_vol[i][j][k][LEFT]       = outflux_vol[i-1][j][k][RIGHT];

               influx_vol[i][j][k][FRONT]      = outflux_vol[i][j][k+1][BACK];
               influx_vol[i][j][k][BACK]       = outflux_vol[i][j][k-1][FRONT];

               /*__________________________________
               *   INFLUX CORNER FLUXES
               *___________________________________*/
               influx_vol_CF[i][j][k][TOP_R]   = outflux_vol_CF[i+1][j+1][k][BOT_L];
               influx_vol_CF[i][j][k][BOT_R]   = outflux_vol_CF[i+1][j-1][k][TOP_L];

               influx_vol_CF[i][j][k][TOP_L]   = outflux_vol_CF[i-1][j+1][k][BOT_R];
               influx_vol_CF[i][j][k][BOT_L]   = outflux_vol_CF[i-1][j-1][k][TOP_R];       

                /*__________________________________
                * Bullet proofing
                *___________________________________*/
                total_fluxvol =  
                              outflux_vol[i][j][k][TOP]         + outflux_vol[i][j][k][BOTTOM]
                            + outflux_vol[i][j][k][RIGHT]       + outflux_vol[i][j][k][LEFT]
                            + outflux_vol_CF[i][j][k][TOP_R]    + outflux_vol_CF[i][j][k][BOT_R]
                            + outflux_vol_CF[i][j][k][TOP_L]    + outflux_vol_CF[i][j][k][BOT_L];
   
                bullet_proof_test = MAX(total_fluxvol, bullet_proof_test);
            }
        }
    }
    
    /*__________________________________
    *  Bulletproofing
    *___________________________________*/
    total_vol = delX * delY * delZ;
    if (bullet_proof_test > total_vol)
    Message(1, "ERROR: advect_q.c \nThe time step is too large.",
               "Outflux_vol > cell volume", "Reduce the CFL and try again");
/*__________________________________
*   Quite fullwarn remarks in a way that
*   is compiler independent
*___________________________________*/
    QUITE_FULLWARN(*wvel_FC[0][0][0][1][1]);      
    QUITE_FULLWARN(delZ);         
    face = face;                  
    
    
/*______________________________________________________________________
*   Section for Plotting
*_______________________________________________________________________*/ 
#if switchDebug_Advect_influx_outflux_volume
    #define switchInclude_Advect_influx_outflux_volume 1
    for ( i = xLoLimit; i <= xHiLimit; i++)
    {
        for ( j = yLoLimit; j <= yHiLimit; j++)
        {
            for ( k = zLoLimit; k <= zHiLimit; k++)
            { 
                plot1[i][j][k]  = 0.0;
                plot2[i][j][k]  = 0.0;

                for( face = TOP; face <= LEFT; face++ )
                {
                    plot1[i][j][k]    = plot1[i][j][k] + outflux_vol[i][j][k][face] * 1000.0;
                }

                for( face = TOP; face <= LEFT; face++ )
                {
                    plot2[i][j][k]  = plot2[i][j][k] +influx_vol[i][j][k][face] * 1000.0;
                }
               
                
            }
        }
    }
    #include "debugcode.i"
    free_darray_3d( plot1,0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    free_darray_3d( plot2,0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    #undef switchInclude_Advect_influx_outflux_volume

 #endif
}
/*STOP_DOC*/                    
