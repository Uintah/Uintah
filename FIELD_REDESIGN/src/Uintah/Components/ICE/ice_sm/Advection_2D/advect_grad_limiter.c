/* 
 ======================================================================*/
#include <sys/types.h>
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include "nrutil+.h"
#include "functionDeclare.h"
#include "switches.h"
#include "parameters.h"
#include "macros.h"
/* ---------------------------------------------------------------------
 Function:  gradient_limiter--ADVECTION: Step 6.?, Compute (alpha) the gradient limiter used when computing second order derivatives.
 Filename:  advect_grad_limiter.c
 Purpose:
   This routine calculates the coefficient (alpha) used in the calculation of the 
   gradients of q used in the advection operator
   
 References:
    "Compatible Fluxes for van Leer Advection" W.B VanderHeyden 
    and B.A. Kashiwa, Journal of Computational Physics, 146, 1-28, (1998) 
    and  Uintah-ICE CFD Multidimensional Compatible Advection Operator


Computational Domain:  
    The gradient limiter is calculated for each cell in the
    computational domain, and it lives at the cell center.  Dat from each of the 
    vertices is needed, step 1.
    
Ghostcell data dependency:
    The value of q is need in the ghost cell layer to compute the max and min value
    of q from the surrounding cells, step 3.
            
 Steps for each cell:
 --------------------     
    1)  Find the max and min of vertex data q for every cell.
    2)  Find the max and min of the surrounding cell-centered data q
    3)  Calculate the max. and min. of the gradient limiter
        and finally compute the actual gradient limiter for cell (i,j,k) 


 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       6/17/99    

 
Vertex Notation 
 
                   4 ______________  3
                    /|            /|
                   / |           / |
                7 /__|__________/ 8|
                 |   |          |  |
                 |   |          |  |
                 | 1 |__________|__| 2
                 |  /           |  /
                 | /            | /
                 |/_____________|/
                5                 6
     y
      |
      |___ x
     /
    z
 
 
 ---------------------------------------------------------------------  */
void gradient_limiter(    
        int     xLoLimit,               /* x-array lower limit              */
        int     yLoLimit,               /* y-array lower limit              */
        int     zLoLimit,               /* z-array lower limit              */
        int     xHiLimit,               /* x-array upper limit              */
        int     yHiLimit,               /* y-array upper limit              */
        int     zHiLimit,               /* z-array upper limit              */
        double  delX,                   /* Cell width                       */
        double  delY,                   /* Cell Width in the y dir          */
        double  delZ,                   /* Cell width in the z dir          */
        double  ****q_CC,               /* q-cell-centered                  (INPUT) */
        double  ***grad_limiter,        /* gradient limiter(i,j,k)          (OUPUT) */
        int     m           )           /* material                         */
  
{
    int         i, j, k,                /* cell face locators               */
                xLo, xHi,
                yLo, yHi,
                zLo, zHi;      
           
    double      temp,                   /* temporary variable               */
                frac,                   /* fraction                         */
                alf,
                alf1,
                alf2,
                grad_lim_max,           /* max. gradient limiter for ijk    */
                grad_lim_min,           /* min. gradient limiter for ijk    */
                q_CC_ijk,               /* abrevation for q_CC[m][i][j][k]  */
                ***q_VRTX_MAX,          /* max value of q at the vertices   */
                ***q_VRTX_MIN,          /* max value of q at the vertices   */
                ***q_CC_max,            /* max value of q in the surrounding cells   */
                ***q_CC_min;            /* min value of q in the surrounding cells   */

/*__________________________________
*   Plotting variables
*___________________________________*/
#if switchDebug_Advect_gradient_limiter 
    #include "plot_declare_vars.h"
    double
            ***test,                    /* testing array                    */       
            ***test2;                   /* testing array                    */    
        test        = darray_3d(0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);    
        test2       = darray_3d(0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM); 
#endif      
/*______________________________________________________________________
*  Allocate some local memory and zero the arrays
*_______________________________________________________________________*/
    q_VRTX_MAX  = darray_3d(0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    q_VRTX_MIN  = darray_3d(0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    q_CC_min    = darray_3d(0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    q_CC_max    = darray_3d(0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);  
    zero_arrays_3d(
                        xLoLimit,       yLoLimit,       zLoLimit,             
                        xHiLimit,       yHiLimit,       zHiLimit,
                        4,
                        q_VRTX_MAX,     q_VRTX_MIN,     q_CC_min,
                        q_CC_max);     
      
/*__________________________________
* double check inputs
*___________________________________*/
    assert ( xLoLimit >= 0 && xHiLimit < X_MAX_LIM);
    assert ( yLoLimit >= 0 && yHiLimit < Y_MAX_LIM);
    assert ( zLoLimit >= 0 && zHiLimit < Z_MAX_LIM); 
    
/*START_DOC*/ 
/*______________________________________________________________________
*
*_______________________________________________________________________*/
  
/*__________________________________
* step 1 find the max and min of 
* vertex data q for the computational
* domain and one layer of ghost cells
*___________________________________*/
    find_q_vertex_max_min(    
                    xLoLimit,       yLoLimit,       zLoLimit,
                    xHiLimit,       yHiLimit,       zHiLimit,
                    delX,           delY,           delZ,
                    q_CC,           q_VRTX_MAX,     q_VRTX_MIN,
                            m);
                            
/*__________________________________
* step 2 max and min of the surrounding
* cell-centered data q for all of the cells
* in the computational domain and one
* layer of ghost cells
*___________________________________*/ 
    find_q_CC_max_min(    
                    xLoLimit,       yLoLimit,       zLoLimit,
                    xHiLimit,       yHiLimit,       zHiLimit,
                    q_CC,           q_CC_max,       q_CC_min,
                            m);  
                            
/*__________________________________
* step 3 calculate the max and min.
* gradient limiter and finally
* that actual gradient limiter for cell
* i,j,k
*___________________________________*/                            
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


 
#if (LIMIT_GRADIENT_FLAG == 0)        
                grad_limiter[i][j][k] = 1.0;  
#endif
            
            
#if(LIMIT_GRADIENT_FLAG == 1)         
                q_CC_ijk =  q_CC[m][i][j][k];
                frac     = (q_CC_max[i][j][k]   - q_CC_ijk + SMALL_NUM)/
                           (q_VRTX_MAX[i][j][k] - q_CC_ijk + SMALL_NUM);
                grad_lim_max = DMAX(0.0, frac); 

                frac     = (q_CC_min[i][j][k]   - q_CC_ijk + SMALL_NUM)/
                           (q_VRTX_MIN[i][j][k] - q_CC_ijk + SMALL_NUM);
                grad_lim_min = DMAX(0.0, frac);

                temp = DMIN(1.0, grad_lim_max);
                temp = DMIN(temp,grad_lim_min);
                grad_limiter[i][j][k] = temp; 
                /*__________________________________
                *   DEBUGGINGS
                *___________________________________*/
#if switchDebug_Advect_gradient_limiter
                test[i][j][k]   =  grad_limiter[i][j][k];
#endif 
#endif
            /*__________________________________
            * CFDLIB limiter
            *___________________________________*/
#if(LIMIT_GRADIENT_FLAG == 2)
                q_CC_ijk=  q_CC[m][i][j][k];
                frac    = DMAX( SMALL_NUM, (q_VRTX_MAX[i][j][k] - q_CC_ijk) );
                alf1    = ( q_CC_max[i][j][k] - DMIN( q_CC_ijk, q_CC_max[i][j][k]) )/frac;
                
                
                frac    = DMAX( SMALL_NUM, (q_CC_ijk - q_VRTX_MIN[i][j][k]) );
                alf2    = ( DMAX( q_CC_ijk, q_CC_min[i][j][k] ) - q_CC_min[i][j][k])/frac;
 
                alf      = DMIN( 1.0, alf1);
                alf      = DMIN( alf, alf2 );
                grad_limiter[i][j][k]  = DMAX( 0.0, alf ); 
                
                /*__________________________________
                *   DEBUGGINGS
                *___________________________________*/
#if switchDebug_Advect_gradient_limiter
                test[i][j][k]   =  grad_limiter[i][j][k];
#endif
                /*__________________________________
                *
                *___________________________________*/
#endif                        
            }
        }
    }
/*STOP_DOC*/
/*______________________________________________________________________
*   Section for Plotting
*_______________________________________________________________________*/ 
#if switchDebug_Advect_gradient_limiter
    #define switchInclude_Advect_gradient_limiter 1
    #include "debugcode.i"
    #undef switchInclude_Advect_gradient_limiter
    
   free_darray_3d( test, 0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
   free_darray_3d( test2,0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);    

#endif 

/*______________________________________________________________________
*  Free the allocated local memory
*_______________________________________________________________________*/
    free_darray_3d( q_VRTX_MAX, 0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    free_darray_3d( q_VRTX_MIN, 0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    free_darray_3d( q_CC_max,   0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    free_darray_3d( q_CC_min,   0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
/*__________________________________
*   Quite fullwarn remarks in a way that
*   is compiler independent
*___________________________________*/
    alf1 = alf1;            alf2 = alf2;        alf = alf;
    temp = temp;            frac = frac;        
    grad_lim_max = grad_lim_max;                grad_lim_min = grad_lim_min;            
    q_CC_ijk = q_CC_ijk;          
}
/*STOP_DOC*/



/* 
 ======================================================================*/
#include <math.h>
#include <assert.h>
#include "nrutil+.h"
#include "functionDeclare.h"
#include "switches.h"
#include "parameters.h"
#include "macros.h"
/* ---------------------------------------------------------------------
 Function:  find_q_CC_max_min--ADVECTION: Step 6.?, Compute the max and min. calculates the max and min value of q_CC from the four surrounding cells. 
 Filename:  advect_grad_limiter.c
 
 Purpose:
   This routine calculates the max and min value of q_CC from the four
   surrounding cells.   

            
 Steps for each cell:
 --------------------
 1) Double check the inputs and determine what walls of the computaional
 domain should be included in the calculation.
 2) Compute q_cc_max and q_cc_min for all cells inside of the computational
 domain.
 3) Calculate q_CC_max and q_CC_man in a single ghostcell layer surrounding
 the computational domain
 4) Now compute the corner ghostcells    
   
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       11/17/99    

 ---------------------------------------------------------------------  */
void find_q_CC_max_min(    
        int     xLoLimit,               /* x-array lower limit              */
        int     yLoLimit,               /* y-array lower limit              */
        int     zLoLimit,               /* z-array lower limit              */
        int     xHiLimit,               /* x-array upper limit              */
        int     yHiLimit,               /* y-array upper limit              */
        int     zHiLimit,               /* z-array upper limit              */
        double  ****q_CC,               /* cell center data                 (INPUT) */
        double  ***q_CC_max,            /* max. value of q_CC               (OUTPUT)*/
        double  ***q_CC_min,            /* min. value of q_CC               (OUTPUT)*/
        int     m                )      /* material                         */
  
{
        int     i, j, k,                /* cell indices                     */
                wall,                   /* wall indices                     */
                wallLo, wallHi, 
                xLo, xHi,
                yLo, yHi,
                zLo, zHi;
           
        double
                q_CC_max_temp,          /* q_CC max and min temp           */
                q_CC_min_temp;
/*__________________________________
*   Plotting variables
*___________________________________*/ 
#if switchDebug_find_q_CC_max_min
    #include plot_declare_vars.h"   
#endif     
/*__________________________________
*   Step 1)
* double check inputs
*___________________________________*/
    assert ( xLoLimit >= 0 && xHiLimit < X_MAX_LIM);
    assert ( yLoLimit >= 0 && yHiLimit < Y_MAX_LIM);
    assert ( zLoLimit >= 0 && zHiLimit < Z_MAX_LIM);
    assert ( m <= N_MATERIAL); 
/*__________________________________
*  Determine what walls are used in the 
*   computational domain.
*___________________________________*/
#if (N_DIMENSIONS == 1)  
        wallLo = LEFT;  wallHi = RIGHT;
#endif

#if (N_DIMENSIONS == 2) 
        wallLo = TOP;   wallHi = LEFT;
#endif
#if (N_DIMENSIONS == 3) 
        wallLo = TOP;   wallHi = BACK;
#endif    
       
/*START_DOC*/  
/*______________________________________________________________________
* step 2)  Find the max and min in the interior
    ---------------------------------------------
    |   |   |   |   |   |   |   |   |   |   |   |   -- top_GC       
    ---------------------------------------------                   
    |   | + | + | + | + | + | + | + | + | + |   |   -- yHiLimit     
    ---------------------------------------------                   
    |   | + | + | + | + | + | o | + | + | + |   |                   
    ---------------------------------------------                   
    |   | + | + | + | + | o | X | o | + | + |   |                   
    ---------------------------------------------                    
    |   | + | + | + | + | + | o | + | + | + |   |   -- yLoLimit     
    ---------------------------------------------                   
    |   |   |   |   |   |   |   |   |   |   |   |   -- bottom_GC    
    ---------------------------------------------                   
      |   | xLoLimit             xHiLimit |   |                     
      |                                       |                     
      left_GC                               right_GC                                                                                
      x = q_CC_max and q_CC_min
      o = data needed: q_CC  
*_______________________________________________________________________*/
    xLo = (xLoLimit);
    xHi = (xHiLimit);
    yLo = (yLoLimit);
    yHi = (yHiLimit);
    zLo = (zLoLimit);
    zHi = (zHiLimit);
                                  
    for ( i = xLo; i <= xHi; i++)
    {
        for ( j = yLo; j <= yHi; j++)
        {
            for ( k = zLo; k <= zHi; k++)
            { 
                q_CC_max_temp       =  DMAX(q_CC[m][i+1][j][k], q_CC[m][i-1][j][k]);
                q_CC_max_temp       =  DMAX(q_CC[m][i][j+1][k], q_CC_max_temp);         /* top  */
                q_CC_max_temp       =  DMAX(q_CC[m][i][j-1][k], q_CC_max_temp);         /*bottom*/
                /* q_CC_max_temp    =  DMAX(q_CC[m][i][j][k+1], q_CC_max_temp); */      /*front */
                /* q_CC_max_temp    =  DMAX(q_CC[m][i][j][k-1], q_CC_max_temp); */      /* back */
                q_CC_max[i][j][k]   = q_CC_max_temp;
                
                q_CC_min_temp       =  DMIN(q_CC[m][i+1][j][k], q_CC[m][i-1][j][k]);
                q_CC_min_temp       =  DMIN(q_CC[m][i][j+1][k], q_CC_min_temp);         /* top  */
                q_CC_min_temp       =  DMIN(q_CC[m][i][j-1][k], q_CC_min_temp);          /*bottom*/
                /* q_CC_min_temp    =  DMIN(q_CC[m][i][j][k+1], q_CC_min_temp); */      /*front */
                /* q_CC_min_temp    =  DMIN(q_CC[m][i][j][k-1], q_CC_min_temp); */      /* back */
                q_CC_min[i][j][k]   = q_CC_min_temp;
       
            }
        }
    }
/*______________________________________________________________________
*   Step 3)
*   Find q_CC_max and q_CC_min in the single layer of ghost cells  This only 
*   looks at the three nearest  neighbors
*
    _____________________________________________
    | o |   |   |   |   |   |   |   | o | x | o |   -- top_GC       
    ---------------------------------------------                   
    | x | o | + | + | + | + | + | + | + | o |   |   -- yHiLimit     
    ---------------------------------------------                   
    | o | + | + | + | + | + | + | + | + | + |   |                   
    ---------------------------------------------                   
    |   | + | + | + | + | + | + | + | + | + | o |                   
    ---------------------------------------------                    
    |   | o | + | + | + | + | + | + | + | o | x |   -- yLoLimit     
    ---------------------------------------------                   
    | o | x | o |   |   |   |   |   |   |   | o |   -- bottom_GC    
    ---------------------------------------------                   
      |   | xLoLimit             xHiLimit |   |                     
      |                                       |                     
      left_GC                               right_GC
      
      x = q_CC_min and q_CC_max
      o = data needed: q_CC 
*_______________________________________________________________________*/
    for( wall = wallLo; wall <= wallHi; wall ++)
    {
        /*__________________________________
        *  Find the looping indices associated
        *   with a particular wall
        *___________________________________*/
         find_loop_index_limits_at_domain_edges(                
                    xLoLimit,                  yLoLimit,                   zLoLimit,
                    xHiLimit,                  yHiLimit,                   zHiLimit,
                    &xLo,                      &yLo,                       &zLo,
                    &xHi,                      &yHi,                       &zHi,
                    wall    );
    
        for ( i = xLo; i <= xHi; i++ )
        {
            for ( j = yLo; j <= yHi; j++ )
            { 
                for ( k = zLo; k <= zHi; k++ )
                {        
                
                   if ( wall == LEFT )
                   {
                       q_CC_max_temp        =  DMAX(q_CC[m][i][j+1][k], q_CC[m][i][j-1][k]);      
                       q_CC_max_temp        =  DMAX(q_CC[m][i+1][j][k], q_CC_max_temp);           
                       q_CC_max[i][j][k]    = q_CC_max_temp;

                       q_CC_min_temp        =  DMIN(q_CC[m][i][j+1][k], q_CC[m][i][j-1][k]);
                       q_CC_min_temp        =  DMIN(q_CC[m][i+1][j][k], q_CC_min_temp);
                       q_CC_min[i][j][k]    = q_CC_min_temp;          
                  }
                                
                   if ( wall == RIGHT )
                   {
                       q_CC_max_temp        =  DMAX(q_CC[m][i][j+1][k], q_CC[m][i][j-1][k]);      
                       q_CC_max_temp        =  DMAX(q_CC[m][i-1][j][k], q_CC_max_temp);           
                       q_CC_max[i][j][k]    = q_CC_max_temp;

                       q_CC_min_temp        =  DMIN(q_CC[m][i][j+1][k], q_CC[m][i][j-1][k]);
                       q_CC_min_temp        =  DMIN(q_CC[m][i-1][j][k], q_CC_min_temp);
                       q_CC_min[i][j][k]    = q_CC_min_temp;                   
                   
                     }
                   if ( wall == TOP )
                   {
                       q_CC_max_temp        =  DMAX(q_CC[m][i+1][j][k], q_CC[m][i-1][j][k]);      
                       q_CC_max_temp        =  DMAX(q_CC[m][i][j-1][k], q_CC_max_temp);           
                       q_CC_max[i][j][k]    = q_CC_max_temp;

                       q_CC_min_temp        =  DMIN(q_CC[m][i+1][j][k], q_CC[m][i-1][j][k]);
                       q_CC_min_temp        =  DMIN(q_CC[m][i][j-1][k], q_CC_min_temp);
                       q_CC_min[i][j][k]    = q_CC_min_temp;                    
                   
                     }
                    if ( wall == BOTTOM )
                   {
                       q_CC_max_temp        =  DMAX(q_CC[m][i+1][j][k], q_CC[m][i-1][j][k]);      
                       q_CC_max_temp        =  DMAX(q_CC[m][i][j+1][k], q_CC_max_temp);           
                       q_CC_max[i][j][k]    = q_CC_max_temp;

                       q_CC_min_temp        =  DMIN(q_CC[m][i+1][j][k], q_CC[m][i-1][j][k]);
                       q_CC_min_temp        =  DMIN(q_CC[m][i][j+1][k], q_CC_min_temp);
                       q_CC_min[i][j][k]    = q_CC_min_temp;
                   }

                }
            }
        }
    }
/*______________________________________________________________________
*   Step 4)
*   Find q_CC_max and q_CC_min in the corner ghostcells  This only 
*   looks at the three nearest neighbors.

    ---------------------------------------------
    | x | o |   |   |   |   |   |   |   | o | x |   -- top_GC       
    ---------------------------------------------                  
    | o | o | + | + | + | + | + | + | + | o | o |   -- yHiLimit     
    ---------------------------------------------                   
    |   | + | + | + | + | + | + | + | + | + |   |                   
    ---------------------------------------------                   
    |   | + | + | + | + | + | + | + | + | + |   |                   
    ---------------------------------------------                    
    | o | o | + | + | + | + | + | + | + | o | o |   -- yLoLimit     
    ---------------------------------------------                   
    | x | o |   |   |   |   |   |   |   | o | x |   -- bottom_GC    
    ---------------------------------------------                   
      |   | xLoLimit             xHiLimit |   |                     
      |                                       |                     
      left_GC                               right_GC

      x = q_CC_max, q_CC_min.
      o = data needed: q_CC 
*_______________________________________________________________________*/
/*   Upper Left ghostcell corner 
*___________________________________*/
    i = GC_LO(xLoLimit);
    j = GC_HI(yHiLimit);
    k = zLoLimit;
    q_CC_max_temp       =  DMAX(q_CC[m][i][j-1][k], q_CC[m][i+1][j-1][k]);      
    q_CC_max_temp       =  DMAX(q_CC[m][i+1][j][k], q_CC_max_temp);           
    q_CC_max[i][j][k]   = q_CC_max_temp;

    q_CC_min_temp       =  DMIN(q_CC[m][i][j-1][k], q_CC[m][i+1][j-1][k]);
    q_CC_min_temp       =  DMIN(q_CC[m][i+1][j][k], q_CC_min_temp);
    q_CC_min[i][j][k]   = q_CC_min_temp;    

/*__________________________________
*   Upper right ghostcell corner
*___________________________________*/
    i = GC_HI(xHiLimit);
    j = GC_HI(yHiLimit);
    k = zLoLimit;
    q_CC_max_temp       =  DMAX(q_CC[m][i][j-1][k], q_CC[m][i-1][j-1][k]);      
    q_CC_max_temp       =  DMAX(q_CC[m][i-1][j][k], q_CC_max_temp);           
    q_CC_max[i][j][k]   = q_CC_max_temp;

    q_CC_min_temp       =  DMIN(q_CC[m][i][j-1][k], q_CC[m][i-1][j-1][k]);
    q_CC_min_temp       =  DMIN(q_CC[m][i-1][j][k], q_CC_min_temp);
    q_CC_min[i][j][k]   = q_CC_min_temp;    

/*__________________________________
*   Lower right ghostcell corner
*___________________________________*/
    i = GC_HI(xHiLimit);
    j = GC_LO(yLoLimit);
    k = zLoLimit;
    q_CC_max_temp       =  DMAX(q_CC[m][i][j+1][k], q_CC[m][i-1][j+1][k]);      
    q_CC_max_temp       =  DMAX(q_CC[m][i-1][j][k], q_CC_max_temp);           
    q_CC_max[i][j][k]   = q_CC_max_temp;

    q_CC_min_temp       =  DMIN(q_CC[m][i][j+1][k], q_CC[m][i-1][j+1][k]);
    q_CC_min_temp       =  DMIN(q_CC[m][i-1][j][k], q_CC_min_temp);
    q_CC_min[i][j][k]   = q_CC_min_temp;

/*__________________________________
*   Lower left ghostcell corner
*___________________________________*/
    i = GC_LO(xLoLimit);
    j = GC_LO(yLoLimit);
    k = zLoLimit;
    q_CC_max_temp       =  DMAX(q_CC[m][i][j+1][k], q_CC[m][i+1][j+1][k]);      
    q_CC_max_temp       =  DMAX(q_CC[m][i+1][j][k], q_CC_max_temp);           
    q_CC_max[i][j][k]   = q_CC_max_temp;

    q_CC_min_temp       =  DMIN(q_CC[m][i][j+1][k], q_CC[m][i+1][j+1][k]);
    q_CC_min_temp       =  DMIN(q_CC[m][i+1][j][k], q_CC_min_temp);
    q_CC_min[i][j][k]   = q_CC_min_temp;   
/*STOP_DOC*/    
/*______________________________________________________________________
*   Section for Plotting
*_______________________________________________________________________*/    

#if switchDebug_find_q_CC_max_min
    #define switchInclude_find_q_CC_max_min 1
    #include debugcode.i"
    #undef switchInclude_find_q_CC_max_min
#endif  

}
/*STOP_DOC*/
