/* 
 ======================================================================*/
#include <sys/types.h>
#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "nrutil+.h"
#include "functionDeclare.h"
#include "switches.h"
#include "parameters.h"
#include "macros.h"
/* ---------------------------------------------------------------------
 Function:  find_q_vertex_max_min--Advection: Step 6.?, Find max. and min. of (q) at vertices 
 Filename:  q_vertex.c
 
 Purpose:
   This routine calculates the max and min value of q at the vertices of each cell.
   
 References:
    "Compatible Fluxes for van Leer Advection" W.B VanderHeyden and 
    B.A. Kashiwa, Journal of Computational Physics, 146, 1-28, (1998) and 
    Uintah-ICE CFD Multidimensional Compatible Advection Operator

Computational Domain:
    The max and min. values of (q) at the vertices is a cell-centered quantity
    that is computed throughout the computational domain.  
    
Ghostcell data dependency:
    In order to compute the vertex values data from the ghost cells is used.
            
 Steps
 --------------------
 1) Declare local memory for the vertex data and equate the common vertices
 2) Interpolate the values of q out to the vertices
 3) Find the max. and min. of the vertex data      
   
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
void find_q_vertex_max_min(    
        int     xLoLimit,               /* x-array lower limit              */
        int     yLoLimit,               /* y-array lower limit              */
        int     zLoLimit,               /* z-array lower limit              */
        int     xHiLimit,               /* x-array upper limit              */
        int     yHiLimit,               /* y-array upper limit              */
        int     zHiLimit,               /* z-array upper limit              */
        double  delX,                   /* Cell width                       */
        double  delY,                   /* Cell Width in the y dir          */
        double  delZ,                   /* Cell width in the z dir          */
        double  ****q_CC,               /* cell center data                 (INPUT) */
        double  ***q_VRTX_MAX,          /* max. value of the vertices       (OUTPUT)*/
        double  ***q_VRTX_MIN,          /* min. value of the vertices       (OUTPUT)*/
        int     m                )      /* material                         */
  
{
        int i, j, k, v,                 /* cell indices                     */ 
            n_vertices,                 /* number of cell vertices          */
            xLo, xHi,
            yLo, yHi,
            zLo, zHi;
           
        double
            *****q_VRTX,                /* q at the vertex (pointer)        */
                                        /* (i, j, k, vertice)               */
            q_vrtex_max,                /* vertex max and min               */
             q_vrtex_min;
/*__________________________________
*   Plotting variables
*___________________________________*/ 
#if switchDebug_find_q_vertex_max
    #include plot_declare_vars.h"   
#endif 
    
/*__________________________________
* double check inputs
*___________________________________*/
    assert ( xLoLimit >= 0 && xHiLimit < X_MAX_LIM);
    assert ( yLoLimit >= 0 && yHiLimit < Y_MAX_LIM);
    assert ( zLoLimit >= 0 && zHiLimit < Z_MAX_LIM);
    assert ( m <= N_MATERIAL);
    assert ( delX > 0.0 );
    assert ( delY > 0.0 );
#if (N_DIMENSIONS == 3)
    assert ( delZ > 0.0 );
#endif
    n_vertices = 4 + IF_3D(4);          /* calculate the number of vertices */
    
/*START_DOC*/  
/*______________________________________________________________________
*   Step 1)
*  - Allocate Memory
*  - Define upper and lower looping indices
*  - Equate the pointer address of the vertices.  For example
*    q_VRTX[1][1][1][1] = q_VRTX[0][1][1][2]
*_______________________________________________________________________*/
    q_VRTX   = darray_5d(0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM, 1, N_CELL_VERTICES,1,1);
    
    /*__________________________________
    * 12/15/99: keep around until you resolve
    * which find_q_vertex function to use
    *___________________________________*/   
/* 
xLo = 1;
    xHi = X_MAX_LIM-1;
    yLo = 1;
    yHi = Y_MAX_LIM-1;
    zLo = 1;
    zHi = Z_MAX_LIM -1;
   
    for ( i = xLo; i <= xHi; i++)
    {
        for ( j = yLo; j <= yHi; j++)
        {
            for ( k = zLo; k <= zHi; k++)
            { 
                 q_VRTX[i-1][j][k][2]        = q_VRTX[i][j][k][1];
                q_VRTX[i-1][j-1][k][3]      = q_VRTX[i][j][k][1];
                q_VRTX[i][j-1][k][4]        = q_VRTX[i][j][k][1]; */
                /* *q_VRTX[i][j][k][1]         = 0.0;  */
                /*__________________________________
                *  I'll need this for 3D
                *___________________________________*/
               /*  q_VRTX[i][j][k-1][5]        = q_VRTX[i][j][k][1];
                q_VRTX[i-1][j][k-1][6]      = q_VRTX[i][j][k][1];
                q_VRTX[i][j-1][k-1][7]      = q_VRTX[i][j][k][1];
                q_VRTX[i-1][j-1][k-1][8]    = q_VRTX[i][j][k][1];
            }
        }
    }
*/
/*______________________________________________________________________
* step 2)  Interpolate the values of q out to the vertices
*_______________________________________________________________________*/

    find_q_vertex(          xLoLimit,       yLoLimit,       zLoLimit,
                            xHiLimit,       yHiLimit,       zHiLimit,
                            delX,           delY,           delZ,
                            q_CC,           q_VRTX,        m);
/*__________________________________
* step 3) Find the max and min of the
*   vertex data
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
                q_vrtex_max = *q_VRTX[i][j][k][1];
                q_vrtex_min = *q_VRTX[i][j][k][1];
                for (v = 2; v <= n_vertices ; v++)
                {
                   q_vrtex_max = DMAX(q_vrtex_max, *q_VRTX[i][j][k][v]);
                   q_vrtex_min = DMIN(q_vrtex_min, *q_VRTX[i][j][k][v]);
                    
                }
                q_VRTX_MAX[i][j][k] = q_vrtex_max;
                q_VRTX_MIN[i][j][k] = q_vrtex_min;
                      
            }
        }
    }
    
/*______________________________________________________________________
*   Section for Plotting
*_______________________________________________________________________*/    

#if switchDebug_find_q_vertex_max
    #define switchInclude_find_q_vertex_max 1
    #include debugcode.i"
    #undef switchInclude_find_q_vertex_max
#endif  
/*______________________________________________________________________
*   Deallocate memory
*_______________________________________________________________________*/
   free_darray_5d( q_VRTX,0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM, 1, N_CELL_VERTICES, 1,1);

}
/*STOP_DOC*/

#if 1
/* 
 ======================================================================*/
#include <sys/types.h>
#include <math.h>
#include <assert.h>
#include "functionDeclare.h"
#include "switches.h"
#include "parameters.h"
#include "macros.h"
/*---------------------------------------------------------------------
 Function:  find_q_vertex--Advection: Step 6.?, Compute (q) at vertices of each cell
 Filename:  advect_q_vertex.c
 Purpose:
   This routine calculates the values of q at the vertices of each cell.
   
 References:
    "Compatible Fluxes for van Leer Advection" W.B VanderHeyden and 
    B.A. Kashiwa, Journal of Computational Physics, 146, 1-28, (1998) and 
    Uintah-ICE CFD Multidimensional Compatible Advection Operator
            
 Steps for each cell:
 --------------------     
            1) Calculate the gradients of q in the x, y and z dir for 
            all of the cells in the domain and one ghostcell 
            layer surrounding the domain.
            2) For each vertice calculate the relative distance between the 
               the cell centroid and the vertex ()_term
            3) determing q_VRTX at each of the vertices   
   
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
 
 Data Dependencies of the corner ghostcells
    x---x-----------------------------------x---x
    | o | o | o |   |   |   |   |   | o | o | o |   -- top_GC       
    x---x-----------------------------------x---x                  
    | o | + | + | + | + | + | + | + | + | + | o |   -- yHiLimit     
    ---------------------------------------------                   
    | o | + | + | + | + | + | + | + | + | + | o |                   
    ---------------------------------------------                   
    | o | + | + | + | + | + | + | + | + | + | o |                   
    ---------------------------------------------                    
    | o | + | + | + | + | + | + | + | + | + | o |   -- yLoLimit     
    x---x-----------------------------------x---x                   
    | o | o | o |   |   |   |   |   | o | o | o |   -- bottom_GC    
    x---x-----------------------------------x---x                   
      |   | xLoLimit             xHiLimit |   |                     
      |                                       |                     
      left_GC                               right_GC

  x = q_vertex in the corner cells
  o = data needed: q_CC in function grad_q only for the corner cells
 
 ---------------------------------------------------------------------  */
void find_q_vertex( 
        int     xLoLimit,               /* x-array lower limit              */
        int     yLoLimit,               /* y-array lower limit              */
        int     zLoLimit,               /* z-array lower limit              */
        int     xHiLimit,               /* x-array upper limit              */
        int     yHiLimit,               /* y-array upper limit              */
        int     zHiLimit,               /* z-array upper limit              */
        double  delX,                   /* Cell width                       */
        double  delY,                   /* Cell Width in the y dir          */
        double  delZ,                   /* Cell width in the z dir          */
        double  ****q_CC,               /* q-cell-centered                  */
        double  *****q_VRTX,            /* q at the vertex                  */
                                        /* (i, j, k, vertice)               */
        int     m         )             /* material                         */
  
{
    int     i, j, k,                    /* cell face locators               */ 
            xLo, xHi,
            yLo, yHi,
            zLo, zHi;
    double 
            ***grad_q_X,                /* gradients of q in the x,y,z dir  */
                                        /* temporary variables              */
            ***grad_q_Y,
            ***grad_q_Z,
                                        /* distance between the cell centroid*/
                                        /* and the vertex in x, y, and zdir */
            q_vrtx1,                    /* temporary variables to used to   */
            q_vrtx2,                    /* help in the debugging process    */
            q_vrtx3,
            q_vrtx4,
            x_term,                                           
            y_term,
            z_term;           
       
/*______________________________________________________________________
*  Allocate Memory and initialize the arrays 
*_______________________________________________________________________*/
    grad_q_X    = darray_3d(0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    grad_q_Y    = darray_3d(0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    grad_q_Z    = darray_3d(0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM); 
       
    zero_arrays_3d(
             xLoLimit,      yLoLimit,       zLoLimit,             
             xHiLimit,      yHiLimit,       zHiLimit,
             3,
             grad_q_X,      grad_q_Y,       grad_q_Z);    
/*__________________________________
* double check inputs
*___________________________________*/
    assert ( xLoLimit >= 0 && xHiLimit < X_MAX_LIM);
    assert ( yLoLimit >= 0 && yHiLimit < Y_MAX_LIM);
    assert ( zLoLimit >= 0 && zHiLimit < Z_MAX_LIM);
    assert ( m <= N_MATERIAL);
    assert ( delX > 0.0 );
    assert ( delY > 0.0 );
#if (N_DIMENSIONS == 3)
    assert ( delZ > 0.0 );
#endif   
/*______________________________________________________________________
* 
*_______________________________________________________________________*/
    /*__________________________________
    * Step 1 
    * calculate the gradients of q
    * in the x, y, and z direction for all
    * cells in the domain and one ghostcell
    * layer surrounding the domain
    *___________________________________*/
    grad_q( 
             xLoLimit,       yLoLimit,       zLoLimit,
             xHiLimit,       yHiLimit,       zHiLimit,
             delX,           delY,           delZ,
             q_CC,           
             grad_q_X,      grad_q_Y,        grad_q_Z,
             m);
    /*__________________________________
    *  Now calculate the vertex values of q
    * for all cells in the domain and one
    * ghostcell layer surrounding the domain
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
            
            /*__________________________________
            * vertex 1 
            *___________________________________*/
            x_term              = grad_q_X[i][j][k] * (-delX/2.0);
            y_term              = grad_q_Y[i][j][k] * (-delY/2.0);
            z_term              = IF_3D(0.0);        /* temporary 6.15.99*/
            q_vrtx1             = q_CC[m][i][j][k] + x_term + y_term + z_term;
            *q_VRTX[i][j][k][1]  = q_vrtx1;  
            
            /*__________________________________
            * vertex 2
            *___________________________________*/
            x_term              = grad_q_X[i][j][k] * (delX - (delX/2.0));
            y_term              = grad_q_Y[i][j][k] * (-delY/2.0);
            z_term              = IF_3D(0.0);       /* temporary 6.15.99*/
            q_vrtx2             = q_CC[m][i][j][k] + x_term + y_term + z_term;
            *q_VRTX[i][j][k][2]  = q_vrtx2;             
           

            /*__________________________________
            * vertex 3
            *___________________________________*/
            x_term              = grad_q_X[i][j][k] * (delX - (delX/2.0));
            y_term              = grad_q_Y[i][j][k] * (delY - (delY/2.0));
            z_term              = IF_3D(0.0);    /* temporary 6.15.99*/
            q_vrtx3             = q_CC[m][i][j][k] + x_term + y_term + z_term;
            *q_VRTX[i][j][k][3]  = q_vrtx3;  
            
            /*__________________________________
            * vertex 4
            *___________________________________*/
            x_term              = grad_q_X[i][j][k] * (-(delX/2.0));
            y_term              = grad_q_Y[i][j][k] * (delY - (delY/2.0));
            z_term              = IF_3D(0.0);     /* temporary 6.15.99*/
            q_vrtx4             = q_CC[m][i][j][k] + x_term + y_term + z_term;
            *q_VRTX[i][j][k][4]  = q_vrtx4;             
                      
            }
        }
    }
/*______________________________________________________________________
*   Deallocate memory
*_______________________________________________________________________*/
   free_darray_3d( grad_q_X, 0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
   free_darray_3d( grad_q_Y, 0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
   free_darray_3d( grad_q_Z, 0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
}
/*STOP_DOC*/
#endif


/*______________________________________________________________________
*   First pass at averaging the 4 neighboring cell-center values of q
*   to determine the vertex values
*_______________________________________________________________________*/
#if 0
/* 
 ======================================================================*/
#include <sys/types.h>
#include <math.h>
#include <assert.h>
#include "functionDeclare.h"
#include "switches.h"
#include "parameters.h"
#include "macros.h"
/*---------------------------------------------------------------------
 Function:  find_q_vertex--Advection: Step 6.?, Compute (q) at vertices of each cell
 Filename:  advect_q_vertex.c
 
 Purpose:
   This routine calculates the values of q at the vertices of each cell.
   
 References:
    "Compatible Fluxes for van Leer Advection" W.B VanderHeyden and 
    B.A. Kashiwa, Journal of Computational Physics, 146, 1-28, (1998) and 
    Uintah-ICE CFD Multidimensional Compatible Advection Operator
 
 Governing Eq:
    for vertex 1
    q_VRTX[i,j,k,1] = 0.25*(  q_CC[i,j,k,m] + q_CC[i-1,j,k,m] 
                            + q_CC[i-1,j-1,k,m] + q[i,j-1,k,m]) 
 Implementation Notes:
    We compute q_VRTX by first 
        q_VRTX[i,j,k,2] = q_VRTX[i,j,k,2] + (q_CC[i,j,k,m] + q_CC[i+1,j,k,m]) /2.0
        q_VRTX[i,j,k,3] = q_VRTX[i,j,k,3] + (q_CC[i,j,k,m] + q_CC[i+1,j,k,m]) /2.0 
        
    Since the initial values of q_VRTX[i,j,k,2] and q_VRTX[i,j,k,3] = 0.0 
    
                                       
 Steps for each cell:
 --------------------     
    1)  Compute the vertex 2 and 3 in all the cell except the left and
        right ghostcell layers.  
    2)  Compute vertices 3 and 4 in the left and right ghost cell layers
    3)  Take care of the lower left and right ghostcell corners
   
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       12/13/99    

 
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
void find_q_vertex( 
        int     xLoLimit,               /* x-array lower limit              */
        int     yLoLimit,               /* y-array lower limit              */
        int     zLoLimit,               /* z-array lower limit              */
        int     xHiLimit,               /* x-array upper limit              */
        int     yHiLimit,               /* y-array upper limit              */
        int     zHiLimit,               /* z-array upper limit              */
        double  delX,                   /* Cell width                       */
        double  delY,                   /* Cell Width in the y dir          */
        double  delZ,                   /* Cell width in the z dir          */
        double  ****q_CC,               /* q-cell-centered                  */
        double  *****q_VRTX,            /* q at the vertex                  */
                                        /* (i, j, k, vertice)               */
        int     m         )             /* material                         */
  
{
    int     i, j, k,                    /* cell face locators               */ 
            xLo, xHi,
            yLo, yHi,
            zLo, zHi;
    double                                   
            q_vrtx_temp;          
/*__________________________________
* double check inputs
*___________________________________*/
    assert ( xLoLimit >= 0 && xHiLimit < X_MAX_LIM);
    assert ( yLoLimit >= 0 && yHiLimit < Y_MAX_LIM);
    assert ( zLoLimit >= 0 && zHiLimit < Z_MAX_LIM);
    assert ( m <= N_MATERIAL);
    assert ( delX > 0.0 );
    assert ( delY > 0.0 );
#if (N_DIMENSIONS == 3)
    assert ( delZ > 0.0 );
#endif   
/*______________________________________________________________________
*   Step 1)
*   Compute vertices 2 and 3 in all the cells except the left and right
*   ghostcells walls.
*_______________________________________________________________________*/
    /*__________________________________
    *  Now calculate the vertex values for
    *   vertex 2 and 3 
    *   x = Half Updated vertex values
    *   o = Fully updated vertex values
    *   ----X---X---X---X---X---X----
    *   |   |   |   |   |   |   |   |   -- top_GC       
    *   |---o---o---o---o---o---o---|                  
    *   |   | + | + | + | + | + |   |   -- yHiLimit     
    *   |---o---o---o---o---o---o---|                   
    *   |   | + | + | + | + | + |   |                   
    *   |---o---o---o---o---o---o---|                   
    *   |   | + | + | + | + | + |   |                   
    *   |---o---o---o---o---o---o---|                    
    *   |   | + | + | + | + | + |   |   -- yLoLimit     
    *   |---o---o---o---o---o---o---|                   
    *   |   |   |   |   |   |   |   |   -- bottom_GC    
    *   ----X---X---X---X---X---X----                   
    *___________________________________*/
    xLo = GC_LO(xLoLimit);
    xHi = (xHiLimit);
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
                q_vrtx_temp          =  (q_CC[m][i][j][k] + q_CC[m][i+1][j][k]) /2.0;
                /*__________________________________
                * vertex 2
                *___________________________________*/
                *q_VRTX[i][j][k][2]  = *q_VRTX[i][j][k][2] + q_vrtx_temp; 

                /*__________________________________
                * vertex 3
                *___________________________________*/
                *q_VRTX[i][j][k][3]  = *q_VRTX[i][j][k][3] + q_vrtx_temp;
            }
        }
    }
/*__________________________________
*   Step 2)
*   Now loop through the left and right
*   layer of ghostcells minus the corner cells
*   LEFT and Right walls
*___________________________________*/
    xLo = GC_LO(xLoLimit);
    xHi = GC_HI(xHiLimit);
    yLo = GC_LO(yLoLimit);
    yHi = GC_HI(yHiLimit);
    zLo = GC_LO(zLoLimit);
    zHi = GC_HI(zHiLimit);
    for ( j = yLo; j <= yHi; j++)
    {
        for ( k = zLo; k <= zHi; k++)
        {
            *q_VRTX[xLo][j][k][4] = (q_CC[m][xLo][j][k] + q_CC[m][xLo][j+1][k]) /2.0;
            *q_VRTX[xHi][j][k][3] = (q_CC[m][xHi][j][k] + q_CC[m][xHi][j+1][k]) /2.0; 
        }
    }
    
/*__________________________________
*   Step 3)
*   Compute vertex 1 in the lower left
*   corner ghostcell and 
*   vertex 2 in the lower right ghostcell
*___________________________________*/
    for ( k = zLo; k <= zHi; k++)
    {
        *q_VRTX[xLo][yLo][k][4] =  q_CC[m][xLo][yLo][k];
        *q_VRTX[xHi][yLo][k][2] =  q_CC[m][xLo][yHi][k];
    } 

}
/*STOP_DOC*/
#endif
