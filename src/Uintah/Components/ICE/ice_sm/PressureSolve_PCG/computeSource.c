#include <math.h>
#include "pcgmg.h"

#include "parameters.h"
#include "functionDeclare.h"
#include "switches.h"
#include "macros.h"
/*---------------------------------------------------------------------  
 Function:  calc_delPress_RHS--PRESS: Computes the RHS of the delPress eq.
 Filename:  computeSource.c
 Purpose:  
    To be filled in
    
 References:
             
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       12/30/99
 ---------------------------------------------------------------------  */
void calc_delPress_RHS(
        int     xLoLimit,                   /* x-array lower limit              */
        int     yLoLimit,                   /* y-array lower limit              */
        int     zLoLimit,                   /* z-array lower limit              */
        int     xHiLimit,                   /* x-array upper limit              */
        int     yHiLimit,                   /* y-array upper limit              */
        int     zHiLimit,                   /* z-array upper limit              */
        double  delX,                       /* distance/cell, xdir              (INPUT) */
        double  delY,                       /* distance/cell, ydir              (INPUT) */
        double  delZ,                       /* distance/cell, zdir              (INPUT) */
            /*------to be treated as pointers---*/
                                            /*______(x,y,z,face, material)______*/
        double  ******uvel_FC,              /* u-face-centered velocity         */
        double  ******vvel_FC,              /* *v-face-centered velocity        */
        double  ******wvel_FC,              /* w face-centered velocity         */
        UserCtx *userctx,
        Vec     *solution,                  
        int     nMaterials)
{
    int           i, j, k, indx, mat;
    int           m, n, N, ierr;
    Scalar        div_vel_FC;               /* divergence of (*)vel_FC          */
/*__________________________________
*   Need by the test code
*___________________________________*/
    Scalar      hx, hy, x, y, v;
/*__________________________________
*   Initialize variables
*___________________________________*/
    mat = nMaterials;
    m   = xHiLimit - xLoLimit + 1;
    n   = yHiLimit - yLoLimit + 1;
    indx= 0;
    N   = m*n; 
/*___________________________________________________________
*   Compute the div_vel_FC terms
*_______________________________________________________________________*/
    k     = 1;
    for ( j = yLoLimit; j <= yHiLimit; j++) 
    {
        for ( i = xLoLimit; i <= xHiLimit; i++) 
        { 
            div_vel_FC =                *uvel_FC[i][j][k][mat][RIGHT] - *uvel_FC[i][j][k][mat][LEFT];
            div_vel_FC = div_vel_FC +   *vvel_FC[i][j][k][mat][TOP]   - *vvel_FC[i][j][k][mat][BOTTOM];
            div_vel_FC = -delX * delY * div_vel_FC;
            VecSetValue( userctx->b, indx, div_vel_FC, INSERT_VALUES );
            indx++;
        }
    }
    
/*__________________________________
*   For testing and debugging
*   Compute the RHS b[] and solution
*___________________________________*/  
#if switchDebug_pcgmg_test
    #define switchInclude_source_test_code 1
    #include "testcode_PressureSolve.i"
    #undef switchInclude_source_test_code
#endif

 /*__________________________________
*   Quite fullwarn remarks in a way that
*   is compiler independent
*___________________________________*/

    delZ        = delZ; 
    zLoLimit    = zLoLimit;
    zHiLimit    = zHiLimit;
    hx          = hx; 
    hy          = hy; 
    x           = x;
    y           = y;
    v           = v;  
    ierr        = ierr;
    N           = N;
    solution[1] = solution[1];
   *wvel_FC[0][0][0][1][1] = *wvel_FC[0][0][0][1][1];      
}
