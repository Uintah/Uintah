/* 
 ======================================================================*/
#include <assert.h>
#include <time.h>
#include <sys/types.h>
#include <math.h>
#include "functionDeclare.h"
#include "parameters.h"
#include "switches.h"
#include "macros.h"

/*
 Function:  calc_flux_or_primitive_vars--MISC: calculates the flux or primitive variables for the entire field including the ghosts cells.
 Filename: flux_or_primitive.c
    
 Purpose:
   This function calculates either the cell-centered,(x,y,x) 
   flux variables (mu), (mv), (mw), (me) or the primitive variables
   (u), (v), (w), (T) depending on the flag
Note:
    This function calculates the flux or primitive variables for the 
    entire field including the ghosts cells.   
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       09/29/99
       
           
Index convention:
                    ****Variable  (i,j,k,material)
                    ***Variable (i,j,k) 
 ---------------------------------------------------------------------  */
void calc_flux_or_primitive_vars(    
        int     flag,                   /* = -1 calc. flux = 1 primitive    */              
        int     xLoLimit,               /* x-array lower limit              */
        int     yLoLimit,               /* y-array lower limit              */
        int     zLoLimit,               /* z-array lower limit              */
        int     xHiLimit,               /* x-array upper limit              */
        int     yHiLimit,               /* y-array upper limit              */
        int     zHiLimit,               /* z-array upper limit              */
        double  ****rho_CC,             /* Density                           (INPUT)*/
        double   ***Vol_CC,             /* cell-centered volume              (INPUT)*/
        double  ****uvel_CC,            /* u-face-centered velocity         (IN/OUT)*/ 
        double  ****vvel_CC,            /* v-face-centered velocity,        (IN/OUT)*/
        double  ****wvel_CC,            /* w face-centered velocity         (IN/OUT)*/
        double  ****xmom_CC,            /* x component of momentum          (IN/OUT)*/
        double  ****ymom_CC,            /* y component of momentum          (IN/OUT)*/
        double  ****zmom_CC,            /* z component of momentum          (IN/OUT)*/
        double  ****cv_CC,              /* specific heat                     (INPUT)*/        
        double  ****int_eng_CC,         /* internal energy                  (IN/OUT)*/
        double  ****Temp_CC,            /* Temperature                      (IN/OUT)*/
        int     nMaterials      )   
{
    int     i, j, k,m;                  /*   loop indices  locators         */ 
    double  mass;                              
    
#if sw_calc_flux_or_primitive_vars    
    time_t start,secs;                  /* timing variables                */
    start = time(NULL);
#endif 

/*__________________________________
* Check that the inputs are reasonable
*___________________________________*/
    assert ( xLoLimit > 0 && xHiLimit < X_MAX_LIM);
    assert ( yLoLimit > 0 && yHiLimit < Y_MAX_LIM);
    assert ( zLoLimit > 0 && zHiLimit < Z_MAX_LIM);
    assert (flag == 1 || flag == -1);
    /*START_DOC*/
/*______________________________________________________________________
*   Calculate the flux variables in the x,y,z dirs 
*   You need to include the ghost cells in the calculation or else
*   the advection routine won't work correctly
*_______________________________________________________________________*/
    if ( flag == -1)
    {
        for (m = 1; m <= nMaterials; m++)
        {     
            for ( i = GC_LO(xLoLimit); i <= GC_HI(xHiLimit); i++)     
            {
                for ( j = GC_LO(yLoLimit); j <= GC_HI(yHiLimit); j++)
                {
                    for ( k = GC_LO(zLoLimit); k <= GC_HI(zHiLimit); k++)
                    {  
                        mass                = rho_CC[m][i][j][k] * Vol_CC[i][j][k];     
                        xmom_CC[m][i][j][k] = mass * uvel_CC[m][i][j][k];
                        ymom_CC[m][i][j][k] = mass * vvel_CC[m][i][j][k];
                        zmom_CC[m][i][j][k] = mass * wvel_CC[m][i][j][k];
                        int_eng_CC[m][i][j][k]
                                            = mass * cv_CC[m][i][j][k] * Temp_CC[m][i][j][k];
                    }
                }
            }
        }
    }
/*__________________________________
*   backout vel from primitive
*   Need to fix this
*   be careful of the boundary conditions don't overwrite dirichlet bcs'
*___________________________________*/    
    if ( flag == 1)
    {
        for (m = 1; m <= nMaterials; m++)
        {
            for ( i = xLoLimit; i <= xHiLimit; i++)
            {
                for ( j = yLoLimit; j <= yHiLimit; j++)
                {
                    for ( k = zLoLimit; k <= zHiLimit; k++)
                    {  
                        mass                = rho_CC[m][i][j][k] * Vol_CC[i][j][k];            
                        uvel_CC[m][i][j][k] = xmom_CC[m][i][j][k]/mass;
                        vvel_CC[m][i][j][k] = ymom_CC[m][i][j][k]/mass;
                        wvel_CC[m][i][j][k] = zmom_CC[m][i][j][k]/mass;
                        Temp_CC[m][i][j][k] = int_eng_CC[m][i][j][k]/(mass * cv_CC[m][i][j][k]);
                    }
                }
            }
        }
    }
 
/*______________________________________________________________________
*   DEBUGGING AND STOP WATCH INFORMATION
*_______________________________________________________________________*/ 
#if switchDebug_calc_flux_or_primitive_vars
    for (m = 1; m <= nMaterials; m++)
    {
        printData_4d(   xLoLimit,       yLoLimit,       zLoLimit,
                        xHiLimit,       yHiLimit,       zHiLimit,
                        m,              m,
                        "calc_primitive","xmom_CC",     xmom_CC);

        printData_4d(   xLoLimit,       yLoLimit,       zLoLimit,
                        xHiLimit,       yHiLimit,       zHiLimit,
                        m,              m,
                        "calc_primitive","uvel_CC",     uvel_CC);
    }
                       
#endif 
       
#if sw_calc_flux_or_primitive_vars
    stopwatch("calc_momentum",start);
#endif

}
/*STOP_DOC*/   

