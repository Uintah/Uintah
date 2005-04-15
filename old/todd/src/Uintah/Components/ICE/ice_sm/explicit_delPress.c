/* 
 ======================================================================*/
#include <math.h>
#include <assert.h>
#include "functionDeclare.h"
#include "switches.h"
#include "parameters.h"
#include "macros.h"
#include "nrutil+.h"
/*
 Function:  explicit_delPress--PRESSURE: Step 2, compute the change in pressure, explicitly
 Filename:  explicit_delPress.c
 Purpose:
   This function calculates the change in pressure explicitly.  Basically it just 
   solves the pressure equation once.
 Note:  Units of delpress are [Pa]
 
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       03/10/00    
 ---------------------------------------------------------------------  */
void explicit_delPress
             (  
        int     xLoLimit,               /* x-array Lower Interior Nodes     */
        int     yLoLimit,               /* y-array Lower Interior Nodes     */
        int     zLoLimit,               /* z-array Lower Interior Nodes     */
        int     xHiLimit,               /* x-array Upper Interior Nodes     */
        int     yHiLimit,               /* y-array Upper Interior Nodes     */
        int     zHiLimit,               /* z-array Upper Interior Nodes     */ 
        double  delX,                   /* distance/cell, xdir              (INPUT) */
        double  delY,                   /* distance/cell, ydir              (INPUT) */
        double  delZ,                   /* distance/cell, zdir              (INPUT) */
        double  ****div_velFC_CC,       /* divergence of face cented vel at (INPUT) */
        double  ****delPress_CC,        /* Change in the cell-centered press(INPUT) */
        double  ****press_CC,           /* Cell-center pressure             (INPUT) */
        double  ****rho_CC,             /* Cell-centered density            (INPUT) */
        double  delt,                   /* delta t                          (INPUT) */
        double  ****speedSound,         /* speed of Sound(x,y,z,material    (INPUT) */
        int     nMaterials )  
    
{
    int         i, j, k, m;

    double      vol, coeff;
/*__________________________________
* double check inputs
* and allocat some memory
*___________________________________*/
    assert ( xLoLimit >= 0 && xHiLimit <= X_MAX_LIM);
    assert ( yLoLimit >= 0 && yHiLimit <= Y_MAX_LIM);
    assert ( zLoLimit >= 0 && zHiLimit <= Z_MAX_LIM);
    /*__________________________________
    *   Now compute the pressure and
    *   change in pressure
    *___________________________________*/
    for ( m = 1; m <= nMaterials; m++)
    {
        for ( j = yLoLimit; j <= yHiLimit; j++)
        {
            for ( k = zLoLimit; k <= zHiLimit; k++)
            {
                for ( i = xLoLimit; i <= xHiLimit; i++)
                { 
                    vol     = delX * delY * delZ; 
                    coeff   = delt * rho_CC[m][i][j][k] * pow(speedSound[m][i][j][k],2) / vol;

                    delPress_CC[m][i][j][k] = -coeff * div_velFC_CC[m][i][j][k]; 

                    press_CC[m][i][j][k]    = press_CC[m][i][j][k] + delPress_CC[m][i][j][k];
                }
            }
        }
    }
 
    
/*______________________________________________________________________
*   DEBUGGING
*_______________________________________________________________________*/
#if switchDebug_explicit_delPress
    fprintf(stderr,"****************************************************************************\n");
    fprintf(stderr,"                        explicit_delPress\n");
    fprintf(stderr,"****************************************************************************\n");
   
                       
    printData_4d(       xLo,                yLo,            zLo,
                        xHi,                yHi,            zHi,
                        m,                  m,
                       "explicit_delPress",     
                       "Press_CC",          press_CC);
                       
    printData_4d(       xLo,                yLo,            zLo,
                        xHi,                yHi,            zHi,
                        m,                  m,
                       "explicit_delPress",     
                       "delPress_CC",       delPress_CC);
   fprintf(stderr,"****************************************************************************\n");
    
    fprintf(stderr,"press return to continue\n");
    getchar();
#endif
/*STOP_DOC*/
}
