/* 
======================================================================*/
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include "nrutil+.h"
#include "functionDeclare.h"
#include "parameters.h"
#include "switches.h"
#include "macros.h"

/*---------------------------------------------------------------------  
 Function:  equation_of_state--EOS: Step 1, Compute the cell centered pressured using the equation of state.
 Filename:  equation_of_state.c
 Purpose:   Compute the time advanced cell-centered pressure
 
 Computatonal Domain: INTERIOR Cells
            
 Ghostcell data dependency: NONE
              
            
   
 Version       Programmer         Date       Description                      
     -------   ----------         ----       -----------                 
        1.0     Todd Harman       10/13/99 

Warning:    This is currently setup for an incompressible ideal gas.
this will need to be changed when the energy equation is added.     
                        
 ---------------------------------------------------------------------  */
void equation_of_state(
        int     xLoLimit,               /* x-array Lower Interior Nodes     */
        int     yLoLimit,               /* y-array Lower Interior Nodes     */
        int     zLoLimit,               /* z-array Lower Interior Nodes     */
        int     xHiLimit,               /* x-array Upper Interior Nodes     */
        int     yHiLimit,               /* y-array Upper Interior Nodes     */
        int     zHiLimit,               /* z-array Upper Interior Nodes     */
        double  *R,                     /* Gas constant                     (INPUT) */
        double  ****press_CC,           /* Cell-center pressure             (OUPUT) */
        double  ****rho_CC,             /* Cell-centered density            (INPUT) */
        double  ****Temp_CC,            /* Cell-centered Temperature        (INPUT) */
        double  ****cv_CC,              /* currently extra stuff            */
        int     nMaterials   )

{
    int     i, j, k, m;                 /* cell face locators               */
    char    should_I_write_output;
    double  neg_test1 = 0.0;            /* test for negative variables      */
    double  neg_test2 = 0.0;
/*__________________________________
*   Plotting variables
*___________________________________*/
#if switchDebug_equation_of_state
    double delX = 1;
    #include "plot_declare_vars.h"
#endif
/*__________________________________
* double check inputs, 
*___________________________________*/
    assert ( xLoLimit >= 0 && xHiLimit <= X_MAX_LIM);
    assert ( yLoLimit >= 0 && yHiLimit <= Y_MAX_LIM);
    assert ( zLoLimit >= 0 && zHiLimit <= Z_MAX_LIM);

    for ( m = 1; m <= nMaterials; m++)
    {
        for ( i = xLoLimit; i <= xHiLimit; i++)
        {
            for ( j = yLoLimit; j <= yHiLimit; j++)
            {
                for ( k = zLoLimit; k <= zHiLimit; k++)
                {
                    press_CC[m][i][j][k] = rho_CC[m][i][j][k] * R[m] * Temp_CC[m][i][j][k];
                    
                    neg_test1 = DMIN(neg_test1, press_CC[m][i][j][k]);
                    neg_test2 = DMIN(neg_test2, Temp_CC[m][i][j][k]);
                }
            }
        }
    }
/*__________________________________
*   Bullet Proofing
*___________________________________*/
    if(neg_test1 <0.0)
        Message(1, "File:equation_of_state.c",
        "Function equation_of_state","Negative pressure detected");

    if(neg_test2 <0.0)
        Message(1, "File:equation_of_state.c",
        "Function equation_of_state","Negative Temperature detected"); 
/*______________________________________________________________________
*   DEBUGGING SECTION   
*_______________________________________________________________________*/
#if switchDebug_equation_of_state
    should_I_write_output = *getenv("SHOULD_I_WRITE_OUTPUT");
    if ( should_I_write_output == '1')
    {
         #define switchInclude_equation_of_state 1
         #include "debugcode.i"
         #undef switchInclude_equation_of_state
    }    
#endif
/*__________________________________
*   Quite fullwarn remarks in a way that
*   is compiler independent
*___________________________________*/
    QUITE_FULLWARN(cv_CC[1][0][0][0]);          QUITE_FULLWARN(nMaterials);
    should_I_write_output = should_I_write_output;
/*STOP_DOC*/
}
