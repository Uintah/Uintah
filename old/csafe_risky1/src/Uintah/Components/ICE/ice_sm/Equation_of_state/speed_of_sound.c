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
 Function:  speed_of_sound--EOS: Step 1, Compute the cell centered speed of sound
 Filename:  speed_of_sound.c
 Purpose:   Compute the time advanced speed of sound
 
 Computatonal Domain: INTERIOR Cells
                    
   
 Version       Programmer         Date       Description                      
     -------   ----------         ----       -----------                 
        1.0     Todd Harman       3/10/99     
                        
 ---------------------------------------------------------------------  */
void speed_of_sound(
        int     xLoLimit,               /* x-array Lower Interior Nodes     */
        int     yLoLimit,               /* y-array Lower Interior Nodes     */
        int     zLoLimit,               /* z-array Lower Interior Nodes     */
        int     xHiLimit,               /* x-array Upper Interior Nodes     */
        int     yHiLimit,               /* y-array Upper Interior Nodes     */
        int     zHiLimit,               /* z-array Upper Interior Nodes     */
        double  *gamma,                 /* ratio of specific heats          (INPUT) */
        double  *R,                     /* ideal gas constant               (INPUT) */
        double  ****Temp_CC,            /* Cell-centered Temperature        (INPUT) */
        double  ****speedSound,         /* speed of sound                   (OUTPUT)*/
        int     nMaterials   )
{
    int     i, j, k, m;                 /* cell face locators               */
    char    should_I_write_output;

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
                    speedSound[m][i][j][k] = sqrt(gamma[m] * R[m] * Temp_CC[m][i][j][k]);
 
                }
            }
        }
    }
/*______________________________________________________________________
*   DEBUGGING SECTION   
*_______________________________________________________________________*/
#if switchDebug_speed_of_sound
    should_I_write_output = *getenv("SHOULD_I_WRITE_OUTPUT");
    if ( should_I_write_output == '1')
    {
         #define switchInclude_speed_of_sound 1
         #include "debugcode.i"
         #undef switchInclude_speed_of_sound
    }    
#endif
/*__________________________________
*   Quite fullwarn remarks in a way that
*   is compiler independent
*___________________________________*/
    QUITE_FULLWARN(nMaterials);
    should_I_write_output = should_I_write_output;
/*STOP_DOC*/
}
