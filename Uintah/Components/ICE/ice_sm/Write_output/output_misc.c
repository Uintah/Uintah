/* 
 ======================================================================*/
#include <stdlib.h>
#include <math.h>
#include "parameters.h"
#include "functionDeclare.h"
#include "switches.h"
#define  EPSILON 1.0e-6

/*
 Function:  Is_it_time_to_write_output--OUTPUT: test to see if it is time to dump an output file or plot to the screen.
 Filename:  output_misc.c
 Purpose:
   Test to see if it is time to dump an output file or plot to the screen.
   If it is then return YES if not then return NO.  A environmental variable
   is written telling whether or not to write the file. 
   
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       10/12/99    
 
 ---------------------------------------------------------------------  */ 
 int Is_it_time_to_write_output(
        double  t,                      /* current time                     */
        double  *t_output_vars  )       /* array holding output timing info */
                                        /* t_output_vars[1] = t_initial     */
                                        /* t_output_vars[2] = t final       */
                                        /* t_output_vars[3] = delta t       */                                
{
    static double   told;
    static int      first_time_through;
    
    first_time_through ++; 
/*__________________________________
* Do this only the first time through
*___________________________________*/
     if( t >= t_output_vars[1] && 
         t <= t_output_vars[2] &&
         first_time_through == 1  )
    {
        told = t;
        putenv("SHOULD_I_WRITE_OUTPUT=1");
        return YES;
    }
    

/*__________________________________
*   For all other passes through routine
*   Add epsilon to the difference of t - told
*   because of roundoff error.
*___________________________________*/   
    if( t >= t_output_vars[1] && 
        t <= t_output_vars[2] &&
         ((fabs(t - told) + EPSILON ) >= t_output_vars[3]) &&
        first_time_through != 1  )
    {
        told = t;
        putenv("SHOULD_I_WRITE_OUTPUT=1");
        return YES;
    }
    else
    {
        putenv("SHOULD_I_WRITE_OUTPUT=-9");
        return NO;
    } 
}
/*STOP_DOC*/
