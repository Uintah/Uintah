/* 
 ======================================================================*/
 #include <assert.h>
 #include <math.h>
 #include "macros.h"
 #include "parameters.h"
 #include "functionDeclare.h"
/* --------------------------------------------------------------------- 
 Function:  initialize_darray_4d--DEBUG: Initialized a 4D array to a constant or a specified gradient.
 Filename:  initializd_variables.c

 Purpose:  Initialize an multimaterial array to the value of Constant.  This routine 
 also allows the user to specify a imposed gradient with the switch grad_dir.
 During the initialization the ghostcells surrounding the domain are included.
 This routine is used in testing functions.  
 
 This function is mainly used in debugging

 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       03/18/99   Written   

 --------------------------------------------------------------------- */

void  initialize_darray_4d( 
    int     xLoLimit,                   /* x-array lower limit              */
    int     yLoLimit,                   /* y-array lower limit              */
    int     zLoLimit,                   /* z-array lower limit              */
    int     xHiLimit,                   /* x-array upper limit              */
    int     yHiLimit,                   /* y-array upper limit              */
    int     zHiLimit,                   /* z-array upper limit              */
    double  ****data_array,             /* data(x,y,z,material                      (IN/OUT)*/                   
    int     m,                          /* material index                           */
    double  constant,                   /* value to initialize the array to (INPUT) */
    int     switch_funct,               /* Switch that specifies which user function*/
                                        /* = 0 data = constant                      */
                                        /* = 1 data = constant + x                  */
                                        /* = 2 data = constant + y                  */
                                        /* = 3 data = constant + z                  */
                                        /* = 4 data = see macros.h                  */
    int     flag_GC )                   /* = 1 if you want to include ghost (INPUT) */
                                        /* = 0 don't want to include ghostcells     */

{                                   
    int i, j, k,                    
    xLo, xHi,
    yLo, yHi,
    zLo, zHi;
/*START_DOC*/
/*______________________________________________________________________
*  Calculate th loop indices including the surrounding ghost cells
*_______________________________________________________________________*/

    if (flag_GC == 1)
    {
        xLo = GC_LO(xLoLimit);
        xHi = GC_HI(xHiLimit);
        yLo = GC_LO(yLoLimit);
        yHi = GC_HI(yHiLimit);
        zLo = GC_LO(zLoLimit);
        zHi = GC_HI(zHiLimit); 
    }
    
    if (flag_GC != 1 )
    {    
        xLo = xLoLimit;
        xHi = xHiLimit;
        yLo = yLoLimit;
        yHi = yHiLimit;
        zLo = zLoLimit;
        zHi = zHiLimit; 
    }
  

    assert ( xLo >= 0 && xHi <= X_MAX_LIM);
    assert ( yLo >= 0 && yHi <= Y_MAX_LIM);
    assert ( zLo >= 0 && zHi <= Z_MAX_LIM);
      
/*__________________________________
*   Now impose gradient
*___________________________________*/            
    for(i = xLo; i <= xHi; i++)
    {
        for(j = yLo; j <= yHi; j++)
        {
             for(k = zLo; k <= zHi; k++)
            {
                    
                data_array[m][i][j][k] = USR_FUNCTION(switch_funct,i,j,k,constant);

            }
        }
    }


 }
/*STOP_DOC*/ 
 
 
/* 
 ======================================================================*/
 #include "parameters.h"
 #include "functionDeclare.h"
 #include "macros.h"
 #include <assert.h>
 #include <math.h>
 
/* --------------------------------------------------------------------- 
 Function:  initialize_darray_3d--DEBUG: Initialized a 3D array to a constant or a specified gradient.
 Filename:  initialize_variables.c
 
 Purpose:  Initialize an singlematerial array to the value of Constant.  This routine 
 also allows the user to specify a imposed gradient with the switch grad_dir.
 During the initialization the ghostcells surrounding the domain are included.
 This routine is used in testing functions.

 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       03/18/99   Written   

 --------------------------------------------------------------------- */

void    initialize_darray_3d(                    
    int     xLoLimit,                   /* x-array lower limit              */
    int     yLoLimit,                   /* y-array lower limit              */
    int     zLoLimit,                   /* z-array lower limit              */
    int     xHiLimit,                   /* x-array upper limit              */
    int     yHiLimit,                   /* y-array upper limit              */
    int     zHiLimit,                   /* z-array upper limit              */
    double  ***data_array,              /* array to monkey with             (IN/OUT)*/ 
    double  constant,                   /* value to initialize the array to (INPUT) */
    int     switch_funct,               /* Switch that specifies which user function*/
                                        /* = 0 data = constant                      */
                                        /* = 1 data = constant + x                  */
                                        /* = 2 data = constant + y                  */
                                        /* = 3 data = constant + z                  */
                                        /* = 4 data = see macros.h                  */    
    int     flag_GC )                   /* = 1 if you want to include ghost (INPUT) */
                                        /* = 0 don't want to include ghostcells     */
{ 
    int i, j, k,
    xLo, xHi,
    yLo, yHi,
    zLo, zHi;
/*START_DOC*/
/*______________________________________________________________________
*  Calculate th loop indices including the surrounding ghost cells
*_______________________________________________________________________*/

    if (flag_GC == 1)
    {
        xLo = GC_LO(xLoLimit);
        xHi = GC_HI(xHiLimit);
        yLo = GC_LO(yLoLimit);
        yHi = GC_HI(yHiLimit);
        zLo = GC_LO(zLoLimit);
        zHi = GC_HI(zHiLimit); 
    }
    
    if (flag_GC != 1 )
    {    
        xLo = xLoLimit;
        xHi = xHiLimit;
        yLo = yLoLimit;
        yHi = yHiLimit;
        zLo = zLoLimit;
        zHi = zHiLimit; 
    }

    assert ( xLo >= 0 && xHi <= X_MAX_LIM);
    assert ( yLo >= 0 && yHi <= Y_MAX_LIM);
    assert ( zLo >= 0 && zHi <= Z_MAX_LIM);   
      
/* ______________________________
  Now print the string
  print a new line if the returnchar
   is found.
______________________________  */                

    for(k = zLo; k <= zHi; k++)
    {
        for(j = yLo; j <= yHi; j++)
        {
            for(i = xLo; i <= xHi; i++)
            {
                data_array[i][j][k] = USR_FUNCTION(switch_funct,i,j,k,constant);
            }
        }
    }   
 }
 /*STOP_DOC*/ 

