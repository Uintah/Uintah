/* 
 ======================================================================*/
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <sys/types.h>
#include <assert.h>
#include "functionDeclare.h"
#include "switches.h"
#include "parameters.h"
#include "macros.h"
/* ---------------------------------------------------------------------  
 Function:  press_face--FACE-CENTERED PRESS: Step 3,Computes the face-centered pressure from cell-centered data.
 Filename:  p_face.c
 
 Purpose:
    This function calculates the face centered pressure on each of the 
    cell faces for every cell in the computational domain and a single layer of
    ghost cells.  This routine assume that there is a single layer of ghostcells

Steps:
    1)  Compute the face-centered pressure for the top and right face for all
    cells in the computational domain.
    
Additional Notes:
    The computations were divided up to eliminate
    the need for a "if" statement inside of the loop.
   
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       02/22/99 
                                 05/23/00   incorporated update physical BC   
 
 NEED to add 3rd dimension
  ---------------------------------------------------------------------  */
void    press_face(
        int     xLoLimit,               /* x-array lower limit              */
        int     yLoLimit,               /* y-array lower limit              */
        int     zLoLimit,               /* z-array lower limit              */
        int     xHiLimit,               /* x-array upper limit              */
        int     yHiLimit,               /* y-array upper limit              */
        int     zHiLimit,               /* z-array upper limit              */
        double  delX,   
        double  delY,   
        double  delZ,
        int     ***BC_types,            /* each variable can have a Neuman, */
                                        /* or Dirichlet type BC             */
                                        /* BC_types[wall][variable][m]=type */
        int     ***BC_float_or_fixed,   /* BC_float_or_fixed[wall][variable][m]*/
                                        /* Variable on boundary is either   */
                                        /* fixed or it floats during the    */
                                        /* compuation                       */
        double  ***BC_Values,           /* BC values BC_values[wall][variable][m]*/  
        
        double  ****press_CC,           /* cell-centered pressure           (INPUT) */   
        double  ******press_FC,         /* face-centered pressure           (OUPUT) */
        double  ****rho_CC,             /* cell-centered density            (INPUT) */
        int     nMaterials      )
{
    int     i,j,k, m,
            xLo,        xHi,
            yLo,        yHi,
            zLo,        zHi, 
            cell;       
    double       
            sp_vol,                     /* specific vol.                    */
            sp_vol_adj;                 /* specific vol. in adjacent cel.   */
    char    should_I_write_output;
#if sw_p_face
    time_t start;                       /* timing variables                */
    start = time(NULL);
#endif 
/*__________________________________
*   Plotting variables
*___________________________________*/
#if switchDebug_p_face
    #include "plot_declare_vars.h"   
#endif

/*__________________________________
*   Step 1:
*   Interpolate the pressure from the 
*   cell-center on to the face on the top
*   and right face.
*___________________________________*/  
    for(m = 1; m <= nMaterials; m++)
    {
        xLo = GC_LO(xLoLimit);
        xHi = GC_HI(xHiLimit);
        yLo = GC_LO(yLoLimit);
        yHi = (yHiLimit);
        zLo = GC_LO(zLoLimit);
        zHi = GC_HI(zHiLimit); 
        
        for(k = zLo; k <= zHi ; k++)
        {
            for(j = yLo; j <= yHi; j++)
            { 
                for(i = xLo; i <= xHi; i++)
                {
                    sp_vol      = 1.0/rho_CC[m][i][j][k];
                    /*__________________________________
                    * Top Face
                    *___________________________________*/
                    cell        = j+1;
                    sp_vol_adj  = 1.0/rho_CC[m][i][cell][k];
                    
                    
                    assert ( (sp_vol_adj + sp_vol) <=BIG_NUM);
                    *press_FC[i][j][k][TOP][m]      = 
                    (press_CC[m][i][cell][k] * sp_vol_adj + press_CC[m][i][j][k] * sp_vol)/
                            (sp_vol + sp_vol_adj);    

                }
            }
        }
        yHi = GC_HI(yHiLimit);
        xHi = xHiLimit;
        xLo = GC_LO(xLoLimit);
        yLo = GC_LO(yLoLimit);
        for(k = zLo; k <= zHi ; k++)
        {
            for(j = yLo; j <= yHi; j++)
            { 
                for(i = xLo; i <= xHi; i++)
                {
                    sp_vol      = 1.0/rho_CC[m][i][j][k];
                    /*__________________________________
                    * Right Face
                    *___________________________________*/
                    cell        = i+1;
                    sp_vol_adj  = 1.0/rho_CC[m][cell][j][k];
                    assert ( (sp_vol_adj + sp_vol) <=BIG_NUM);
                    
                    *press_FC[i][j][k][RIGHT][m]    =
                    (press_CC[m][cell][j][k] * sp_vol_adj + press_CC[m][i][j][k] * sp_vol)/
                            (sp_vol_adj + sp_vol);                
                }
            }
        }
/*`==========TESTING==========*/ 
        /*__________________________________
        *  Do something in the third dimension
        *  Need to add front or back face
        *___________________________________*/    
        yHi = GC_HI(yHiLimit);
        xHi = GC_HI(xHiLimit);
        xLo = GC_LO(xLoLimit);
        yLo = GC_LO(yLoLimit);
        for(k = zLo; k <= zHi ; k++)
        {
            for(j = yLo; j <= yHi; j++)
            { 
                for(i = xLo; i <= xHi; i++)
                {
                    *press_FC[i][j][k][FRONT][m]   = 0.0;
                    *press_FC[i][j][k][BACK][m]    = 0.0;                              
                }
            }
        }
 /*==========TESTING==========`*/
        
    }
    /*__________________________________
    * Update the boundary conditions
    *___________________________________*/    
        update_CC_FC_physical_boundary_conditions( 
                        xLoLimit,       yLoLimit,       zLoLimit,             
                        xHiLimit,       yHiLimit,       zHiLimit,             
                        delX,           delY,           delZ,
                        BC_types,       BC_float_or_fixed,
                        BC_Values, 
                        nMaterials,     1,                 
                        press_CC,       PRESS,          press_FC);
    
    
/*STOP_DOC*/
/*______________________________________________________________________
*   DEBUGGING AND STOP WATCH INFORMATION
*_______________________________________________________________________*/
#if switchDebug_p_face
    should_I_write_output = *getenv("SHOULD_I_WRITE_OUTPUT");
    if ( should_I_write_output == '1')
    {
         #define switchInclude_p_face 1
         #include "debugcode.i"
         #undef switchInclude_p_face
    }                            
#endif 
       
#if sw_p_face
    stopwatch("press_face", start);
#endif
/*__________________________________
*   Quite fullwarn comments
*___________________________________*/
    should_I_write_output = should_I_write_output;
}
