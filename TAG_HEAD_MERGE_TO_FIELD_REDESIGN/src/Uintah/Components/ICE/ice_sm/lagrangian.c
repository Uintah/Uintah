 /* 
 ======================================================================*/
#include <assert.h>
#include <time.h>
#include <sys/types.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "macros.h"
#include "functionDeclare.h"
#include "parameters.h"
#include "switches.h"
#include "nrutil+.h"

/* ---------------------------------------------------------------------
 Function:  lagrangian_vol--LAGRANGIAN: Step 5, Computes the cell-centered, time n+1, lagrangian volume 
 Filename:  lagrangian.c

 Purpose:
   This function calculates the The cell-centered, time n+1, lagrangian volume
   
 Computational Domain:
    The face-centered velocity of each face is needed by each cell in the 
    domain
    
 Ghostcell data dependency: 
   None

 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       03/22/99    

 
 ---------------------------------------------------------------------  */
void    lagrangian_vol(  
        int     xLoLimit,               /* x-array lower limit              */
        int     yLoLimit,               /* y-array lower limit              */
        int     zLoLimit,               /* z-array lower limit              */
        int     xHiLimit,               /* x-array upper limit              */
        int     yHiLimit,               /* y-array upper limit              */
        int     zHiLimit,               /* z-array upper limit              */
        double  delX,                   /* distance/cell, xdir              (INPUT) */
        double  delY,                   /* distance/cell, ydir              (INPUT) */
        double  delZ,                   /* distance/cell, zdir              (INPUT) */
        double  delt,                   /* delta t                          (INPUT) */
        double  ****Vol_L_CC,           /* Lagrangian volume                (OUPUT) */
                                        /* (i,j,k,m)                        */
        double   ***Vol_CC,             /* cell-centered volume             (INPUT) */
                                        /* (i,j,k)                          */
                                        /* (*)vel_FC(x,y,z, face,material)  */
        double  ******uvel_FC,          /* u-face-centered velocity         (INPUT) */ 
        double  ******vvel_FC,          /*  v-face-centered velocity        (INPUT) */
        double  ******wvel_FC,          /* w face-centered velocity         (INPUT) */
        int     nMaterials      )
{
    int     i, j, k,m;                  /*   loop indices  locators         */ 
                         
    double  ****div_vel_FC;             /* array containing the divergence  */            
    char    should_I_write_output;             
/*__________________________________
*   Plotting variables
*___________________________________*/ 
#if switchDebug_lagrangian_vol
    double ****plot_1;                     /* plot_1ing array                    */       
    #include "./Header_files/plot_declare_vars.h"   
    plot_1    = darray_4d(0, X_MAX_LIM,   0, Y_MAX_LIM, 0, Z_MAX_LIM, 1, N_MATERIAL); 
    
    zero_arrays_4d(     (xLoLimit),     (yLoLimit),      (zLoLimit),              
                        (xHiLimit),     (yHiLimit),      (zHiLimit), 
                        1,              nMaterials,         
                        1,              plot_1); 
     
#endif  
/*__________________________________
* Check that the inputs are reasonable
*___________________________________*/
    assert ( xLoLimit > 0 && xHiLimit < X_MAX_LIM);
    assert ( yLoLimit > 0 && yHiLimit < Y_MAX_LIM);
    assert ( zLoLimit > 0 && zHiLimit < Z_MAX_LIM);
    assert ( delt > 0 );
    assert ( delX > 0 && delY > 0 && delZ > 0);
    div_vel_FC= darray_4d(1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);    

/*______________________________________________________________________
*
*_______________________________________________________________________*/
    for (m = 1; m <= nMaterials; m++)
    {  
    /*__________________________________
    *   Compute the divergence of the
    *   face centered velocity
    *___________________________________*/
    divergence_of_face_centered_velocity(  
                    xLoLimit,           yLoLimit,           zLoLimit,
                    xHiLimit,           yHiLimit,           zHiLimit,
                    delX,               delY,               delZ,
                    uvel_FC,            vvel_FC,            wvel_FC,
                    div_vel_FC,         nMaterials);      
                    
        /*__________________________________
        *   Now compute the lagrangian volume
        *___________________________________*/   
        for ( i = xLoLimit; i <= xHiLimit; i++)
        {
            for ( j = yLoLimit; j <= yHiLimit; j++)
            {
                for ( k = zLoLimit; k <= zHiLimit; k++)
                {     
                    Vol_L_CC[m][i][j][k] = Vol_CC[i][j][k] + delt* div_vel_FC[m][i][j][k];

                    #if switchDebug_lagrangian_vol
                        plot_1[m][i][j][k] = delt * div_vel_FC[m][i][j][k];
                    #endif
                }
            }
        }
    }
/*STOP_DOC*/
/*______________________________________________________________________
*   DEBUGGING information see debugcode.i 
*_______________________________________________________________________*/ 
#if switchDebug_lagrangian_vol
    should_I_write_output = *getenv("SHOULD_I_WRITE_OUTPUT");
    if ( should_I_write_output == '1')
    {
        #define switchInclude_lagrangian_vol 1
        #include "debugcode.i" 
        #undef switchInclude_lagrangian_vol  
        free_darray_4d( plot_1,1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    }
#endif 
/*__________________________________
*   Deallocate memory
*___________________________________*/
   free_darray_4d(div_vel_FC, 1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
/*__________________________________
*   Quite fullwarn remarks in a way that
*   is compiler independent
*___________________________________*/
    QUITE_FULLWARN(*wvel_FC[0][0][0][1][1]);
    should_I_write_output = should_I_write_output;

}


/* 
 ======================================================================*/
#include "functionDeclare.h"
#include "parameters.h"
#include "switches.h"
#include <assert.h>
#include <time.h>
#include <sys/types.h>
#include <math.h>

/* --------------------------------------------------------------------- 
 Function:  lagrangian_values--LAGRANGIAN: Step 5, calculates the cell-centered, time n+1, lagrangian mass, momentum and internal energy 
 Filename:  lagrangian.c
 
 Purpose:
   This function calculates the cell-centered, time n+1, lagrangian mass, 
   momentum and internal energy
  
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       03/22/99    
       
Note:
    Index convention
                    ****Variable  (i,j,k,material)
                    ***Variable (i,j,k)
Implementation notes:
    Note that the lagrangian values are computed over the entire 
    domain including the ghostcells.  The lagrangian values in the 
    ghost cells are assumed to be the cell-centered quantities.  Therefore
    the mass, xmom, ymom, zmom, int_eng source terms must be zero in the ghost
    cells  
 
 ---------------------------------------------------------------------  */
void lagrangian_values(            
        int     xLoLimit,               /* x-array lower limit              */
        int     yLoLimit,               /* y-array lower limit              */
        int     zLoLimit,               /* z-array lower limit              */
        int     xHiLimit,               /* x-array upper limit              */
        int     yHiLimit,               /* y-array upper limit              */
        int     zHiLimit,               /* z-array upper limit              */
        double  ****Vol_L_CC,           /* Lagrangian volume                (INPUT) */
        double  ***Vol_CC,              /* cell-centered volume             (INPUT) */
        double  ****rho_CC,             /* cell-centered material density   (INPUT) */
        double  ****rho_L_CC,           /* cell-centered lagrangian density (INPUT) */
        double  ****xmom_CC,            /* cell-centered x-momentum         (INPUT) */
        double  ****ymom_CC,            /* cell-centered x-momentum         (INPUT) */
        double  ****zmom_CC,            /* cell-centered x-momentum         (INPUT) */        
        double  ****uvel_CC,            /* u-cell-centered velocity         (INPUT) */
        double  ****vvel_CC,            /*  v-cell-centered velocity        (INPUT) */
        double  ****wvel_CC,            /* w cell-centered velocity         (INPUT) */
        double  ****xmom_L_CC,          /* Lagrangian cell-centered momentum(OUPUT) */
        double  ****ymom_L_CC,          /* Lagrangian cell-centered momentum(OUPUT) */
        double  ****zmom_L_CC,          /* Lagrangian cell-centered momentum(OUPUT) */
        double  ****mass_L_CC,          /* cell-centered lagrangian mass    (OUPUT) */
        double  ****mass_source,        /* cell-centered source term for mass(INPUT)*/
        double  ****xmom_source,        /* cell-centered source term        (INPUT) */
        double  ****ymom_source,        /* cell-centered source term        (INPUT) */
        double  ****zmom_source,        /* cell-centered source term        (INPUT) */
        double  ****int_eng_CC,         /* internal energy                  (INPUT) */
        double  ****int_eng_L_CC,       /* internal energy                  (OUPUT) */
        double  ****int_eng_source,     /* internal energy                  (INPUT) */        
        int     nMaterials     )
{
    int i, j, k,m;                      /*   loop indices  locators         */
    
#if sw_lagrangian_values
    time_t start,secs;                  /* timing variables                */
    start = time(NULL);
#endif 

/*__________________________________
* Check that the inputs are reasonable
*___________________________________*/
    assert ( xLoLimit > 0 && xHiLimit < X_MAX_LIM);
    assert ( yLoLimit > 0 && yHiLimit < Y_MAX_LIM);
    assert ( zLoLimit > 0 && zHiLimit < Z_MAX_LIM);
 
/*START_DOC*/
/*______________________________________________________________________
*  CODE
*  The source terms must be 0.0 in the ghostcells
*_______________________________________________________________________*/
    for (m = 1; m <= nMaterials; m++)
    {
        for ( i = GC_LO(xLoLimit); i <= GC_HI(xHiLimit); i++)
        {
            for ( j = GC_LO(yLoLimit); j <= GC_HI(yHiLimit); j++)
            {
                for ( k = GC_LO(zLoLimit); k <= GC_HI(zHiLimit); k++)
                { 

                    /*__________________________________
                    * Lagrangian mass
                    *___________________________________*/
                    mass_L_CC[m][i][j][k] = ( rho_CC[m][i][j][k] * Vol_CC[i][j][k] 
                                           +  mass_source[m][i][j][k]);
                                           
                    rho_L_CC[m][i][j][k] = mass_L_CC[m][i][j][k]/Vol_CC[i][j][k];
                    /*__________________________________
                    * Lagrangian Momentum
                    *___________________________________*/           
                    xmom_L_CC[m][i][j][k] = xmom_CC[m][i][j][k]
                    - uvel_CC[m][i][j][k] * mass_source[m][i][j][k]
                    + xmom_source[m][i][j][k];

                    ymom_L_CC[m][i][j][k] = ymom_CC[m][i][j][k]
                    - vvel_CC[m][i][j][k] * mass_source[m][i][j][k]
                    + ymom_source[m][i][j][k];

                    zmom_L_CC[m][i][j][k] = zmom_CC[m][i][j][k]
                    - wvel_CC[m][i][j][k] * mass_source[m][i][j][k]
                    + zmom_source[m][i][j][k];
                    /*__________________________________
                    * Lagrangian energy
                    *___________________________________*/
                    int_eng_L_CC[m][i][j][k] = int_eng_CC[m][i][j][k] 
                    - int_eng_CC[m][i][j][k] * mass_source[m][i][j][k]
                    + int_eng_source[m][i][j][k];
                }
            }
        }
    }

/*STOP_DOC*/
/*______________________________________________________________________
*   DEBUGGING AND STOP WATCH INFORMATION
*_______________________________________________________________________*/

#if switchDebug_lagrangian_values
    for (m = 1; m <= nMaterials; m++)
    {
        fprintf(stderr, "\t Material %i \n",m);
        printData_4d(       GC_LO(xLoLimit),    GC_LO(yLoLimit),    (zLoLimit),
                            GC_HI(xHiLimit),    GC_HI(yHiLimit),    zHiLimit,
                            m,                  m,
                           "lagrangian_values","xmom_L_CC",         xmom_L_CC);

        printData_4d(       GC_LO(xLoLimit),    GC_LO(yLoLimit),    (zLoLimit),
                           GC_HI(xHiLimit),     GC_HI(yHiLimit),    zHiLimit,
                            m,                  m,
                           "lagrangian_values","ymom_L_CC",         ymom_L_CC);

        printData_4d(       GC_LO(xLoLimit),    GC_LO(yLoLimit),    (zLoLimit),
                            GC_HI(xHiLimit),    GC_HI(yHiLimit),    zHiLimit,
                            m,                  m,
                           "lagrangian_values","zmom_L_CC",         zmom_L_CC);     

        printData_4d(       GC_LO(xLoLimit),    GC_LO(yLoLimit),     (zLoLimit),
                            GC_HI(xHiLimit),    GC_HI(yHiLimit),     zHiLimit,
                            m,                  m,
                           "lagrangian_values","mass_L_CC",         mass_L_CC    ); 
            fprintf(stderr, "Press return to continue\n");
            getchar();
        
    }                     
#endif 
       
#if sw_lagrangian_values
    stopwatch("lagragian values",start);
#endif
/*__________________________________
*   Quite fullwarn remarks in a way that
*   is compiler independent
*___________________________________*/
    QUITE_FULLWARN(wvel_CC[1][0][0][0]);
    QUITE_FULLWARN(Vol_L_CC[1][0][0][0]); 
   rho_L_CC = rho_L_CC;   
}

