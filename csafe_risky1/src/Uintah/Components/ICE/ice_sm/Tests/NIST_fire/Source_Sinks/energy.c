/* 
======================================================================*/
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include "functionDeclare.h"
#include "nrutil+.h"
#include "parameters.h"
#include "switches.h"
#include "macros.h"

/* --------------------------------------------------------------------- 
 Function:  accumulate_energy_source_sinks--SOURCE: Step 4, Accumulate all of the energy source terms.
 Filename:  energy.c
 
 Computational Domain:
    The energy sources are accumulated at the cell-center of each
    cell of the domain.
    
Ghostcell data dependency: 
    None

 Purpose:   This function accumulates all of the sources/sinks of energy
            which is added to the current value for the energy to form
            the Lagrangian energy  
 Version       Programmer         Date       Description                      
     -------   ----------         ----       -----------                 
        1.0     Todd Harman       10/22/99 

 Currently the kinetic energy isn't 
 included.
 
 This is the routine where you would add additional sources/sinks of energy
                                 
 ---------------------------------------------------------------------  */

void accumulate_energy_source_sinks(
    int         xLoLimit,               /* x-array Lower Interior Nodes     */
    int         yLoLimit,               /* y-array Lower Interior Nodes     */
    int         zLoLimit,               /* z-array Lower Interior Nodes     */
    int         xHiLimit,               /* x-array Upper Interior Nodes     */
    int         yHiLimit,               /* y-array Upper Interior Nodes     */
    int         zHiLimit,               /* z-array Upper Interior Nodes     */
    double      delt,                   /* time increment                   (INPUT) */
    double      delX,                   /* cell spacing                     (INPUT) */
    double      delY,                   /*          ---//---                (INPUT) */
    double      delZ,                   /*          ---//---                (INPUT) */
    double      *grav,                  /* gravity (dir)                    (INPUT) */
                                        /* 1 = x, 2 = y, 3 = z              (INPUT) */
    double      ****mass_CC,            /* cell-centered mass               (INPUT) */
    double      ****rho_CC,             /* Cell-centered density            (INPUT) */
    double      ****press_CC,           /* Cell-center pressure             (INPUT) */
    double      ****delPress_CC,        /* Cell-centered change in press.   (INPUT) */
    double      ****Temp_CC,            /* Cell-centered Temperature        (INPUT) */
    double      ****cv_CC,              /* Extra varialble for now          (INPUT) */
    double      ****speedSound,         /* speed of sound (x,y,z, material) (INPUT) */
    double      ****uvel_CC,            /* cell-centered velocities         (INPUT) */
    double      ****vvel_CC,            /*          ---//---                (INPUT) */
    double      ****wvel_CC,            /*          ---//---                (INPUT) */
    double      ****div_velFC_CC,       /* divergence of face centered vel. (INPUT) */
                                        /* that lives at the CC                     */
    double      ***Q_chem,              /* contribution from NIST fire model(INPUT) */
    double      ****int_eng_source,     /* internal energy source/sink      (OUPUT) */
    int         nMaterials   )
{
    int     i, j, k, m;                 /* cell face locators               */

    double 
            A,B,C,                      /* terms in the equations           */
            dummy,
            int_eng_source_tmp,         /* temporary variables that make    */         
            switch_eng;                 /* = 1 calculate, =0 Don't calc.    */  

            
    char    should_I_write_output;
    
/*__________________________________
*   Plotting variables
*___________________________________*/
#if switchDebug_accumulate_energy_source_sinks
    #include "plot_declare_vars.h"   
#endif
/*__________________________________
*   HARDWIRE FOR NOW amd initialize 
*   variables
*___________________________________*/
    m           = nMaterials;
    dummy       = 0.0;
    switch_eng  = 1.0;
/*__________________________________
* double check inputs, 
*___________________________________*/
    assert ( xLoLimit >= 0 && xHiLimit <= X_MAX_LIM);
    assert ( yLoLimit >= 0 && yHiLimit <= Y_MAX_LIM);
    assert ( zLoLimit >= 0 && zHiLimit <= Z_MAX_LIM);
    
/*START_DOC*/
/*__________________________________
*   Calculate the terms in the 
*   right hand side of the energy eq.
*   This is where you calculate all of the
*   the additional sources/sinks of energy
*___________________________________*/


/*__________________________________
*   Now accumulate the different contributions
*___________________________________*/
    for (m = 1; m <= nMaterials; m++)
    {
        for ( i = xLoLimit; i <= xHiLimit; i++)
        {
            for ( j = yLoLimit; j <= yHiLimit; j++)
            {
                for ( k = zLoLimit; k <= zHiLimit; k++)
                {
                    /*__________________________________
                    * contribution from pressure
                    *___________________________________*/  
                    int_eng_source_tmp          =   -delt * press_CC[m][i][j][k] * div_velFC_CC[m][i][j][k];
                    
                    #if switch_step4_NIST_fire
                    int_eng_source_tmp          =   int_eng_source_tmp + Q_chem[i][j][k] * (1.0 - RAD_COEFFICIENT);
                    #endif
                    
                    int_eng_source[m][i][j][k]  =   switch_eng * int_eng_source_tmp;

                }
            }
        }
    }

/*______________________________________________________________________
*   DEBUGGING SECTION   
*_______________________________________________________________________*/
#if switchDebug_accumulate_energy_source_sinks
    should_I_write_output = *getenv("SHOULD_I_WRITE_OUTPUT");
    if ( should_I_write_output == '1')
    {
         #define switchInclude_accumulate_energy_source_sinks 1
         #include "debugcode.i"
         #undef switchInclude_accumulate_enegy_source_sinks
    }
#endif

/*__________________________________
*   Quite all fullwarn compiler remarks
*___________________________________*/
    QUITE_FULLWARN(Q_chem[1][1][1]);
    QUITE_FULLWARN(delX);                   QUITE_FULLWARN(delY);
    QUITE_FULLWARN(delZ);
    QUITE_FULLWARN(delPress_CC[1][0][0][0]);                  
    QUITE_FULLWARN(mass_CC[1][0][0][0]);    QUITE_FULLWARN(speedSound[1][0][0][0]);
    QUITE_FULLWARN(rho_CC[1][0][0][0]);     QUITE_FULLWARN(Temp_CC[1][0][0][0]);
    QUITE_FULLWARN(cv_CC[1][0][0][0]);      QUITE_FULLWARN(grav[1]);
    QUITE_FULLWARN(uvel_CC[1][0][0][0]);    QUITE_FULLWARN(vvel_CC[1][0][0][0]);
    QUITE_FULLWARN(wvel_CC[1][0][0][0]);    QUITE_FULLWARN(nMaterials);
    should_I_write_output = should_I_write_output; 
    A        =   A;
    B        =   B;
    C        =   C;
    dummy    =   dummy;   
}
/*STOP_DOC*/
