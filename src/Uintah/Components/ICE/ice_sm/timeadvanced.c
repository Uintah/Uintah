/* 
 ======================================================================*/
#include <assert.h>
#include <time.h>
#include <sys/types.h>
#include <math.h>
#include <stdlib.h>              
#include "functionDeclare.h"
#include "parameters.h"
#include "switches.h"
#include "nrutil+.h"
#include "macros.h"

/* --------------------------------------------------------------------- 

 Function:  advect_and_advance_in_time--Steps 6 and 7:  controller for step 6 and updates the variables to n+1.   
 Filename:  timeadvanced.c 

 Purpose:
   This function calculates the The cell-centered, time n+1, mass, momentum
   and internal energy
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       09/??/99    
       2.0      //               04/19/00   Now computes the flux across each 
                                            face independently. 

Need to include kinetic energy 
 ---------------------------------------------------------------------  */
void advect_and_advance_in_time(  
        int     xLoLimit,               /* x-array lower limit              */
        int     yLoLimit,               /* y-array lower limit              */
        int     zLoLimit,               /* z-array lower limit              */
        int     xHiLimit,               /* x-array upper limit              */
        int     yHiLimit,               /* y-array upper limit              */
        int     zHiLimit,               /* z-array upper limit              */
        double  delX,                   /* distance/cell, xdir              (INPUT) */
        double  delY,                   /* distance/cell, ydir              (INPUT) */
        double  delZ,                   /* distance/cell, zdir              (INPUT) */ 
        double  ***Vol_CC,              /* cell-centered volume             (INPUT) */
        double  ****rho_CC,             /* cell-centered density            (OUPUT) */
        double  ****xmom_CC,            /* cell-centered x-momentum         (OUPUT) */
        double  ****ymom_CC,            /* cell-centered x-momentum         (OUPUT) */
        double  ****zmom_CC,            /* cell-centered x-momentum         (OUPUT) */
        double  ****Vol_L_CC,           /* Lagrangian cell-centered volume  (INPUT) */
        double  ****rho_L_CC,           /* Lagrangian cell-centered density (INPUT) */
        double  ****mass_L_CC,          /* Lagrangian cell-centered mass    (INPUT) */
        double  ****xmom_L_CC,          /* Lagrangian cell-centered momentum(INPUT) */
        double  ****ymom_L_CC,          /* Lagrangian cell-centered momentum(INPUT) */
        double  ****zmom_L_CC,          /* Lagrangian cell-centered momentum(INPUT) */
        double  ****int_eng_CC,         /* internal energy                  (OUPUT) */
        double  ****int_eng_L_CC,       /* Lagrangian CC internal energy    (INPUT) */
        double  ******uvel_FC,          /*  u-face-centered velocity        (INPUT) */
        double  ******vvel_FC,          /*  v-face-centered velocity        (INPUT) */
        double  ******wvel_FC,          /* w face-centered velocity         (INPUT) */
        double  delt,                   /* delta t                          (INPUT) */
        int     nMaterials      )
/* Local Definitions________________________________________________________*/
{
    int     i, j, k,m;                  /*   loop indices  locators         */
           
    
    double  vol,                        /* Temporary variable               */
            ****advct_xmom_CC,          /* Advected momemtum                */
            ****advct_ymom_CC,
            ****advct_zmom_CC,
            ****mass_L_TEMP,
            ****xmom_L_TEMP,              /* temporary variables              */
            ****ymom_L_TEMP,
            ****zmom_L_TEMP,
            ****int_eng_L_TEMP,
            ****advct_rho_CC,
            ****advct_int_eng_CC,       /* Advected interal energy          */            
            ****r_out_x,                /* x-dir centroid array (i,j,k,vol  */           
            ****r_out_y,                /* y-dir centroid array             */                            
            ****r_out_z,                /* z-dir centroid array             */            
            ****outflux_vol,            /* array containing the size of each*/
                                        /* of the outflux volumes           */ 
                                        /* (i,j,k,vol)                      */  
            ****influx_vol,             /* influx volume for each slab      */
            ****r_out_x_CF,             /* corner flux centroid x-dir       */
            ****r_out_y_CF,             /* corner flux centroid y-dir       */ 
            ****r_out_z_CF,             /* corner flux centroid z-dir       */
            ****influx_vol_CF,          /* corner flux contributions        */ 
            ****outflux_vol_CF;         /* corner flux contributions        */                   
                                      
    char    should_I_write_output;       
/*__________________________________
*   Plotting variables
*___________________________________*/ 
#if (switchDebug_advect_and_advance_in_time)
    #include "plot_declare_vars.h"   
#endif
#if sw_advect_and_advance_in_time    
    time_t start,secs;                  /* timing variables                 */               
    start = time(NULL);
#endif
/*START_DOC*/ 

/*__________________________________
* Check that the inputs are reasonable
*___________________________________*/
    assert ( xLoLimit > 0 && xHiLimit < X_MAX_LIM);
    assert ( yLoLimit > 0 && yHiLimit < Y_MAX_LIM);
    assert ( zLoLimit > 0 && zHiLimit < Z_MAX_LIM);
/*__________________________________
* Allocate memory for local arrays
*___________________________________*/
    advct_xmom_CC   = darray_4d(1, N_MATERIAL,      0, X_MAX_LIM, 0,    Y_MAX_LIM,      0, Z_MAX_LIM);
    advct_ymom_CC   = darray_4d(1, N_MATERIAL,      0, X_MAX_LIM, 0,    Y_MAX_LIM,      0, Z_MAX_LIM);
    advct_zmom_CC   = darray_4d(1, N_MATERIAL,      0, X_MAX_LIM, 0,    Y_MAX_LIM,      0, Z_MAX_LIM);
    advct_rho_CC    = darray_4d(1, N_MATERIAL,      0, X_MAX_LIM, 0,    Y_MAX_LIM,      0, Z_MAX_LIM);
    advct_int_eng_CC= darray_4d(1, N_MATERIAL,      0, X_MAX_LIM, 0,    Y_MAX_LIM,      0, Z_MAX_LIM);

    mass_L_TEMP     = darray_4d(1, N_MATERIAL,      0, X_MAX_LIM, 0,    Y_MAX_LIM,      0, Z_MAX_LIM);
    xmom_L_TEMP     = darray_4d(1, N_MATERIAL,      0, X_MAX_LIM, 0,    Y_MAX_LIM,      0, Z_MAX_LIM);
    ymom_L_TEMP     = darray_4d(1, N_MATERIAL,      0, X_MAX_LIM, 0,    Y_MAX_LIM,      0, Z_MAX_LIM);
    zmom_L_TEMP     = darray_4d(1, N_MATERIAL,      0, X_MAX_LIM, 0,    Y_MAX_LIM,      0, Z_MAX_LIM);
    int_eng_L_TEMP  = darray_4d(1, N_MATERIAL,      0, X_MAX_LIM, 0,    Y_MAX_LIM,      0, Z_MAX_LIM);

    r_out_x         = darray_4d(0, X_MAX_LIM,       0, Y_MAX_LIM, 0,    Z_MAX_LIM,      1, N_CELL_FACES);
    r_out_y         = darray_4d(0, X_MAX_LIM,       0, Y_MAX_LIM, 0,    Z_MAX_LIM,      1, N_CELL_FACES);
    r_out_z         = darray_4d(0, X_MAX_LIM,       0, Y_MAX_LIM, 0,    Z_MAX_LIM,      1, N_CELL_FACES);    
    outflux_vol     = darray_4d(0, X_MAX_LIM,       0, Y_MAX_LIM, 0,    Z_MAX_LIM,      1, N_CELL_FACES);
    influx_vol      = darray_4d(-1,X_MAX_LIM+1,     -1,Y_MAX_LIM+1,-1,  Z_MAX_LIM+1,    1, N_CELL_FACES);  
    
    r_out_x_CF      = darray_4d(0, X_MAX_LIM,       0, Y_MAX_LIM, 0,    Z_MAX_LIM,      1, N_CELL_VERTICES);
    r_out_y_CF      = darray_4d(0, X_MAX_LIM,       0, Y_MAX_LIM, 0,    Z_MAX_LIM,      1, N_CELL_VERTICES);
    r_out_z_CF      = darray_4d(0, X_MAX_LIM,       0, Y_MAX_LIM, 0,    Z_MAX_LIM,      1, N_CELL_VERTICES);
    outflux_vol_CF  = darray_4d(0, X_MAX_LIM,       0, Y_MAX_LIM, 0,    Z_MAX_LIM,      1, N_CELL_VERTICES);
    influx_vol_CF   = darray_4d(-1,   X_MAX_LIM+1, -1, Y_MAX_LIM+1,-1,  Z_MAX_LIM+1,    1, N_CELL_VERTICES);



    zero_arrays_4d(
                        xLoLimit,       yLoLimit,       zLoLimit,             
                        xHiLimit,       yHiLimit,       zHiLimit,
                        1,              nMaterials,     11,             
                        advct_xmom_CC,  advct_ymom_CC,  advct_zmom_CC,  
                        advct_int_eng_CC,advct_rho_CC,  int_eng_L_TEMP, 
                        xmom_L_TEMP,    ymom_L_TEMP,    zmom_L_TEMP,
                        mass_L_TEMP,
                        outflux_vol);
    
#if switch_step6_OnOff

/*__________________________________
*   since all of the momentum arrays
*   are defined as mass*velocity
*   we need to convert them into 
*   rho * velocity before we pass that
*   data into the advection operator.
*   So we divide the momentum arrays by the volume
*___________________________________*/
    for (m = 1; m <= nMaterials; m++)
    {     
        for ( i = GC_LO(xLoLimit); i <= GC_HI(xHiLimit); i++)
        {
            for ( j = GC_LO(yLoLimit); j <= GC_HI(yHiLimit); j++)
            {
                for ( k = GC_LO(zLoLimit); k <= GC_HI(zHiLimit); k++)
                {            
                    assert ( Vol_CC[i][j][k] >0.0);

                    vol = Vol_CC[i][j][k];                
                    mass_L_TEMP[m][i][j][k]       = mass_L_CC[m][i][j][k]/vol;
                    xmom_L_TEMP[m][i][j][k]       = xmom_L_CC[m][i][j][k]/vol;
                    ymom_L_TEMP[m][i][j][k]       = ymom_L_CC[m][i][j][k]/vol;
                    zmom_L_TEMP[m][i][j][k]       = zmom_L_CC[m][i][j][k]/vol;
                    int_eng_L_TEMP[m][i][j][k]    = int_eng_L_CC[m][i][j][k]/vol;

                }
            }
        }
    }
/*______________________________________________________________________
* Calculate the advection terms
*_______________________________________________________________________*/

    for (m = 1; m <= nMaterials; m++)
    {   
        putenv("PGPLOT_PLOTTING_ON_OFF=1");
    /*__________________________________
    * Compute stuff that only needs to be
    * calculated only once (i.e. r_out[*],
    * influx_vol, outflux_vol, 
    * influx_counter
    *___________________________________*/  
        advect_preprocess(
                        xLoLimit,       yLoLimit,       zLoLimit,         
                        xHiLimit,       yHiLimit,       zHiLimit,         
                        delX,           delY,           delZ,
                        delt, 
                        uvel_FC,        vvel_FC,        wvel_FC,                             
                        r_out_x,        r_out_y,        r_out_z,
                        r_out_x_CF,     r_out_y_CF,     r_out_z_CF,     
                        outflux_vol,    outflux_vol_CF, 
                        influx_vol,     influx_vol_CF,
                        m );   
    putenv("PGPLOT_PLOTTING_ON_OFF=0");
        /*-------density-------*/                                
         advect_q(
                        xLoLimit,       yLoLimit,       zLoLimit,
                        xHiLimit,       yHiLimit,       zHiLimit,
                        delX,           delY,           delZ,
                        mass_L_TEMP,         
                        r_out_x,        r_out_y,        r_out_z,
                        r_out_x_CF,     r_out_y_CF,     r_out_z_CF,
                        outflux_vol,    outflux_vol_CF, 
                        influx_vol,     influx_vol_CF,
                        advct_rho_CC,   m);

          
        
        /*-----Internal Energy-----*/
         advect_q(
                        xLoLimit,       yLoLimit,       zLoLimit,
                        xHiLimit,       yHiLimit,       zHiLimit,
                        delX,           delY,           delZ,
                        int_eng_L_TEMP,         
                        r_out_x,        r_out_y,        r_out_z,
                        r_out_x_CF,     r_out_y_CF,     r_out_z_CF,
                        outflux_vol,    outflux_vol_CF, 
                        influx_vol,     influx_vol_CF,
                        advct_int_eng_CC,m);
 

        /*-------x-momentum------*/
         advect_q(
                        xLoLimit,       yLoLimit,       zLoLimit,
                        xHiLimit,       yHiLimit,       zHiLimit,
                        delX,           delY,           delZ,
                        xmom_L_TEMP,         
                        r_out_x,        r_out_y,        r_out_z,
                        r_out_x_CF,     r_out_y_CF,     r_out_z_CF,
                        outflux_vol,    outflux_vol_CF, 
                        influx_vol,     influx_vol_CF,
                        advct_xmom_CC,  m); 
       
        /*-------y-momentum------*/
         putenv("PGPLOT_PLOTTING_ON_OFF=1");
         advect_q(
                        xLoLimit,       yLoLimit,       zLoLimit,
                        xHiLimit,       yHiLimit,       zHiLimit,
                        delX,           delY,           delZ,
                        ymom_L_TEMP,         
                        r_out_x,        r_out_y,        r_out_z,
                        r_out_x_CF,     r_out_y_CF,     r_out_z_CF,
                        outflux_vol,    outflux_vol_CF, 
                        influx_vol,     influx_vol_CF,
                        advct_ymom_CC,  m);


    /*  advect_q(
                        xLoLimit,       yLoLimit,       zLoLimit,
                        xHiLimit,       yHiLimit,       zHiLimit,
                        delX,           delY,           delZ,
                        zmom_L_TEMP,         
                        r_out_x,        r_out_y,        r_out_z,
                        r_out_x_CF,     r_out_y_CF,     r_out_z_CF,
                        outflux_vol,    outflux_vol_CF, 
                        influx_vol,     influx_vol_CF,
                        advct_zmom_CC,  m); */

    }
                    

#endif

#if switch_step7_OnOff 
putenv("PGPLOT_PLOTTING_ON_OFF=1");
/*__________________________________
* Now advance in time
*___________________________________*/
    for (m = 1; m <= nMaterials; m++)
    { 
        for ( i = xLoLimit; i <= xHiLimit; i++)
        {
            for ( j = yLoLimit; j <= yHiLimit; j++)
            {
                for ( k = zLoLimit; k <= zHiLimit; k++)
                {            
                    assert ( Vol_CC[i][j][k] >0.0);

                    /*__________________________________
                    * update x, y, z components of momentum
                    *___________________________________*/
                    xmom_CC[m][i][j][k] = ( xmom_L_CC[m][i][j][k] + advct_xmom_CC[m][i][j][k] );
                    ymom_CC[m][i][j][k] = ( ymom_L_CC[m][i][j][k] + advct_ymom_CC[m][i][j][k] );
                    zmom_CC[m][i][j][k] = ( zmom_L_CC[m][i][j][k] + advct_zmom_CC[m][i][j][k] );  

                    /*__________________________________
                    *   Update the internal energy
                    *___________________________________*/
                    int_eng_CC[m][i][j][k] = ( int_eng_L_CC[m][i][j][k] + advct_int_eng_CC[m][i][j][k] );
                    
                    /*__________________________________
                    *  update density You must do this last
                    *   so you don't overwrite the rho_CC
                    *___________________________________*/

                   rho_CC[m][i][j][k] = (mass_L_CC[m][i][j][k] + 
                                         advct_rho_CC[m][i][j][k] )/ Vol_CC[i][j][k];                             

                }
            }
        }
    }
#endif
/*______________________________________________________________________
*   DEBUGGING AND STOP WATCH INFORMATION
*_______________________________________________________________________*/
#if switchDebug_advect_and_advance_in_time
    should_I_write_output = *getenv("SHOULD_I_WRITE_OUTPUT");
    if ( should_I_write_output == '1')
    {
        #define switchInclude_advect_and_advance_in_time 1
        #include "debugcode.i"
        #undef switchInclude_advect_and_advance_in_time 
    }                  
#endif 
       
#if sw_advect_and_advance_in_time
    stopwatch("time_advance function",start);
#endif


/*__________________________________
* Free local memory
*___________________________________*/
   free_darray_4d( advct_xmom_CC,   1, N_MATERIAL,      0, X_MAX_LIM, 0,    Y_MAX_LIM,      0, Z_MAX_LIM);
   free_darray_4d( advct_ymom_CC,   1, N_MATERIAL,      0, X_MAX_LIM, 0,    Y_MAX_LIM,      0, Z_MAX_LIM);
   free_darray_4d( advct_zmom_CC,   1, N_MATERIAL,      0, X_MAX_LIM, 0,    Y_MAX_LIM,      0, Z_MAX_LIM);
   free_darray_4d( mass_L_TEMP,     1, N_MATERIAL,      0, X_MAX_LIM, 0,    Y_MAX_LIM,      0, Z_MAX_LIM);
   free_darray_4d( xmom_L_TEMP,     1, N_MATERIAL,      0, X_MAX_LIM, 0,    Y_MAX_LIM,      0, Z_MAX_LIM);
   free_darray_4d( ymom_L_TEMP,     1, N_MATERIAL,      0, X_MAX_LIM, 0,    Y_MAX_LIM,      0, Z_MAX_LIM);
   free_darray_4d( zmom_L_TEMP,     1, N_MATERIAL,      0, X_MAX_LIM, 0,    Y_MAX_LIM,      0, Z_MAX_LIM);
   free_darray_4d( int_eng_L_TEMP,  1, N_MATERIAL,      0, X_MAX_LIM, 0,    Y_MAX_LIM,      0, Z_MAX_LIM);

   free_darray_4d( advct_rho_CC,    1, N_MATERIAL,      0, X_MAX_LIM, 0,    Y_MAX_LIM,      0, Z_MAX_LIM);
   free_darray_4d( advct_int_eng_CC,1, N_MATERIAL,      0, X_MAX_LIM, 0,    Y_MAX_LIM,      0, Z_MAX_LIM);
 
   free_darray_4d( r_out_x,         0, X_MAX_LIM,   0, Y_MAX_LIM,    0, Z_MAX_LIM,   1, N_CELL_FACES);
   free_darray_4d( r_out_y,         0, X_MAX_LIM,   0, Y_MAX_LIM,    0, Z_MAX_LIM,   1, N_CELL_FACES);
   free_darray_4d( r_out_z,         0, X_MAX_LIM,   0, Y_MAX_LIM,    0, Z_MAX_LIM,   1, N_CELL_FACES);
   free_darray_4d(outflux_vol,      0, X_MAX_LIM,   0, Y_MAX_LIM,    0, Z_MAX_LIM,   1, N_CELL_FACES);
   free_darray_4d(influx_vol,       -1,X_MAX_LIM+1,-1, Y_MAX_LIM+1, -1, Z_MAX_LIM+1, 1, N_CELL_FACES);
   
   free_darray_4d( r_out_x_CF,      0, X_MAX_LIM,   0, Y_MAX_LIM,    0, Z_MAX_LIM,   1, N_CELL_VERTICES);
   free_darray_4d( r_out_y_CF,      0, X_MAX_LIM,   0, Y_MAX_LIM,    0, Z_MAX_LIM,   1, N_CELL_VERTICES);
   free_darray_4d( r_out_z_CF,      0, X_MAX_LIM,   0, Y_MAX_LIM,    0, Z_MAX_LIM,   1, N_CELL_VERTICES);
   free_darray_4d(outflux_vol_CF,   0, X_MAX_LIM,   0, Y_MAX_LIM,    0, Z_MAX_LIM,   1, N_CELL_VERTICES);
   free_darray_4d(influx_vol_CF,    -1,X_MAX_LIM+1,-1, Y_MAX_LIM+1, -1, Z_MAX_LIM+1, 1, N_CELL_VERTICES);
   
/*__________________________________
*   Quite fullwarn remarks in a way that
*   is compiler independent
*___________________________________*/
    QUITE_FULLWARN(Vol_L_CC[1][0][0][0]); 
    QUITE_FULLWARN(rho_L_CC[1][0][0][0]);
    should_I_write_output = should_I_write_output;

}
/*STOP_DOC*/






























































#if 0 
/* 
 ======================================================================*/
#include <assert.h>
#include <time.h>
#include <stdio.h>
#include <sys/types.h>
#include <math.h>
#include <stdlib.h>              
#include "functionDeclare.h"
#include "parameters.h"
#include "switches.h"
#include "nrutil+.h"
#include "macros.h"

/* --------------------------------------------------------------------- 

 Function:  doubleCheckConservationLaws--checks to see if mass, momentum and energy are conserved over the whole domain  
 Filename:  timeadvanced.c 

 Purpose:
   This function calculates the time n+1, density,
   internal energy and momentum over the entire domain.
   
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       09/??/99     

Need to include kinetic energy 
 ---------------------------------------------------------------------  */
void doubleCheckConservationLaws(  
        int     xLoLimit,               /* x-array lower limit              */
        int     yLoLimit,               /* y-array lower limit              */
        int     zLoLimit,               /* z-array lower limit              */
        int     xHiLimit,               /* x-array upper limit              */
        int     yHiLimit,               /* y-array upper limit              */
        int     zHiLimit,               /* z-array upper limit              */
        double  delX,                   /* distance/cell, xdir              (INPUT) */
        double  delY,                   /* distance/cell, ydir              (INPUT) */
        double  delZ,                   /* distance/cell, zdir              (INPUT) */ 
        double  ***Vol_CC,              /* cell-centered volume             (INPUT) */
        double  ****rho_CC,             /* cell-centered density            (OUPUT) */
        double  ****uvel_CC,            /* u cell-centered velocity         (INPUT) */
        double  ****vvel_CC,            /* v cell-centered velocity         (INPUT) */
        double  ****wvel_CC,            /* w cell-centered velocity         (INPUT) */
        double  ****xmom_CC,            /* cell-centered x-momentum         (OUPUT) */
        double  ****ymom_CC,            /* cell-centered x-momentum         (OUPUT) */
        double  ****zmom_CC,            /* cell-centered x-momentum         (OUPUT) */
        double  ****Vol_L_CC,           /* Lagrangian cell-centered volume  (INPUT) */
        double  ****rho_L_CC,           /* Lagrangian cell-centered density (INPUT) */
        double  ****mass_L_CC,          /* Lagrangian cell-centered mass    (INPUT) */
        double  ****xmom_L_CC,          /* Lagrangian cell-centered momentum(INPUT) */
        double  ****ymom_L_CC,          /* Lagrangian cell-centered momentum(INPUT) */
        double  ****zmom_L_CC,          /* Lagrangian cell-centered momentum(INPUT) */
        double  ****int_eng_CC,         /* internal energy                  (OUPUT) */
        double  ****int_eng_L_CC,       /* Lagrangian CC internal energy    (INPUT) */
        double  ******uvel_FC,          /*  u-face-centered velocity        (INPUT) */
        double  ******vvel_FC,          /*  v-face-centered velocity        (INPUT) */
        double  ******wvel_FC,          /* w face-centered velocity         (INPUT) */
        double  delt,                   /* delta t                          (INPUT) */
        int     nMaterials      )
/* Local Definitions________________________________________________________*/
{
    int     i, j, k,m,                  /*   loop indices  locators         */
            ***influx_counter,          /* volume segments flowing from one */
                                        /* cell to another                  */
            wall, wallLo, wallHi,       /* wall upper and lower indices     */ 
            xLo, xHi,
            yLo, yHi,
            zLo, zHi;           
    
    double  vol,                        /* Temporary variable               */
            *sum_xmom_CC,               /* sum over the entire domain of the*/   
            *sum_ymom_CC,               /* conserved quantities             */
            *sum_zmom_CC,               /*----------//----------------------*/
            *sum_int_eng_CC,            /*----------//----------------------*/
            *sum_mass_CC,               /*----------//----------------------*/
            ****advct_xmom_CC,          /* Advected momemtum                */
            ****advct_ymom_CC,
            ****advct_zmom_CC,
            ****mass_L_TEMP,
            ****xmom_L_TEMP,            /* temporary variables              */
            ****ymom_L_TEMP,
            ****zmom_L_TEMP,
            ****int_eng_L_TEMP,
            ****advct_rho_CC,
            ****advct_int_eng_CC,       /* Advected interal energy          */            
            *****r_out_x,               /* x-dir centroid array (i,j,k,vol  */ 
            *****r_out_y,               /* y-dir centroid array             */                   
            *****r_out_z,               /* z-dir centroid array             */                 
            ****outflux_vol,         /* array containing the size of each*/
                                        /* of the outflux volumes           */ 
                                        /* (i,j,k,vol)                      */                      
            ****influx_vol;          /* array containing the size of each*/
                                        /* of the influx volumes            */ 
                                        /* (i,j,k,counter)                  */    
/*__________________________________
* Check that the inputs are reasonable
*___________________________________*/
    assert ( xLoLimit > 0 && xHiLimit < X_MAX_LIM);
    assert ( yLoLimit > 0 && yHiLimit < Y_MAX_LIM);
    assert ( zLoLimit > 0 && zHiLimit < Z_MAX_LIM);
/*__________________________________
* Allocate memory for local arrays
*___________________________________*/

    sum_xmom_CC     = dvector_nr(1, N_MATERIAL);
    sum_ymom_CC     = dvector_nr(1, N_MATERIAL);
    sum_zmom_CC     = dvector_nr(1, N_MATERIAL);
    sum_int_eng_CC  = dvector_nr(1, N_MATERIAL);
    sum_mass_CC     = dvector_nr(1, N_MATERIAL);
      
    advct_xmom_CC   = darray_4d(1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    advct_ymom_CC   = darray_4d(1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    advct_zmom_CC   = darray_4d(1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    advct_rho_CC    = darray_4d(1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    advct_int_eng_CC= darray_4d(1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);

    mass_L_TEMP     = darray_4d(1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    xmom_L_TEMP     = darray_4d(1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    ymom_L_TEMP     = darray_4d(1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    zmom_L_TEMP     = darray_4d(1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    int_eng_L_TEMP  = darray_4d(1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);

    influx_counter  = iarray_3d(-1, X_MAX_LIM+1,    -1, Y_MAX_LIM+1,-1, Z_MAX_LIM+1);
    r_out_x         = darray_5d(0, X_MAX_LIM,       0, Y_MAX_LIM, 0,    Z_MAX_LIM, 1, N_MATERIAL,  1, N_OUTFLOW_DEL_V);
    r_out_y         = darray_5d(0, X_MAX_LIM,       0, Y_MAX_LIM, 0,    Z_MAX_LIM, 1, N_MATERIAL,  1, N_OUTFLOW_DEL_V);
    r_out_z         = darray_5d(0, X_MAX_LIM,       0, Y_MAX_LIM, 0,    Z_MAX_LIM, 1, N_MATERIAL,  1, N_OUTFLOW_DEL_V);    
    outflux_vol  = darray_4d(0, X_MAX_LIM,       0, Y_MAX_LIM, 0,    Z_MAX_LIM, 1, N_OUTFLOW_DEL_V);
    influx_vol   = darray_4d(-1,   X_MAX_LIM+1, -1, Y_MAX_LIM+1,-1,  Z_MAX_LIM+1,               1, N_INFLOW_DEL_V);    
/*__________________________________
*   Determine the looping indices
*   for multidimensional problems
*___________________________________*/
#if (N_DIMENSIONS == 1)  
    wallLo = LEFT;  wallHi = RIGHT;
#endif

#if (N_DIMENSIONS == 2) 
    wallLo = TOP;   wallHi = LEFT;
#endif
#if (N_DIMENSIONS == 3) 
    wallLo = TOP;   wallHi = BACK;
#endif

   /*__________________________________
   * zero the locally defined arrays
   * I don't need to zero the entire
   * domain just each wall.  However,
   * it is simpler this way
   *___________________________________*/
    zero_arrays_4d(
                        xLoLimit,           yLoLimit,           zLoLimit,             
                        xHiLimit,           yHiLimit,           zHiLimit,
                        1,                  nMaterials,         11,             
                        advct_xmom_CC,      advct_ymom_CC,      advct_zmom_CC,  
                        advct_int_eng_CC,   advct_rho_CC,       int_eng_L_TEMP, 
                        xmom_L_TEMP,        ymom_L_TEMP,        zmom_L_TEMP,
                        mass_L_TEMP,
                        outflux_vol);

    zero_arrays_5d(
                        xLoLimit,           yLoLimit,           zLoLimit,             
                        xHiLimit,           yHiLimit,           zHiLimit,
                        1,                  nMaterials,         1,
                        N_OUTFLOW_DEL_V,                    3,             
                        r_out_x,            r_out_y,            r_out_z);
/*__________________________________
*   since all of the conserved quantities
*   are defined as q*Volume
*   we need to convert them mass, momentum
*   and internal energy densities before we pass that
*   data into the advection operator.
*   So we divide the Lagrangian arrays by the volume
*___________________________________*/ 
    for (m = 1; m <= nMaterials; m++)
    {    
        for ( i = GC_LO(xLoLimit); i <= GC_HI(xHiLimit); i++)
        {
            for ( j = GC_LO(yLoLimit); j <= GC_HI(yHiLimit); j++)
            {
                for ( k = GC_LO(zLoLimit); k <= GC_HI(zHiLimit); k++)
                {            
                    assert ( Vol_CC[i][j][k] >0.0);

                    vol = Vol_CC[i][j][k];                
                    mass_L_TEMP[m][i][j][k]       = mass_L_CC[m][i][j][k]/vol;
                    xmom_L_TEMP[m][i][j][k]       = xmom_L_CC[m][i][j][k]/vol;
                    ymom_L_TEMP[m][i][j][k]       = ymom_L_CC[m][i][j][k]/vol;
                    zmom_L_TEMP[m][i][j][k]       = zmom_L_CC[m][i][j][k]/vol;
                    int_eng_L_TEMP[m][i][j][k]    = int_eng_L_CC[m][i][j][k]/vol;
                }
            }
        }
    }
/*______________________________________________________________________
*   For each material advect each of the conserved quantities
*   Note:   You only need to advect on each of the walls.  I'm doing WAY too
*           much work here.  I only need to advect on each of the walls.
*           
*_______________________________________________________________________*/
    for(m = 1; m <=nMaterials; m++)
    { 
        /*______________________________________________________________________
        * Calculate the advection terms
        *_______________________________________________________________________*/

        /*__________________________________
        * Compute stuff that only needs to be
        * calculated only once (i.e. r_out[*],
        * influx_vol, outflux_vol, 
        * influx_counter
        *___________________________________*/  
            advect_preprocess(
                            xLoLimit,       yLoLimit,       zLoLimit,
                            xHiLimit,       yHiLimit,       zHiLimit,
                            delX,           delY,           delZ,
                            delt,             
                            uvel_CC,        vvel_CC,        wvel_CC,             
                            uvel_FC,        vvel_FC,        wvel_FC,    
                            r_out_x,        r_out_y,        r_out_z,     
                            outflux_vol,  influx_counter, influx_vol,
                            m );   

            /*-------density-------*/                                
             advect_q(
                            xLoLimit,       yLoLimit,       zLoLimit,
                            xHiLimit,       yHiLimit,       zHiLimit,
                            delX,           delY,           delZ,
                            mass_L_TEMP,
                            uvel_FC,        vvel_FC,        wvel_FC,         
                            r_out_x,        r_out_y,        r_out_z,
                            outflux_vol, influx_counter, influx_vol,
                            advct_rho_CC,   m);   

            /*-----Internal Energy-----*/
            advect_q(
                            xLoLimit,       yLoLimit,       zLoLimit,
                            xHiLimit,       yHiLimit,       zHiLimit,
                            delX,           delY,           delZ,
                            int_eng_L_TEMP,
                            uvel_FC,        vvel_FC,        wvel_FC, 
                            r_out_x,        r_out_y,        r_out_z,
                            outflux_vol, influx_counter, influx_vol,
                            advct_int_eng_CC,m);

            /*-------x-momentum------*/
            advect_q(
                            xLoLimit,       yLoLimit,       zLoLimit,
                            xHiLimit,       yHiLimit,       zHiLimit,
                            delX,           delY,           delZ,
                            xmom_L_TEMP,
                            uvel_FC,        vvel_FC,        wvel_FC, 
                             r_out_x,        r_out_y,        r_out_z,
                            outflux_vol, influx_counter, influx_vol,
                            advct_xmom_CC,  m); 
            /*-------y-momentum------*/
             advect_q(
                            xLoLimit,       yLoLimit,       zLoLimit,
                            xHiLimit,       yHiLimit,       zHiLimit,
                            delX,           delY,           delZ,
                            ymom_L_TEMP,
                            uvel_FC,        vvel_FC,        wvel_FC,      
                            r_out_x,        r_out_y,        r_out_z,
                            outflux_vol, influx_counter, influx_vol,
                            advct_ymom_CC,  m);


        /*      advect_q(
                            xLoLimit,            yLoLimit,            zLoLimit,
                            xHiLimit,            yHiLimit,            zHiLimit,
                            delX,           delY,           delZ,
                            zmom_L_TEMP,
                            uvel_FC,        vvel_FC,        wvel_FC,    
                             r_out_x,        r_out_y,        r_out_z,
                            outflux_vol, influx_counter, influx_vol,
                            advct_zmom_CC,  m); */    
        }
        
        /*______________________________________________________________________
        *       Need to straighten this out.
        *_______________________________________________________________________*/
        
       sum_xmom_CC[m]     = 0.0;
        sum_ymom_CC[m]     = 0.0;
        sum_zmom_CC[m]     = 0.0;
        sum_int_eng_CC[m]  = 0.0;
        sum_mass_CC[m]     = 0.0;        
        
        
        
        
        for(wall = wallLo; wall <= wallHi; wall ++)
        {
           /*__________________________________
           * Defined the looping indices for each
           *   edge
           *___________________________________*/
           if( wall == LEFT )   /* includes upper and lower corner cells        */ 
           {
                xLo = xLoLimit;     xHi = xLoLimit;
                yLo = yLoLimit;     yHi = yHiLimit; 
                zLo = zLoLimit;     zHi = zHiLimit;
            }
           if( wall == RIGHT )  /* includes upper and lower corner cells        */
           {
                xLo = xHiLimit;     xHi = xHiLimit;
                yLo = yLoLimit;     yHi = yHiLimit; 
                zLo = zLoLimit;     zHi = zHiLimit;
            }
           if( wall == TOP )    /* left and right corner cells NOT included     */
           {
                xLo = xLoLimit + 1; xHi = xHiLimit - 1;
                yLo = yHiLimit;     yHi = yHiLimit; 
                zLo = zLoLimit;     zHi = zHiLimit;
            }
           if( wall == BOTTOM )/* left and right corner cells NOT included     */ 
           {
                xLo = xLoLimit + 1; xHi = xHiLimit - 1;
                yLo = yLoLimit;     yHi = yLoLimit; 
                zLo = zLoLimit;     zHi = zHiLimit;
            }
           if( wall == FRONT ) 
           {
         /*     xLo = xLoLimit + N_GHOSTCELLS; xHi = xHiLimit - N_GHOSTCELLS;
                yLo = yLoLimit + N_GHOSTCELLS; yHi = yHiLimit - N_GHOSTCELLS; 
                zLo = zLoLimit;     zHi = zLoLimit; */
            }
           if( wall == BACK ) 
           {
        /*      xLo = xLoLimit + N_GHOSTCELLS; xHi = xHiLimit - N_GHOSTCELLS;
                yLo = yLoLimit + N_GHOSTCELLS; yHi = yHiLimit - N_GHOSTCELLS; 
                zLo = zHiLimit;     zHi = zHiLimit; */
            }                            
            sum_advct_xmom      = 0.0;
            sum_advct_ymom      = 0.0;
            sum_advct_zmom      = 0.0;
            sum_advct_int_eng   = 0.0;
            sum_advct_rho       = 0.0;
            /*__________________________________
            * Now sum the various contributions
            *___________________________________*/
            for ( i = xLo; i <= xHi; i++)
            {
                for ( j = yLo; j <= yHi; j++)
                {
                    for ( k = zLo; k <= zHi; k++)
                    {            
                        assert ( Vol_CC[i][j][k] >0.0);
                        /*__________________________________
                        * x, y, z components of momentum
                        *___________________________________*/
                        sum_advct_xmom      = sum_advct_xmom    + advct_xmom_CC[m][i][j][k];
                        sum_advct_ymom      = sum_advct_ymom    + advct_ymom_CC[m][i][j][k];
                        sum_advct_xmom      = sum_advct_zmom    + advct_zmom_CC[m][i][j][k];
                        /*__________________________________
                        *   Internal energy
                        *___________________________________*/
                        sum_advct_int_eng   = sum_advct_int_eng + advct_int_eng_CC[m][i][j][k];
                        /*__________________________________
                        *  Mass
                        *___________________________________*/
                        sum_advct_rho       = sum_advct_rho     + advct_rho_CC[m][i][j][k];
                    }
                }
            }
        }       /* wall loop    */   
        /*__________________________________
        * Sum up the lagrangian contributions
        * on the interior of the domain
        *___________________________________*/    
        for ( i = xLoLimit; i <= xHiLimit; i++)
        {
            for ( j = yLoLimit; j <= yHiLimit; j++)
            {
                for ( k = zLoLimit; k <= zHiLimit; k++)
                {                      
                    sum_xmom_L_CC[m] = sum_xmom_L_CC[m] + xmom_L_CC[m][i][j][k];
                    sum_ymom_L_CC[m] = sum_ymom_L_CC[m] + ymom_L_CC[m][i][j][k];
                    sum_zmom_L_CC[m] = sum_zmom_L_CC[m] + zmom_L_CC[m][i][j][k];  
                }
            }
        }        
        
    }           /* material loop*/
/*______________________________________________________________________
*  Output info here.
*_______________________________________________________________________*/
    for (m = 1; m <= nMaterials; m++)
    {   fprintf(stderr,"Material: \t %i\n",m);
        fprintf(stderr,"sum_xmom_CC \t %f  \t sum_ymom_CC \t %f \t sum_zmom_CC \t %f \n",sum_xmom_CC[m], sum_ymom_CC[m], sum_zmom_CC[m]);
        fprintf(stderr,"sum_int_eng_CC \t %f\n",sum_int_eng_CC[m]);
        fprintf(stderr,"sum_mass_CC \t %f\n",sum_mass_CC[m]);

    }
/*__________________________________
* Free local memory
*___________________________________*/
   free_dvector_nr( sum_xmom_CC,       1, N_MATERIAL);
   free_dvector_nr( sum_ymom_CC,       1, N_MATERIAL);
   free_dvector_nr( sum_zmom_CC,       1, N_MATERIAL);
   free_dvector_nr( sum_int_eng_CC,    1, N_MATERIAL);
   free_dvector_nr( sum_mass_CC,       1, N_MATERIAL);
   
   free_darray_4d( advct_xmom_CC,   1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
   free_darray_4d( advct_ymom_CC,   1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
   free_darray_4d( advct_zmom_CC,   1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
   free_darray_4d( mass_L_TEMP,     1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
   free_darray_4d( xmom_L_TEMP,     1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
   free_darray_4d( ymom_L_TEMP,     1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
   free_darray_4d( zmom_L_TEMP,     1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
   free_darray_4d( int_eng_L_TEMP,  1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);

   free_darray_4d( advct_rho_CC,    1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
   free_darray_4d( advct_int_eng_CC,1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
  
   free_darray_4d( r_out_x,         0, X_MAX_LIM, 0,    Y_MAX_LIM,  0,  Z_MAX_LIM,  1, N_CELL_FACES);
   free_darray_4d( r_out_y,         0, X_MAX_LIM, 0,    Y_MAX_LIM,  0,  Z_MAX_LIM,  1, N_CELL_FACES);
   free_darray_4d( r_out_z,         0, X_MAX_LIM, 0,    Y_MAX_LIM,  0,  Z_MAX_LIM,  1, N_CELL_FACES);
   free_darray_4d( outflux_vol,     0, X_MAX_LIM, 0,    Y_MAX_LIM,  0,  Z_MAX_LIM,  1, N_CELL_FACES);
   free_darray_4d( influx_vol,     -1, X_MAX_LIM+1,-1, Y_MAX_LIM+1,-1, Z_MAX_LIM+1,    1, N_CELL_FACES);
   
/*__________________________________
*   Quite fullwarn remarks in a way that
*   is compiler independent
*___________________________________*/
    QUITE_FULLWARN(Vol_L_CC[1][0][0][0]); 
    rho_L_CC = rho_L_CC; 
    QUITE_FULLWARN(xmom_CC[1][0][0][0]);
    QUITE_FULLWARN(ymom_CC[1][0][0][0]);
    QUITE_FULLWARN(zmom_CC[1][0][0][0]);    
    QUITE_FULLWARN(rho_CC[1][0][0][0]);
    QUITE_FULLWARN(int_eng_CC[1][0][0][0]);

}
/*STOP_DOC*/

#endif
