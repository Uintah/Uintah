    

/* 
 ======================================================================*/
#include <math.h>
#include <stdio.h>
#include "functionDeclare.h"
#include "parameters.h"
#include "switches.h"
#include "nrutil+.h"
#include "macros.h"

/*
 Function:  find_delta_time_based_on_FC_vel--MISC: Computes the time step for the next cycle. 
 Filename:  commonFunctions.c

 Purpose:
   This function calculates delta time based on the Courant number < 1.0
   Where the Courant number is u[i][j][k] Delta{time}/Delta{x}
 Note:
    Delt is computed using face-centered velocities.
  
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       03.20.00    
 Refererences:
    "Computational Fluid Mechanics and Heat Transfer" 2nd edition
    pg 56
 ---------------------------------------------------------------------  */
 
 void find_delta_time_based_on_FC_vel(
        int     xLoLimit,               /* x-array lower limit              */
        int     yLoLimit,               /* y-array lower limit              */
        int     zLoLimit,               /* z-array lower limit              */
        int     xHiLimit,               /* x-array upper limit              */
        int     yHiLimit,               /* y-array upper limit              */
        int     zHiLimit,               /* z-array upper limit              */
        double  *delt,                  /* delta t                          (OUTPUT)*/
        double  *delt_limits,           /* delt_limits[1] = delt minimum    (INPUT) */
                                        /* delt_limits[2] = delt_maximum    (INPUT) */
        double  delX,                   /* distance/cell, xdir              (INPUT) */
        double  delY,                   /* distance/cell, ydir              (INPUT) */
        double  delZ,                   /* distance/cell, zdir              (INPUT) */
                                        /* (*)vel_CC(x,y,z,material         */
        double  ******uvel_FC,          /* u-face-centered velocity         (INPUT) */
        double  ******vvel_FC,          /* v-face-centered velocity         (INPUT) */
        double  ******wvel_FC,          /* v-face-centered velocity         (INPUT) */
        double  ****speedSound,         /* speed of sound cell cell-center  (INPUT) */
        double  CFL,                     /* CFL number                       (INPUT) */
        int     nMaterials          )
{
    int     i, j, k,m, f,                /*   loop indices  locators        */
            faceLo, faceHi,
            xLo,    xHi,
            yLo,    yHi,
            zLo,    zHi;
    double  A, B,
            fudge_factor,
            delt_stability,             /* based on stability               */
            delt_CFL;                   /* based on the CFL number          */
/*START_DOC*/
/*__________________________________
*   Determine the upper and lower 
*   looping indices for the faces
*___________________________________*/
#if (N_DIMENSIONS == 1)  
        faceLo = LEFT;  faceHi = RIGHT;
#endif

#if (N_DIMENSIONS == 2) 
        faceLo = TOP;   faceHi = LEFT;
#endif
#if (N_DIMENSIONS == 3) 
        faceLo = TOP;   faceHi = BACK;
#endif

/*__________________________________
* Initialize variables
* and calculate the looping indicies to 
* include the ghostcells
*___________________________________*/
    *delt           = BIG_NUM;
    delt_stability  = BIG_NUM;
    delt_CFL        = BIG_NUM;
    fudge_factor= .95;
    
    xLo = GC_LO(xLoLimit);
    yLo = GC_LO(yLoLimit);
    zLo = GC_LO(zLoLimit);
    xHi = GC_HI(xHiLimit);
    yHi = GC_HI(yHiLimit);
    zHi = GC_HI(zHiLimit);
/*______________________________________________________________________
* Now calculate the next time step based on the CFL restraint
* Note this 
*_______________________________________________________________________*/
    for ( m = 1; m <= nMaterials; m ++)
    {
        for ( i = xLo; i <= xHi; i++)
        {
            for ( j = yLo; j <= yHi; j++)
            {
                for ( k = zLo; k <= zHi; k++)
                {  
                    for(f = faceLo; f <= faceHi; f++)
                    {
                        /*__________________________________
                        *   Based on the convective velocity
                        *___________________________________*/  
                    #if (compute_delt_based_on_velocity == 1)
                        A   = fudge_factor*CFL*delX/fabs(*uvel_FC[i][j][k][f][m] + SMALL_NUM); 
                        B   = fudge_factor*CFL*delY/fabs(*vvel_FC[i][j][k][f][m] + SMALL_NUM);
                        delt_CFL = DMIN(A, delt_CFL);
                        delt_CFL = DMIN(B, delt_CFL);
                    #endif
                        /*__________________________________
                        *   Based on the speed of sound
                        *___________________________________*/
                    #if (compute_delt_based_on_velocity == 2)
                        A   = fudge_factor*CFL*delX/fabs(speedSound[m][i][j][k] + SMALL_NUM); 
                        B   = fudge_factor*CFL*delY/fabs(speedSound[m][i][j][k] + SMALL_NUM);
                        delt_CFL = DMIN(A, delt_CFL);
                        delt_CFL = DMIN(B, delt_CFL);
                    #endif

                        /*__________________________________
                        *   Based on the speed of sound AND
                        *   the convective velocity
                        *___________________________________*/
                    #if (compute_delt_based_on_velocity == 3)
                        A   = fudge_factor*CFL*delX/( speedSound[m][i][j][k] + fabs(*uvel_FC[i][j][k][f][m]) + SMALL_NUM); 
                        B   = fudge_factor*CFL*delY/( speedSound[m][i][j][k] + fabs(*vvel_FC[i][j][k][f][m]) + SMALL_NUM);
                        delt_CFL = DMIN(A, delt_CFL);
                        delt_CFL = DMIN(B, delt_CFL);
                    #endif
                        /*__________________________________
                        * based on stability requirements
                        * see references
                        *___________________________________*/
                        A   = fudge_factor * 0.5 * pow(delX, 2.0)/fabs(*uvel_FC[i][j][k][f][m]); 
                        B   = fudge_factor * 0.5 * pow(delY, 2.0)/fabs(*vvel_FC[i][j][k][f][m]);
                    
                        delt_stability = DMIN(A, delt_stability);
                        delt_stability = DMIN(B, delt_stability);
                    }
                }
            }
        }
    }
    
    *delt = DMIN(delt_stability, delt_CFL);
    
 /*`==========TESTING========== Not sure whether to used delt_stability or delt_cfl*/ 
 *delt = delt_CFL;
 /*==========TESTING==========`*/
    /*__________________________________
    *   Print some error messages
    *___________________________________*/
    if ( *delt < delt_limits[1] )
    {
        fprintf(stderr,"Current delt %f , minimum allowable delt: %f\n",
        *delt, delt_limits[1]);
        Message(1,"Warning:","The current time step is < than the allowable minimum",
           "specifed in the input file, now setting delt = delt_min.");
        *delt = delt_limits[1];
    }
    
    if ( *delt > delt_limits[2] )
    {
        fprintf(stderr,"Current delt %f , max. allowable delt: %f\n",
        *delt, delt_limits[2]);
        Message(0,"Warning:","The current time step is > than the allowable max",
           "now setting the delt = delt_maximum");
        *delt = delt_limits[2];
    }   

/*______________________________________________________________________
*   DEBUGGING INFORMATION
*_______________________________________________________________________*/ 
#if switchDebug_find_delta_time                                          
        fprintf(stderr," ______________________________________________\n");
        fprintf(stderr,"find_delta_time\n");
        fprintf(stderr,"delta_time based on CFL=%f \t %f \n",CFL, delt_CFL);
        fprintf(stderr,"delta_time based on stability  \t%f \n",delt_stability);        
        fprintf(stderr,"The new time step is \t\t%f\n",*delt);
        fprintf(stderr," ______________________________________________\n");

#endif 
/*__________________________________
*   Quite fullwarn remarks in a way that
*   is compiler independent
*___________________________________*/
    QUITE_FULLWARN(delZ);   QUITE_FULLWARN(wvel_FC[0][0][0][1][1]);
    speedSound  = speedSound;
    CFL         = CFL;

}
/*STOP_DOC*/
/* 
 ======================================================================*/
#include <math.h>
#include <stdio.h>
#include "functionDeclare.h"
#include "parameters.h"
#include "switches.h"
#include "nrutil+.h"
#include "macros.h"

/*
 Function:  find_delta_time_based_on_CC_vel--MISC: Computes the time step for the next cycle. 
 Filename:  commonFunctions.c
  

 Purpose:
   This function calculates delta time based on the Courant number < 1.0
   Where the Courant number is u[i][j][k] Delta{time}/Delta{x}
  
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       05/12/99    
 Refererences:
    "Computational Fluid Mechanics and Heat Transfer" 2nd edition
    pg 56
    
 Implementation Note:
     When testing Sod's shock tube problem it was discovered that the solution
     had to stabilize over several iterations before the computed velocity field 
     was close to the actual solution.  Subsequently, during the first few 
     iterations the computed delt was significantly larger than what
     it should be.  to get around this during the first N_ITERATIONS_TO_STABILIZE
     the CFL linearly increases  from 1/N_ITERATIONS_TO_STABILIZE to 
     CFL from the input file over N_ITERATIONS_TO_STABILIZE.
 ---------------------------------------------------------------------  */
 
 void find_delta_time_based_on_CC_vel(
        int     xLoLimit,               /* x-array lower limit              */
        int     yLoLimit,               /* y-array lower limit              */
        int     zLoLimit,               /* z-array lower limit              */
        int     xHiLimit,               /* x-array upper limit              */
        int     yHiLimit,               /* y-array upper limit              */
        int     zHiLimit,               /* z-array upper limit              */
        double  *delt,                  /* delta t                          (OUTPUT)*/
        double  *delt_limits,           /* delt_limits[1] = delt minimum    (INPUT) */
                                        /* delt_limits[2] = delt_maximum    (INPUT) */
        double  delX,                   /* distance/cell, xdir              (INPUT) */
        double  delY,                   /* distance/cell, ydir              (INPUT) */
        double  delZ,                   /* distance/cell, zdir              (INPUT) */
                                        /* (*)vel_CC(x,y,z,material         */
        double  ****uvel_CC,            /* u-cell-centered velocity         (INPUT) */
        double  ****vvel_CC,            /* v-cell-centered velocity         (INPUT) */
        double  ****wvel_CC,            /* v-cell-centered velocity         (INPUT) */
        double  ****speedSound,         /* speed of sound cell cell-center  (INPUT) */
        double  CFL,                     /* CFL number                       (INPUT) */
        int     nMaterials          )
{
    int     i, j, k,m,                   /*   loop indices  locators        */
            xLo,    xHi,
            yLo,    yHi,
            zLo,    zHi;
            
static int  iterNum;                    /* Iteration number                 */
    
    double  A, B,
            fudge_factor,
            delt_stability,             /* based on stability               */
            delt_CFL;                   /* based on the CFL number          */
/*START_DOC*/
    iterNum ++;
/*______________________________________________________________________
*  While the solution is stabilizing linearly increase the CFL until
*   it reaches the value from the input file
*_______________________________________________________________________*/
    if ( iterNum < N_ITERATIONS_TO_STABILIZE )
    {
        CFL =  CFL * (double)iterNum * (1.0/N_ITERATIONS_TO_STABILIZE);
    }
        
/*__________________________________
* Initialize variables
* and calculate the looping indicies to 
* include the ghostcells
*___________________________________*/
    *delt           = BIG_NUM;
    delt_stability  = BIG_NUM;
    delt_CFL        = BIG_NUM;
    fudge_factor    = 1.0;

    xLo = GC_LO(xLoLimit);
    yLo = GC_LO(yLoLimit);
    zLo = GC_LO(zLoLimit);
    xHi = GC_HI(xHiLimit);
    yHi = GC_HI(yHiLimit);
    zHi = GC_HI(zHiLimit);
/*______________________________________________________________________
* Now calculate the next time step based on the CFL restraint
* Note this 
*_______________________________________________________________________*/
    for ( m = 1; m <= nMaterials; m ++)
    {
        for ( i = xLo; i <= xHi; i++)
        {
            for ( j = yLo; j <= yHi; j++)
            {
                for ( k = zLo; k <= zHi; k++)
                {  
                    /*__________________________________
                    *   Based on the convective velocity
                    *___________________________________*/  
                #if (compute_delt_based_on_velocity == 1)
                    A   = fudge_factor*CFL*delX/fabs(uvel_CC[m][i][j][k] + SMALL_NUM); 
                    B   = fudge_factor*CFL*delY/fabs(vvel_CC[m][i][j][k] + SMALL_NUM);
                    delt_CFL = DMIN(A, delt_CFL);
                    delt_CFL = DMIN(B, delt_CFL);
                #endif
                    /*__________________________________
                    *   Based on the speed of sound
                    *___________________________________*/
                #if (compute_delt_based_on_velocity == 2)
                    A   = fudge_factor*CFL*delX/fabs(speedSound[m][i][j][k] + SMALL_NUM); 
                    B   = fudge_factor*CFL*delY/fabs(speedSound[m][i][j][k] + SMALL_NUM);
                    delt_CFL = DMIN(A, delt_CFL);
                    delt_CFL = DMIN(B, delt_CFL);
                #endif

                    /*__________________________________
                    *   Based on the speed of sound AND
                    *   the convective velocity
                    *___________________________________*/
                #if (compute_delt_based_on_velocity == 3)
                    A   = fudge_factor*CFL*delX/( speedSound[m][i][j][k] + fabs(uvel_CC[m][i][j][k]) + SMALL_NUM); 
                    B   = fudge_factor*CFL*delY/( speedSound[m][i][j][k] + fabs(vvel_CC[m][i][j][k]) + SMALL_NUM);
                    delt_CFL = DMIN(A, delt_CFL);
                    delt_CFL = DMIN(B, delt_CFL);
                #endif
                    /*__________________________________
                    * based on stability requirements
                    * see references
                    *___________________________________*/
                    A   = fudge_factor * 0.5 * pow(delX, 2.0)/fabs(uvel_CC[m][i][j][k]); 
                    B   = fudge_factor * 0.5 * pow(delY, 2.0)/fabs(vvel_CC[m][i][j][k]);
                    delt_stability = DMIN(A, delt_stability);
                    delt_stability = DMIN(B, delt_stability);
                }
            }
        }
    }

    *delt = DMIN(delt_stability, delt_CFL);

   /*`==========TESTING==========*/ 
 *delt = delt_CFL;
 /*==========TESTING==========`*/
    /*__________________________________
    *   Print some error messages
    *___________________________________*/
    if ( *delt < delt_limits[1] )
    {
        fprintf(stderr,"Current delt %f , minimum allowable delt: %f\n",
        *delt, delt_limits[1]);
        Message(1,"Warning:","The current time step is < than the allowable minimum",
           "specifed in the input file, now setting delt = delt_min.");
        *delt = delt_limits[1];
    }

    if ( *delt > delt_limits[2] )
    {
        fprintf(stderr,"Current delt %f , max. allowable delt: %f\n",
        *delt, delt_limits[2]);
        Message(0,"Warning:","The current time step is > than the allowable max",
           "now setting the delt = delt_maximum");
        *delt = delt_limits[2];
    }  

/*______________________________________________________________________
*   DEBUGGING INFORMATION
*_______________________________________________________________________*/ 
#if switchDebug_find_delta_time                                          
        fprintf(stderr," ______________________________________________\n");
        fprintf(stderr,"find_delta_time\n");
        fprintf(stderr,"delta_time while the solution stabilizes CFL =%f \t %f\n", 
                        CFL, delt_CFL);
        fprintf(stderr,"delta_time based on CFL=%f \t %f \n",CFL, delt_CFL);
        fprintf(stderr,"delta_time based on stability  \t%f \n",delt_stability);        
        fprintf(stderr,"The new time step is \t\t%f\n",*delt);
        fprintf(stderr," ______________________________________________\n");

#endif 
/*__________________________________
*   Quite fullwarn remarks in a way that
*   is compiler independent
*___________________________________*/
    QUITE_FULLWARN(delZ);   QUITE_FULLWARN(wvel_CC[1][0][0][0]);
    speedSound  = speedSound;
    CFL         = CFL;

}
/*STOP_DOC*/ 


#if 0


This doesn't work so eventually trash it.
/* 
 ======================================================================*/
#include <math.h>
#include <stdio.h>
#include "functionDeclare.h"
#include "parameters.h"
#include "switches.h"
#include "nrutil+.h"
#include "macros.h"

/*
 Function:  find_delta_time_based_on_on_change_in_vol
 Filename:  commonFunctions.c
  

 Purpose:
   This function calculates deltak time interms of the volume that will
   be advected out of a cell.  To understand what is going on see the reference
  
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       05/26/00    
 Refererences:
    Caveat: A computer Code for Fluid Dynamics Problems with Large Distortion
    and Internal slip, pg 45
    
 Implementation Note:
     When testing Sod's shock tube problem it was discovered that the solution
     had to stabilize over several iterations before the computed velocity field 
     was close to the actual solution.  Subsequently, during the first few 
     iterations the computed delt was significantly larger than what
     it should be.  to get around this during the first N_ITERATIONS_TO_STABILIZE
     the CFL linearly increases  from 1/N_ITERATIONS_TO_STABILIZE to 
     CFL from the input file over N_ITERATIONS_TO_STABILIZE.
 ---------------------------------------------------------------------  */
 
 void find_delta_time_based_on_change_in_vol(
        int     xLoLimit,               /* x-array lower limit              */
        int     yLoLimit,               /* y-array lower limit              */
        int     zLoLimit,               /* z-array lower limit              */
        int     xHiLimit,               /* x-array upper limit              */
        int     yHiLimit,               /* y-array upper limit              */
        int     zHiLimit,               /* z-array upper limit              */
        double  *delt,                  /* delta t                          (OUTPUT)*/
        double  *delt_limits,           /* delt_limits[1] = delt minimum    (INPUT) */
                                        /* delt_limits[2] = delt_maximum    (INPUT) */
        double  delX,                   /* distance/cell, xdir              (INPUT) */
        double  delY,                   /* distance/cell, ydir              (INPUT) */
        double  delZ,                   /* distance/cell, zdir              (INPUT) */
                                        /* (*)vel_CC(x,y,z,material         */
        double  ****uvel_CC,            /* u-cell-centered velocity         (INPUT) */
        double  ****vvel_CC,            /* v-cell-centered velocity         (INPUT) */
        double  ****wvel_CC,            /* v-cell-centered velocity         (INPUT) */
        double  ******uvel_FC,          /* u-cell-face centered velocity    (INPUT) */
        double  ******vvel_FC,          /* v-cell-face centered velocity    (INPUT) */
        double  ******wvel_FC,          /* v-cell-face centered velocity    (INPUT) */
        double  ****speedSound,         /* speed of sound cell cell-center  (INPUT) */
        double  CFL,                     /* CFL number                      (INPUT) */
        int     nMaterials          )
{
    int     i, j, k,m;                   /*   loop indices  locators        */

            
static int  iterNum;                    /* Iteration number                 */
    
    double  A, B, C,
            sum_flux,                   /* characteristic speed * Face area */
            vel_top,    vel_bottom,     /* characteristic speed of face     */
            vel_left,   vel_right,
            vel_front,  vel_back,
            delt_tmp;                  
/*START_DOC*/
    iterNum ++;
/*______________________________________________________________________
*  While the solution is stabilizing linearly increase the CFL until
*   it reaches the value from the input file
*_______________________________________________________________________*/
    if ( iterNum < N_ITERATIONS_TO_STABILIZE )
    {
        CFL =  CFL * (double)iterNum * (1.0/N_ITERATIONS_TO_STABILIZE);
    }
        
/*__________________________________
* Initialize variables
* and calculate the looping indicies to 
* include the ghostcells
*___________________________________*/
    *delt           = BIG_NUM;
    delt_tmp        = BIG_NUM;
/*______________________________________________________________________
* Now calculate the next time step based on the CFL restraint
* Note this 
*_______________________________________________________________________*/
    for ( m = 1; m <= nMaterials; m ++)
    {
        for ( i = xLoLimit; i <= xHiLimit; i++)
        {
            for ( j = yLoLimit; j <= yHiLimit; j++)
            {
                for ( k = zLoLimit; k <= zHiLimit; k++)
                {  
                /*__________________________________
                *  Top face
                *___________________________________*/
                    A = MAX(speedSound[m][i][j][k], speedSound[m][i][j+1][k]);
                    B = fabs( vvel_CC[m][i][j][k] - vvel_CC[m][i][j+1][k] );
                    C = fabs( *vvel_FC[i][j][k][TOP][m] );
                    #if (compute_delt_based_on_velocity == 1)
                    A = 0.0;
                    #endif
                    vel_top = A + B + C;
                /*__________________________________
                *  Bottom face
                *___________________________________*/
                    A = MAX(speedSound[m][i][j][k], speedSound[m][i][j-1][k]);
                    B = fabs( vvel_CC[m][i][j][k] - vvel_CC[m][i][j-1][k] );
                    C = fabs( *vvel_FC[i][j][k][BOTTOM][m] );
                    #if (compute_delt_based_on_velocity == 1)
                    A = 0.0;
                    #endif

                    vel_bottom = A + B + C;
                /*__________________________________
                *   Right face
                *___________________________________*/
                    A = MAX(speedSound[m][i][j][k], speedSound[m][i+1][j][k]);
                    B = fabs( uvel_CC[m][i][j][k] - uvel_CC[m][i+1][j][k] );
                    C = fabs( *uvel_FC[i][j][k][RIGHT][m] );
                    #if (compute_delt_based_on_velocity == 1)
                    A = 0.0;
                    #endif

                    vel_right = A + B + C;
                /*__________________________________
                *   Left Face
                *___________________________________*/
                    A = MAX(speedSound[m][i][j][k], speedSound[m][i-1][j][k]);
                    B = fabs( uvel_CC[m][i][j][k] - uvel_CC[m][i-1][j][k] );
                    C = fabs( *uvel_FC[i][j][k][LEFT][m] );
                    #if (compute_delt_based_on_velocity == 1)
                    A = 0.0;
                    #endif

                    vel_right = A + B + C;   
/*`==========TESTING==========*/ 
                /*__________________________________
                *   Front face
                *___________________________________*/
                    A = MAX(speedSound[m][i][j][k], speedSound[m][i][j][k+1]);
                    B = fabs( wvel_CC[m][i][j][k] - wvel_CC[m][i][j][k+1] );
                    C = fabs( *wvel_FC[i][j][k][FRONT][m] );
                    #if (compute_delt_based_on_velocity == 1)
                    A = 0.0;
                    #endif

                    vel_front = A + B + C;  
                    
                    vel_front = 0.0;
                /*__________________________________
                *   Back face
                *___________________________________*/  
                    A = MAX(speedSound[m][i][j][k], speedSound[m][i][j][k-1]);
                    B = fabs( wvel_CC[m][i][j][k] - wvel_CC[m][i][j][k-1] );
                    C = fabs( *wvel_FC[i][j][k][BACK][m] );
                    #if (compute_delt_based_on_velocity == 1)
                    A = 0.0;
                    #endif

                    vel_back = A + B + C; 
                    
                    vel_back = 0.0;
 /*==========TESTING==========`*/  
                    
                    sum_flux =  (vel_top    + vel_bottom ) * delX * delZ 
                            +   (vel_left   + vel_right )  * delY * delZ  
                            +   (vel_front  + vel_back )   * delX * delY; 
                             
                delt_tmp = CFL * delX * delY * delZ/(2.0 * sum_flux);
                *delt = DMIN(*delt, delt_tmp);

                }
            }
        }
    }

    /*__________________________________
    *   Print some error messages
    *___________________________________*/
    if ( *delt < delt_limits[1] )
    {
        fprintf(stderr,"Current delt %f , minimum allowable delt: %f\n",
        *delt, delt_limits[1]);
        Message(1,"Warning:","The current time step is < than the allowable minimum",
           "specifed in the input file, now setting delt = delt_min.");
        *delt = delt_limits[1];
    }

    if ( *delt > delt_limits[2] )
    {
        fprintf(stderr,"Current delt %f , max. allowable delt: %f\n",
        *delt, delt_limits[2]);
        Message(0,"Warning:","The current time step is > than the allowable max",
           "now setting the delt = delt_maximum");
        *delt = delt_limits[2];
    }  

/*______________________________________________________________________
*   DEBUGGING INFORMATION
*_______________________________________________________________________*/ 
#if switchDebug_find_delta_time                                          
        fprintf(stderr," ______________________________________________\n");
        fprintf(stderr,"find_delta_time\n");
        fprintf(stderr,"delta_time based on change in cell volume: CFL%f \tdelt %f \n",CFL, *delt);        
        fprintf(stderr," ______________________________________________\n");

#endif 
/*__________________________________
*   Quite fullwarn remarks in a way that
*   is compiler independent
*___________________________________*/
    QUITE_FULLWARN(delZ);   QUITE_FULLWARN(wvel_CC[1][0][0][0]);
    speedSound  = speedSound;
    CFL         = CFL;

}
/*STOP_DOC*/ 

#endif



/* 
 ======================================================================*/
#include <math.h>
#include <assert.h>
#include "parameters.h"
#include "functionDeclare.h"
/* ---------------------------------------------------------------------
 Function:  find_loop_index_limits_at_domain_edges--MISC: Deterimine 
 the looping limits for each ghostcell wall.
 Filename:  commonFunctions.c
 Purpose:   Determine the loop limits along one ghostcell outside of the 
 computational domain. 

NOTE:   The returned indices do not include the corner cells. you need
to deal with those separately.
    
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       09/9/99    

 ---------------------------------------------------------------------  */
 void find_loop_index_limits_at_domain_edges(                
    int     xLoLimit,                   /* x-array lower limit              (INPUT) */
    int     yLoLimit,                   /* y-array lower limit              (INPUT) */
    int     zLoLimit,                   /* z-array lower limit              (INPUT) */
    int     xHiLimit,                   /* x-array upper limit              (INPUT) */
    int     yHiLimit,                   /* y-array upper limit              (INPUT) */
    int     zHiLimit,                   /* z-array upper limit              (INPUT) */
    int     *xLo,                       /* Modified values                  (OUTPUT)*/
    int     *yLo,                       /* Indices for the walls            (OUTPUT)*/
    int     *zLo,                       /*                                  (OUTPUT)*/
    int     *xHi,                       /*                                  (OUTPUT)*/
    int     *yHi,                       /*                                  (OUTPUT)*/
    int     *zHi,                       /*                                  (OUTPUT)*/
    int     wall    )                   /* What wall are we interesed in    (INPUT) */
{

/*START_DOC*/

/*______________________________________________________________________
*
*                           xLoLimit    xHiLimit
*                               |          |
*                        _________________________
*                       /    /    /    /    /    /|
*                      /    /    /    /    /    / |
*                     /____/____/____/____/____/  |
*                    /    /    /    /    /    /| -------         back_GC
*                   /    /    /    /    /    / |  | 
*                  /____/____/____/____/____/  | /|
*                 /    /    /    /    /    /| -|--|--         zHiLimit
*                /    /    /    /    /    / |  |  |
*               /____/____/____/____/____/  | /|  |
*              /    /    /    /    /    /| -|--|-/|       zLoLimit
*             /    /    /    /    /    / |  |  |/ |
*            /____/____/____/____/____/  | /|  |  |
*            |    |    |    |    |    |  |/ | /|  |
*            |    |    |    |    |    |  |  |/ | /|
*            |    |    |    |    |    | /|  |  |/ |
*            |____|____|____|____|____|/ | /|  |  /                           
*            |    |    |    |    |    |  |/ | /| /
*   yHiLimit |    |    |    |    |    | /|  |/ |/   
*            |____|____|____|____|____|/ | /|  /
*            |    |    |    |    |    |  |/ | / 
*            |    |    |    |    |    |  |  |/
*   yLoLimit |    |    |    |    |    | /|  /
*            |____|____|____|____|____|/ | /
*            |    |    |    |    |    |  |/
*            |    |    |    |    |    |  /
*            |    |    |    |    |    | /
*            |____|____|____|____|____|/                
               
                    
*
*_______________________________________________________________________*/
    /*__________________________________
    *   Define the corner of each wall
    *   BE CAREFULL HERE NOT TO OVERLAP!
    *   NEED TO ADD 3D 
    *___________________________________*/
   if( wall == LEFT ) 
   {
        *xLo = xLoLimit - N_GHOSTCELLS;     *xHi = xLoLimit - N_GHOSTCELLS;
        *yLo = yLoLimit;                    *yHi = yHiLimit; 
        *zLo = zLoLimit - N_GHOSTCELLS;     *zHi = zHiLimit + N_GHOSTCELLS;
    }
   if( wall == RIGHT ) 
   {
        *xLo = xHiLimit + N_GHOSTCELLS;     *xHi = xHiLimit + N_GHOSTCELLS;
        *yLo = yLoLimit;                    *yHi = yHiLimit; 
        *zLo = zLoLimit - N_GHOSTCELLS;     *zHi = zHiLimit + N_GHOSTCELLS;
    }
   if( wall == TOP ) 
   {
        *xLo = xLoLimit;                    *xHi = xHiLimit;
        *yLo = yHiLimit + N_GHOSTCELLS;     *yHi = yHiLimit + N_GHOSTCELLS; 
        *zLo = zLoLimit - N_GHOSTCELLS;     *zHi = zHiLimit + N_GHOSTCELLS;
    }
   if( wall == BOTTOM ) 
   {
        *xLo = xLoLimit;                    *xHi = xHiLimit;
        *yLo = yLoLimit - N_GHOSTCELLS;     *yHi = yLoLimit - N_GHOSTCELLS; 
        *zLo = zLoLimit - N_GHOSTCELLS;     *zHi = zHiLimit + N_GHOSTCELLS;
    }
   if( wall == FRONT ) 
   {
 /*        *xLo = xLoLimit + N_GHOSTCELLS; *xHi = xHiLimit - N_GHOSTCELLS;
        *yLo = yLoLimit + N_GHOSTCELLS; *yHi = yHiLimit - N_GHOSTCELLS; 
        *zLo = zLoLimit;     *zHi = zLoLimit; */
    }
   if( wall == BACK ) 
   {
       /*  *xLo = xLoLimit + N_GHOSTCELLS; *xHi = xHiLimit - N_GHOSTCELLS;
        *yLo = yLoLimit + N_GHOSTCELLS; *yHi = yHiLimit - N_GHOSTCELLS; 
        *zLo = zHiLimit;     *zHi = zHiLimit; */
    }
/*STOP_DOC*/

}


/* 
 ======================================================================*/
 #include "parameters.h"
 #include "macros.h"
 #include "functionDeclare.h"
 #include <assert.h>
 #include <stdarg.h>
/*
 Function:  zero_arrays_4d--MISC: initialize input data arrays
 Filename: commonFunctions.c

 Purpose:  Zero 4d double arrays

Note on implementation:
            This function uses a variable length argument list.  
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       12/27/99   Written   
_______________________________________________________________________ */

void    zero_arrays_4d(  
    int     xLoLimit,               /* x-array lower limit              */
    int     yLoLimit,               /* y-array lower limit              */
    int     zLoLimit,               /* z-array lower limit              */
    int     xHiLimit,               /* x-array upper limit              */
    int     yHiLimit,               /* y-array upper limit              */
    int     zHiLimit,               /* z-array upper limit              */
    int     n4dl,
    int     n4dh,
    int     n_data_arrays,          /* number of data arrays            (INPUT) */
    double  ****array1,...)         /* data(x,y,z,M)                    (INPUT) */             
{
    va_list ptr_data_array;         /* pointer to each data array       */ 
    int i, j, k, m,
        array;

/*__________________________________
* double check inputs
*___________________________________*/
    assert ( xLoLimit >= 0 && xHiLimit < X_MAX_LIM);
    assert ( yLoLimit >= 0 && yHiLimit < Y_MAX_LIM);
    assert ( zLoLimit >= 0 && zHiLimit < Z_MAX_LIM);       
/* ______________________________
  Now print the string
  print a new line if the returnchar
   is found.
______________________________  */
    va_start(ptr_data_array, n_data_arrays);
    array = 0;
    /*__________________________________
    *   Loop through each data array in the
    *   argument list
    *___________________________________*/ 
    for (array = 1; array <=n_data_arrays; array++)
    {   
        array1 = va_arg(ptr_data_array, double****);
         
        for(m = n4dl; m <= n4dh; m++)
        {                
            for(k = GC_LO(zLoLimit); k <= GC_HI(zHiLimit); k++)
            {
                for(j = GC_LO(yLoLimit); j <= GC_HI(yHiLimit); j++)
                {
                    for(i = GC_LO(xLoLimit); i <= GC_HI(xHiLimit); i++)
                    {
                       array1[m][i][j][k] = 0.0;              
                    }
                }
            }
        }
    } 
    va_end(ptr_data_array);                     /* clean up when done   */            
   
 }
/*STOP_DOC*/

/* 
 ======================================================================*/
 #include "parameters.h"
 #include "macros.h"
 #include "functionDeclare.h"
 #include <assert.h>
 #include <stdarg.h>
/*
 Function:  zero_arrays_5d--MISC: initialize input data arrays
 Filename: commonFunctions.c

 Purpose:  Initialize 5d double arrays

Note on implementation:
            This function uses a variable length argument list.  
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       12/27/99   Written   
_______________________________________________________________________ */

void    zero_arrays_5d(  
    int     xLoLimit,               /* x-array lower limit              */
    int     yLoLimit,               /* y-array lower limit              */
    int     zLoLimit,               /* z-array lower limit              */
    int     xHiLimit,               /* x-array upper limit              */
    int     yHiLimit,               /* y-array upper limit              */
    int     zHiLimit,               /* z-array upper limit              */
    int     n4dlo,                  /* 4d lower limit                   */
    int     n4dhi,                  /* 4d upper limit                   */
    int     n5dlo,                  /* 5d lower limit                   */
    int     n5dhi,                  /* 5d upper limit                   */
    int     n_data_arrays,          /* number of data arrays            (INPUT) */
    double  *****array1,...)        /* data(x,y,z,M)                    (INPUT) */             
{
    va_list ptr_data_array;         /* pointer to each data array       */ 
    int i, j, k, m, n,
        array;

/*__________________________________
* double check inputs
*___________________________________*/
    assert ( xLoLimit >= 0 && xHiLimit < X_MAX_LIM);
    assert ( yLoLimit >= 0 && yHiLimit < Y_MAX_LIM);
    assert ( zLoLimit >= 0 && zHiLimit < Z_MAX_LIM);       
/* ______________________________
  Now print the string
  print a new line if the returnchar
   is found.
______________________________  */
    va_start(ptr_data_array, n_data_arrays);
    array = 0;
    /*__________________________________
    *   Loop through each data array in the
    *   argument list
    *___________________________________*/ 
    for (array = 1; array <=n_data_arrays; array++)
    {   
        array1 = va_arg(ptr_data_array, double*****);


        for(n = n5dlo; n <= n5dhi; n++)
        {         
            for(m = n4dlo; m <= n4dhi; m++)
            {                
                for(k = GC_LO(zLoLimit); k <= GC_HI(zHiLimit); k++)
                {
                    for(j = GC_LO(yLoLimit); j <= GC_HI(yHiLimit); j++)
                    {
                        for(i = GC_LO(xLoLimit); i <= GC_HI(xHiLimit); i++)
                        {
                           array1[i][j][k][m][n] = 0.0;              
                        }
                    }
                }
            }
        }
    } 
    va_end(ptr_data_array);                     /* clean up when done   */            
   
 }
/*STOP_DOC*/


/* 
 ======================================================================*/
 #include "parameters.h"
 #include "macros.h"
 #include "functionDeclare.h"
 #include <assert.h>
 #include <stdarg.h>
/*
 Function:  zero_arrays_6d--MISC: initialize input data arrays
 Filename: commonFunctions.c

 Purpose:  Initialize 6d double arrays

Note on implementation:
            This function uses a variable length argument list.  
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       01/20/99   Written   
_______________________________________________________________________ */

void    zero_arrays_6d(  
    int     xLoLimit,               /* x-array lower limit              */
    int     yLoLimit,               /* y-array lower limit              */
    int     zLoLimit,               /* z-array lower limit              */
    int     xHiLimit,               /* x-array upper limit              */
    int     yHiLimit,               /* y-array upper limit              */
    int     zHiLimit,               /* z-array upper limit              */
    int     n4dlo,                  /* 4d lower limit                   */
    int     n4dhi,                  /* 4d upper limit                   */
    int     n5dlo,                  /* 5d lower limit                   */
    int     n5dhi,                  /* 5d upper limit                   */
    int     n_data_arrays,          /* number of data arrays            (INPUT) */
    double  ******array1,...)       /* data(x,y,z,f,M)                  (INPUT) */             
{
    va_list ptr_data_array;         /* pointer to each data array       */ 
    int i, j, k, f, m,
        array;

/*__________________________________
* double check inputs
*___________________________________*/
    assert ( xLoLimit >= 0 && xHiLimit < X_MAX_LIM);
    assert ( yLoLimit >= 0 && yHiLimit < Y_MAX_LIM);
    assert ( zLoLimit >= 0 && zHiLimit < Z_MAX_LIM);       
/* ______________________________
  Now print the string
  print a new line if the returnchar
   is found.
______________________________  */
    va_start(ptr_data_array, n_data_arrays);
    array = 0;
    /*__________________________________
    *   Loop through each data array in the
    *   argument list
    *___________________________________*/ 
    for (array = 1; array <=n_data_arrays; array++)
    {   
        array1 = va_arg(ptr_data_array, double******);

        
        for(m = n5dlo; m <= n5dhi; m++)
        {         
            for(f = n4dlo; f <= n4dhi; f++)
            {                
                for(k = GC_LO(zLoLimit); k <= GC_HI(zHiLimit); k++)
                {
                    for(j = GC_LO(yLoLimit); j <= GC_HI(yHiLimit); j++)
                    {
                        for(i = GC_LO(xLoLimit); i <= GC_HI(xHiLimit); i++)
                        {
                           *array1[i][j][k][f][m] = 0.0;              
                        }
                    }
                }
            }
        }
    } 
    va_end(ptr_data_array);                     /* clean up when done   */            
   
 }
/*STOP_DOC*/

/* 
 ======================================================================*/
 #include "parameters.h"
 #include "macros.h"
 #include "functionDeclare.h"
 #include <assert.h>
 #include <stdarg.h>
/*
 Function:  zero_arrays_3d--MISC: initialize input data arrays
 Filename: commonFunctions.c

 Purpose:  Zero multi-material arrays

Note on implementation:
            This function uses a variable length argument list.  
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       12/27/99   Written   
_______________________________________________________________________ */

void    zero_arrays_3d(  
    int     xLoLimit,               /* x-array lower limit              */
    int     yLoLimit,               /* y-array lower limit              */
    int     zLoLimit,               /* z-array lower limit              */
    int     xHiLimit,               /* x-array upper limit              */
    int     yHiLimit,               /* y-array upper limit              */
    int     zHiLimit,               /* z-array upper limit              */
    int     n_data_arrays,          /* number of data arrays            (INPUT) */
    double  ***array1,...)          /* data(x,y,z)                      (INPUT) */             
{
    va_list ptr_data_array;         /* pointer to each data array       */ 
    int i, j, k,
        array;

/*__________________________________
* double check inputs
*___________________________________*/
    assert ( xLoLimit >= 0 && xHiLimit < X_MAX_LIM);
    assert ( yLoLimit >= 0 && yHiLimit < Y_MAX_LIM);
    assert ( zLoLimit >= 0 && zHiLimit < Z_MAX_LIM);       
/* ______________________________
  Now print the string
  print a new line if the returnchar
   is found.
______________________________  */
    va_start(ptr_data_array, n_data_arrays);
    array = 0;
    /*__________________________________
    *   Loop through each data array in the
    *   argument list
    *___________________________________*/ 
    for (array = 1; array <=n_data_arrays; array++)
    {   array1 = va_arg(ptr_data_array, double***);
                      
        for(k = GC_LO(zLoLimit); k <= GC_HI(zHiLimit); k++)
        {
            for(j = GC_LO(yLoLimit); j<= GC_HI(yHiLimit); j++)
            {
                for(i = GC_LO(xLoLimit); i <= GC_HI(xHiLimit); i++)
                {
                   array1[i][j][k] = 0.0;              
                }
            }
        }
        
    } 
    va_end(ptr_data_array);                     /* clean up when done   */            
   
 }
/*STOP_DOC*/
/* 
 ======================================================================*/
 #include "parameters.h"
 #include "macros.h"
 #include "functionDeclare.h"
 #include <assert.h>
 #include <stdarg.h>
/*
 Function:  divergence_of_face_centered_velocity--MISC: Computes that divergence of the face centered velocity
 Filename: commonFunctions.c

 Purpose:   For each cell compute the divergence of the face centered 
            velocities.  This is done in a function as opposed to a 
            inside each function inorder to keep the discretization 
            identical.  
Note:       When you apply the divergence theorem the divergence of the 
            the velocity field is the surface integral over a volume.
            The units of div_vel_FCl are  m^3/sec
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       03/13/99   Written   
       
Reference:
            "Computational Fluid Mechanics and Heat Transfer", Tannehill
            Anderson and Plether, pg 73
_______________________________________________________________________ */

void    divergence_of_face_centered_velocity(  
    int     xLoLimit,               /* x-array lower limit              */
    int     yLoLimit,               /* y-array lower limit              */
    int     zLoLimit,               /* z-array lower limit              */
    int     xHiLimit,               /* x-array upper limit              */
    int     yHiLimit,               /* y-array upper limit              */
    int     zHiLimit,               /* z-array upper limit              */
    double  delX,                   /* distance/cell, xdir              (INPUT) */
    double  delY,                   /* distance/cell, ydir              (INPUT) */
    double  delZ,                   /* distance/cell, zdir              (INPUT) */
    double  ******uvel_FC,          /* u-face-centered velocity         (INPUT) */
    double  ******vvel_FC,          /*  v-face-centered velocity        (INPUT) */
    double  ******wvel_FC,          /* w face-centered velocity         (INPUT) */
    double  ****div_vel_FC,
    int     nMaterials)
{
   
    int     i, j, k, m;
      
    double  topface, bottomface,        /* temp symbols to represent terms  */
            rightface, leftface,
            frontface, backface;

/*__________________________________
* double check inputs
*___________________________________*/
    assert ( xLoLimit >= 0 && xHiLimit < X_MAX_LIM);
    assert ( yLoLimit >= 0 && yHiLimit < Y_MAX_LIM);
    assert ( zLoLimit >= 0 && zHiLimit < Z_MAX_LIM);  
    frontface   = 0.0;
    backface    = 0.0;     
       
    for (m = 1; m <= nMaterials; m++)
    {  
        for ( i = xLoLimit; i <= xHiLimit; i++)
        {
            for ( j = yLoLimit; j <= yHiLimit; j++)
            {
                for ( k = zLoLimit; k <= zHiLimit; k++)
                { 
                   /*__________________________________
                    * top and bottom face contributions
                    *___________________________________*/
                   topface      =  delX*delZ* *vvel_FC[i][j][k][TOP][m];
                   bottomface   = -delX*delZ* *vvel_FC[i][j][k][BOTTOM][m];
                    /*__________________________________
                    * left and right face contributions
                    *___________________________________*/
                   leftface     = -delY*delZ* *uvel_FC[i][j][k][LEFT][m];
                   rightface    =  delY*delZ* *uvel_FC[i][j][k][RIGHT][m];
#if (N_DIMENSIONS == 3)           
                    /*__________________________________
                    * front and back face contributions
                    *___________________________________*/
                   frontface    =  delX*delY* *wvel_FC[i][j][k][FRONT][m];
                   backface     = -delX*delY* *wvel_FC[i][j][k][BACK][m];
#endif
                    /*__________________________________
                    * 
                    *___________________________________*/
                    div_vel_FC[m][i][j][k] = (   topface   + bottomface    + leftface 
                                               + rightface + frontface     + backface );
                }
            }
        }  
    }
/*__________________________________
*   QUITE FULLWARN 
*___________________________________*/ 
    delX    = delX;         delY    = delY;     delZ    = delZ;
    QUITE_FULLWARN(wvel_FC[1][1][1][1][1]);      
 }
/*STOP_DOC*/
/* 
 ======================================================================*/
#include <math.h>
#include <assert.h>
#include <stdio.h>
#include "nrutil+.h"
#include "functionDeclare.h"
#include "switches.h"
#include "parameters.h"
#include "macros.h"
/* ---------------------------------------------------------------------  
 Function:  grad_q--MISC: Computes the gradients of (q) in the x, y and z dirs.
 Filename:  commonFunctions.c
 Purpose:
   This routine calculates the gradient in the x, y and z directions.  The general 
   algorithm is described in the references shown below.  In this code where delX = delY = delZ = constant
   the derviatives reduce to second order centered finite difference expressions  
   
 References:
    CFDLIB98
    
    "Computational Methods in Viscous Aerodynamics", edited by T.K.S Murthy and C.A. Brebbia,
    Elsevier, 1990, pg. 123.
    
    Kashiwa, B, (1987) "Statistical Theory of Turbulent Incompressible Multimaterial Flow" 
    Technical Report LA-11088, Los Alamos National Laborator            
 Steps
 -------------------- 
    1)  check to see that the inputs are valid and determine what walls of the computational
    domain should be included in the calculation.
    2)  Compute the cell-centered gradients for all of the cells inside of the
    computational domain.
    3)  Calculate the gradients for the cells in a single ghostcell layer
    surrounding the computational domain.
    4)  Compute the gradients in each of the corner cells.      
   
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       7/1/99 
       
       
 NEED TO INCLUDE DERIVATIVES IN THE Z DIRECTION   
 ---------------------------------------------------------------------  */
void grad_q(
        int     xLoLimit,               /* x-array lower limit              */
        int     yLoLimit,               /* y-array lower limit              */
        int     zLoLimit,               /* z-array lower limit              */
        int     xHiLimit,               /* x-array upper limit              */
        int     yHiLimit,               /* y-array upper limit              */
        int     zHiLimit,               /* z-array upper limit              */
        double  delX,                   /* Cell width                       */
        double  delY,                   /* Cell Width in the y dir          */
        double  delZ,                   /* Cell width in the z dir          */
        double  ****q_CC,               /* cell center data                 (INPUT) */
        double  ***grad_q_X,            /* gradient of q in x dir           (OUTPUT)*/
        double  ***grad_q_Y,            /* gradient of q in y dir           (OUTPUT)*/
        double  ***grad_q_Z,             /* gradient of q in z dir          (OUTPUT)*/
        int     m           )           /* material                         */
{
        int     i, j, k,                /* cell indices                     */
                wall,
                wallLo, wallHi,
                xLo, xHi,
                yLo, yHi,
                zLo, zHi;
         
/*START_DOC*/                      
/*__________________________________
*   Step 1)
* double check inputs
*___________________________________*/
    assert ( xLoLimit > 0 && xHiLimit < X_MAX_LIM);
    assert ( yLoLimit > 0 && yHiLimit < Y_MAX_LIM);
    assert ( zLoLimit > 0 && zHiLimit < Z_MAX_LIM);
    assert ( m <= N_MATERIAL);
    assert ( delX > (double)0.0 );
    assert ( delY > (double)0.0 );
#if N_DIMENSIONS == 3
    assert ( delZ > 0.0 );
#endif
/*__________________________________
*  Determine what walls are used in the 
*   computational domain.
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


/*______________________________________________________________________
*   Step 2)
*   Calculate the gradients inside of the computational domain
*   2nd order center differences in all directions
    ---------------------------------------------
    |   |   |   |   |   |   |   |   |   |   |   |   -- top_GC       
    ---------------------------------------------                   
    |   | + | o | + | + | + | + | + | + | + |   |   -- yHiLimit     
    ---------------------------------------------                   
    |   | o | o | o | + | + | + | + | + | + |   |                   
    ---------------------------------------------                   
    |   | + | o | + | + | + | + | + | + | + |   |                   
    ---------------------------------------------                    
    |   | + | + | + | + | + | + | + | + | + |   |   -- yLoLimit     
    ---------------------------------------------                   
    |   |   |   |   |   |   |   |   |   |   |   |   -- bottom_GC    
    ---------------------------------------------                   
      |   | xLoLimit             xHiLimit |   |                     
      |                                       |                     
      left_GC                               right_GC                

      
      x = grad_q_X and grad_q_Y
      o = data needed: q_CC 
*_______________________________________________________________________*/
                                  
    for ( i = xLoLimit; i <= xHiLimit; i++)
    {
        for ( j = yLoLimit; j <= yHiLimit; j++)
        {
            for ( k = zLoLimit; k <= zHiLimit; k++)
            { 
            grad_q_X[i][j][k] =         (q_CC[m][i+1][j][k] - q_CC[m][i-1][j][k])/(2.0*delX);
            grad_q_Y[i][j][k] =         (q_CC[m][i][j+1][k] - q_CC[m][i][j-1][k])/(2.0*delY); 
            grad_q_Z[i][j][k] = IF_3D(  (q_CC[m][i][j][k+1] - q_CC[m][i][j][k-1])/(2.0*delZ) );   
                      
            }
        }
    }  
/*______________________________________________________________________
*   Step 3)
*   Compute the gradients in a single layer of ghostcells surrounding the 
*   the computational domain.  The derivatives use data from inside
*   of the computational domain and adjacent ghostcells.
*
*   2nd order center differences along the wall 
*   2nd order forward or backward differences perpendicular to the wall
*
    _____________________________________________
    |   |   |   |   |   | o | xo| o |   |   |   |   -- top_GC       
    ---------------------------------------------                   
    | o | + | + | + | + | + | o | + | + | + | o |   -- yHiLimit     
    ---------------------------------------------                   
    | xo| o | o | + | + | + | o | + | o | o | xo|                   
    ---------------------------------------------                   
    | o | + | + | + | o | + | + | + | + | + | o |                   
    ---------------------------------------------                    
    |   | + | + | + | o | + | + | + | + | + |   |   -- yLoLimit     
    ---------------------------------------------                   
    |   |   |   | o | xo| o |   |   |   |   |   |   -- bottom_GC    
    ---------------------------------------------                   
      |   | xLoLimit             xHiLimit |   |                     
      |                                       |                     
      left_GC                               right_GC
      
      x = grad_q_X and grad_q_Y
      o = data needed: q_CC 
*_______________________________________________________________________*/
    for( wall = wallLo; wall <= wallHi; wall ++)
    {
        /*__________________________________
        *  Find the looping indices associated
        *   with a particular wall
        *___________________________________*/
         find_loop_index_limits_at_domain_edges(                
                    xLoLimit,                  yLoLimit,                   zLoLimit,
                    xHiLimit,                  yHiLimit,                   zHiLimit,
                    &xLo,                      &yLo,                       &zLo,
                    &xHi,                      &yHi,                       &zHi,
                    wall    );
    
        for ( i = xLo; i <= xHi; i++ )
        {
            for ( j = yLo; j <= yHi; j++ )
            { 
                for ( k = zLo; k <= zHi; k++ )
                {        
                
                   if ( wall == LEFT )
                   {
                        grad_q_X[i][j][k] =         (-3.0 * q_CC[m][i][j][k]   + 4.0 * q_CC[m][i+1][j][k] - q_CC[m][i+2][j][k])/(delX);
                        grad_q_Y[i][j][k] =         (q_CC[m][i][j+1][k] - q_CC[m][i][j-1][k])/(2.0*delY); 
                        grad_q_Z[i][j][k] =         0.0;
                       /*  grad_q_Z[i][j][k] = IF_3D(  (q_CC[m][i][j][k+1] - q_CC[m][i][j][k-1])/(2.0*delZ) ); */   
                   }
                                
                   if ( wall == RIGHT )
                   {
                        grad_q_X[i][j][k] =         (3.0 * q_CC[m][i][j][k]   - 4.0 * q_CC[m][i-1][j][k] + q_CC[m][i-2][j][k])/(delX);
                        grad_q_Y[i][j][k] =         (q_CC[m][i][j+1][k] - q_CC[m][i][j-1][k])/(2.0*delY); 
                        grad_q_Z[i][j][k] =         0.0;
                       /*  grad_q_Z[i][j][k] = IF_3D(  (q_CC[m][i][j][k+1] - q_CC[m][i][j][k-1])/(2.0*delZ) ); */   
                    }
                   if ( wall == TOP )
                   {
                        grad_q_X[i][j][k] =         (q_CC[m][i+1][j][k] - q_CC[m][i-1][j][k])/(2.0*delX);
                        grad_q_Y[i][j][k] =         (3.0 * q_CC[m][i][j][k]   - 4.0 * q_CC[m][i][j-1][k] + q_CC[m][i][j-2][k])/(delY);
                        grad_q_Z[i][j][k] =         0.0;
                       /*  grad_q_Z[i][j][k] = IF_3D(  (q_CC[m][i][j][k+1] - q_CC[m][i][j][k-1])/(2.0*delZ) ); */   
                    }
                    if ( wall == BOTTOM )
                   {
                        grad_q_X[i][j][k] =         (q_CC[m][i+1][j][k] - q_CC[m][i-1][j][k])/(2.0*delX);
                        grad_q_Y[i][j][k] =         (-3.0 * q_CC[m][i][j][k]   + 4.0 * q_CC[m][i][j+1][k] - q_CC[m][i][j+2][k])/(delY);
                        grad_q_Z[i][j][k] =         0.0;
                       /*  grad_q_Z[i][j][k] = IF_3D(  (q_CC[m][i][j][k+1] - q_CC[m][i][j][k-1])/(2.0*delZ) ); */   
                    }

                }
            }
        }
    }
/*______________________________________________________________________
*   Step 4)
*   Compute the gradients in each of the corner ghostcells.
*   Need to include 3D here
*   2nd order forward or backward difference are used.

    ---------------------------------------------
    | xo| o | o |   |   |   |   |   | o | o | xo|   -- top_GC       
    ---------------------------------------------                  
    | o | + | + | + | + | + | + | + | + | + | o |   -- yHiLimit     
    ---------------------------------------------                   
    | o | + | + | + | + | + | + | + | + | + | o |                   
    ---------------------------------------------                   
    | o | + | + | + | + | + | + | + | + | + | o |                   
    ---------------------------------------------                    
    | o | + | + | + | + | + | + | + | + | + | o |   -- yLoLimit     
    ---------------------------------------------                   
    | xo| o | o |   |   |   |   |   | o | o | xo|   -- bottom_GC    
    ---------------------------------------------                   
      |   | xLoLimit             xHiLimit |   |                     
      |                                       |                     
      left_GC                               right_GC

      x = grad_q_X and grad_q_Y
      o = data needed: q_CC 
*_______________________________________________________________________*/
/*   Upper Left ghostcell corner 
*___________________________________*/
    i   = GC_LO(xLoLimit);
    j   = GC_HI(yHiLimit);
    zLo = GC_LO(zLoLimit);
    zHi = GC_HI(zHiLimit);
    for (k = zLo; k <=  zHi; k++)
    {
        grad_q_X[i][j][k] =         (-3.0 * q_CC[m][i][j][k]   + 4.0 * q_CC[m][i+1][j][k] - q_CC[m][i+2][j][k])/(delX);
        grad_q_Y[i][j][k] =         ( 3.0 * q_CC[m][i][j][k]   - 4.0 * q_CC[m][i][j-1][k] + q_CC[m][i][j-2][k])/(delY);
        grad_q_Z[i][j][k] =         0.0;
       /*  grad_q_Z[i][j][k] = IF_3D(  (q_CC[m][i][j][k+1] - q_CC[m][i][j][k-1])/(2.0*delZ) ); */   
    }

/*__________________________________
*   Upper right ghostcell corner
*___________________________________*/
    i = GC_HI(xHiLimit);
    j = GC_HI(yHiLimit);
    for (k = zLo; k <=  zHi; k++)
    {
        grad_q_X[i][j][k] =         ( 3.0 * q_CC[m][i][j][k]   - 4.0 * q_CC[m][i-1][j][k] + q_CC[m][i-2][j][k])/(delX);
        grad_q_Y[i][j][k] =         ( 3.0 * q_CC[m][i][j][k]   - 4.0 * q_CC[m][i][j-1][k] + q_CC[m][i][j-2][k])/(delY);
        grad_q_Z[i][j][k] =         0.0;
       /*  grad_q_Z[i][j][k] = IF_3D(  (q_CC[m][i][j][k+1] - q_CC[m][i][j][k-1])/(2.0*delZ) ); */   
    }
    

/*__________________________________
*   Lower right ghostcell corner
*___________________________________*/
    i = GC_HI(xHiLimit);
    j = GC_LO(yLoLimit);
    for (k = zLo; k <=  zHi; k++)
    {
        grad_q_X[i][j][k] =         (  3.0 * q_CC[m][i][j][k]   - 4.0 * q_CC[m][i-1][j][k] + q_CC[m][i-2][j][k])/(delX);
        grad_q_Y[i][j][k] =         ( -3.0 * q_CC[m][i][j][k]   + 4.0 * q_CC[m][i][j+1][k] - q_CC[m][i][j+2][k])/(delY);
        grad_q_Z[i][j][k] =         0.0;
       /*  grad_q_Z[i][j][k] = IF_3D(  (q_CC[m][i][j][k+1] - q_CC[m][i][j][k-1])/(2.0*delZ) ); */   
    }

/*__________________________________
*   Lower left ghostcell corner
*___________________________________*/
    i = GC_LO(xLoLimit);
    j = GC_LO(yLoLimit);
    for (k = zLo; k <=  zHi; k++)
    {
        grad_q_X[i][j][k] =         ( -3.0 * q_CC[m][i][j][k]   + 4.0 * q_CC[m][i+1][j][k] - q_CC[m][i+2][j][k])/(delX);
        grad_q_Y[i][j][k] =         ( -3.0 * q_CC[m][i][j][k]   + 4.0 * q_CC[m][i][j+1][k] - q_CC[m][i][j+2][k])/(delY);
        grad_q_Z[i][j][k] =         0.0;
       /*  grad_q_Z[i][j][k] = IF_3D(  (q_CC[m][i][j][k+1] - q_CC[m][i][j][k-1])/(2.0*delZ) ); */   
    }   

/*__________________________________
*   Quite fullwarn remarks in a way that
*   is compiler independent
*___________________________________*/
    QUITE_FULLWARN(delZ);   
}
/*STOP_DOC*/ 


/* 
 ======================================================================*/
#include "parameters.h"
#include "functionDeclare.h"
#include <assert.h>
/*
 Function:  grad_FC_Xdir--MISC: Calculates gradients of cell-centered data 
 in the (x) dir. that live on the face-center.
 Filename:  CommonFunctions.c

 Purpose:
   This function calculates the gradient of a scalar using cell-center data
   in the x-dir that is located on a face.  This is second order.
   
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       02/22/99    
 ---------------------------------------------------------------------  */
 void grad_FC_Xdir(
        int i,                  /* array index                              (INPUT) */
        int j,                  /* array index                              (INPUT) */
        int k,                  /* array index                              (INPUT) */
        int m,                  /* material                                 (INPUT) */
        double  ****data,       /* data(i,j,k,material                      (INPUT) */
        double  delX,           /* cell width in x-dir                      (INPUT) */
        double  *grad )         /* resultant gradient                       (OUTPUT)*/
{
/*__________________________________
* bullet proofing
*___________________________________*/
    assert ( i > 0 || i < X_MAX_LIM);
    assert ( j > 0 || j < Y_MAX_LIM);
    assert ( k > 0 || k < Z_MAX_LIM);
/*__________________________________
* Note that this is centered on the 
* face, so the 2 cancels in the denominator
*___________________________________*/
    assert( delX >= SMALL_NUM);             /* bullet proofing          */
    
    grad[LEFT]  = (data[m][i][j][k] - data[m][i-1][j][k]) /(delX);
    
    grad[RIGHT] = (data[m][i+1][j][k] - data[m][i][j][k]) /(delX);    
}
/*STOP_DOC*/



/* 
 ======================================================================*/
#include "parameters.h"
#include "functionDeclare.h"
#include <assert.h>
/*
 Function:  grad_FC_Ydir--MISC: Calculates gradients of cell-centered data in the (y) dir. that live on the face-center.
 Filename:  commonFunctions.c
 Purpose:
   This function calculates the gradient of a multimaterial scalar using 
   cell center data in the y-dir that lives on a cell face.  This is second order.
   
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       02/22/99    
 ---------------------------------------------------------------------  */
 void grad_FC_Ydir(
        int i,                  /* array index                              (INPUT) */
        int j,                  /* array index                              (INPUT) */
        int k,                  /* array index                              (INPUT) */
        int m,                  /* material                                 (INPUT) */
        double  ****data,       /* data(i,j,k,material                      (INPUT) */
        double  delY,           /* cell width in y-dir                      (INPUT) */
        double  *grad )         /* resultant gradient                       (OUTPUT)*/
{
/*__________________________________
* bullet proofing
*___________________________________*/
    assert ( i >= 0 || i < X_MAX_LIM);
    assert ( j >  0 || j < Y_MAX_LIM);
    assert ( k >= 0 || k < Z_MAX_LIM);
/*__________________________________
* Note that this is centered on the 
* face, so the 2 cancels in the denominator
*___________________________________*/ 
    assert( delY >= SMALL_NUM);             /* bullet proofing          */
    
    grad[TOP]       = (data[m][i][j+1][k]   - data[m][i][j][k])   /delY;
    
    grad[BOTTOM]    = (data[m][i][j][k]     - data[m][i][j-1][k]) /delY;    
}
/*STOP_DOC*/



/* 
 ======================================================================*/
#include "parameters.h"
#include "functionDeclare.h"
#include <assert.h>
/*
 Function:  grad_FC_Zdir--MISC: Calculates gradients of cell-centered datat in the (z) dir. that live on the face-center.
 Filename:  commonFunctions.c

 Purpose:
   This function calculates the gradient of a scalar using cell center data
   in the Z-dir.  This is second order
   
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       02/22/99    
 Prerequisites: The vector grad[1:6] must be previously defined
 
 ---------------------------------------------------------------------  */
 void grad_FC_Zdir(
        int i,                  /* array index                              (INPUT) */
        int j,                  /* array index                              (INPUT) */
        int k,                  /* array index                              (INPUT) */
        int m,                  /* material                                 (INPUT) */
        double  ****data,       /* data(i,j,k,material                      (INPUT) */
        double  delZ,           /* cell width in z-dir                      (INPUT) */
        double  *grad )         /* resultant gradient                       (OUTPUT)*/
{
/*__________________________________
* bullet proofing
*___________________________________*/
    assert ( i >= 0 || i < X_MAX_LIM);
    assert ( j >= 0 || j < Y_MAX_LIM);
    assert ( k > 0 || k < Z_MAX_LIM);   
/*__________________________________
* Note that this is centered on the 
* face, so the 2 cancels in the denominator
*___________________________________*/ 
    assert( delZ >= SMALL_NUM);             /* bullet proofing          */
    grad[FRONT] = (data[m][i][j][k+1] - data[m][i][j][k]) /delZ;
    
    grad[BACK]  = (data[m][i][j][k] - data[m][i][j][k-1]) /delZ;    
} 
/*STOP_DOC*/

       
/* 
 ======================================================================*/
#include <time.h>
#include "functionDeclare.h"
#include <assert.h>
#include "switches.h"
/*
 Function:  interpolate_to_FC--MISC: interpolate cell-centered data to face-center, weighted by the density.
 Filename:  commonFunctions.c

 Purpose:
   This function calculates the interpolated value of cell centered data
   to the face-center.  The interpolated value is weighted by A
   For example an interpolated value on the left cell face 
   result[face] = 
   
   A[i-1][j][k]*B[i-1][j][k] - A[i][j][k]*B[i][j][k]
   ---------------------------------------------------------------------
              A[i-1][j][k] - A[i][j][k]
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       02/22/99    

 IN args/commons         Units      Description
 ---------------         -----      ----------- 
  needed
  
 Prerequisites: Memory for the single dimensional array "results" needs to be 
                previously allocated.
                
                                 ________ 
                                /  1    /|
                               /_______/ |
                              |       | ______(3)
                       (4)____| I,J,K |  |     
                              |       | /      
                              |_______|/
                                  |               (6) = back face
                                 (2)              (5) = front face
                                 
 
 ---------------------------------------------------------------------  */
 
 void interpolate_to_FC_MF(
        double  ****A,                  /* A (x, y, z, material)        */
        double  ****B,                  /* B (x, y, z, material)        */
        double  *results,                /* interpolated results for each*/
                                        /* cell face                    */
        int     i,
        int     j,
        int     k,
        int     m )

         
{
    int     f,
            cell;
#if sw_interpolate_to_FC      
    time_t start;                       /* timing variables             */
    start = time(NULL);
#endif  
/* _______________________________________________________________________
*  Check for valid inputs
*_______________________________________________________________________*/
    assert ( i >= 0 || i <= X_MAX_LIM);
    assert ( j >= 0 || j <= Y_MAX_LIM);
    assert ( k >= 0 || k <= Z_MAX_LIM);
    assert ( m >= 0 || m <= N_MATERIAL);
/*___________________________________*   
*  Top and bottom cell faces
*___________________________________*/
    
    cell = j + 1;
    
    for (f = TOP; f <= BOTTOM; f++)
    {
        results[f]=     (A[m][i][cell][k]      * B[m][i][cell][k]
                    +    A[m][i][j][k]         * B[m][i][j][k])/
                        (A[m][i][cell][k]      + A[m][i][j][k]);
        cell = j - 1;
    }
/*__________________________________
*left and right cell faces
*___________________________________*/
    cell = i + 1;
    for (f = RIGHT; f <= LEFT; f++)
    {
        results[f]=     (A[m][cell][j][k]      * B[m][cell][j][k]
                    +    A[m][i][j][k]         * B[m][i][j][k])/
                        (A[m][cell][j][k]      + A[m][i][j][k]);
        cell = i - 1;
    }
/*__________________________________
* front and back cell faces
*___________________________________*/
    cell = k + 1;
    for (f = FRONT; f <= BACK; f++)
    {
        results[f]=     (A[m][i][j][cell]      * B[m][i][j][cell]
                    +    A[m][i][j][k]         * B[m][i][j][k])/
                        (A[m][i][j][cell]      + A[m][i][j][k]);
        cell = k - 1;
    }
/*______________________________________________________________________
*  Now test the outputs
*_______________________________________________________________________*/ 
    assert(results[TOP]     <= BIG_NUM);
    assert(results[BOTTOM]  <= BIG_NUM);
    assert(results[RIGHT]   <= BIG_NUM);
    assert(results[LEFT]    <= BIG_NUM);
    assert(results[FRONT]   <= BIG_NUM);
    assert(results[BACK]    <= BIG_NUM);  
/*__________________________________
* Nprintout debugging and timing info
*___________________________________*/         
#if sw_interpolate_to_FC
     stopwatch("interpolate_to_FC",start);
#endif
     
}
/*STOP_DOC*/

/* 
 ======================================================================*/
 #include <stdlib.h>
 #include "functionDeclare.h"
/* 
 Function:  Message--BULLET PROOFING: Writes an error message to (stderr) and stop the program if requested.
 Filename:  commonFunctions.c

 Purpose:  Output an error message and stop the program if requested. 

 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       02/22/99   Written   
_______________________________________________________________________ */

void    Message(
        int     abort,          /* =1 then abort                            */                 
        char    filename[],     /* description of filename                  */
        char    subroutine[],   /* description of function your in          */
        char    message[])      /* message to the user                      */
{        
/* ______________________________
  Now print the string
  print a new line if the returnchar
   is found.
______________________________  */                
        fprintf(stderr,"\n\n ______________________________________________\n");
        fprintf(stderr,"%s\n",filename);
        fprintf(stderr,"%s\n",subroutine);
        fprintf(stderr,"%s\n",message);
        fprintf(stderr,"\n\n ______________________________________________\n");

/* ______________________________
 Now aborting program
______________________________ */
        if(abort == 1)
             exit(1);
       
 }
/*STOP_DOC*/


/* 
 ======================================================================*/
 #include "functionDeclare.h"
 #include <time.h>
/* 

 Function:  stopwatch--PERFORMANCE:
 Filename:  commonFunctions.c
 Purpose:   Output a message and print a time  

 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       02/22/99      
_______________________________________________________________________ */

void stopwatch(
    char message[],             /* print message to the user                */
    time_t start)               /* time in seconds since watch started      */
 
{    
    double secs;
    time_t stop;                 /* timing variables             */
            
/* ______________________________
  Now print the string
______________________________  */ 
    stop = time(NULL);
    secs = difftime(stop, start);               

    fprintf(stderr,"\n___________________________________________TIMER\n");
    fprintf(stderr,"Function %s\n",message);
    fprintf(stderr,"Time it took %lg seconds,\n",secs);
    fprintf(stderr,"______________________________________________\n");       
 }
/*STOP_DOC*/ 
 
 
/* 
 ======================================================================*/
 #include "parameters.h"
 #include "functionDeclare.h"
 #include <assert.h>
/*
 Function:  printData_5d--DEBUG: Write a 5D array to stderr.
 Filename:  commonFunctions.c

 Purpose:  Print to stderr the data array 

 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       02/22/99   Written   
_______________________________________________________________________ */

void    printData_5d(                         
    int     xLoLimit,                   /* x-array lower limit              */
    int     yLoLimit,                   /* y-array lower limit              */
    int     zLoLimit,                   /* z-array lower limit              */
    int     xHiLimit,                   /* x-array upper limit              */
    int     yHiLimit,                   /* y-array upper limit              */
    int     zHiLimit,                   /* z-array upper limit              */
    int     n4dlo, 
    int     n4dhi,
    int     n5dlo, 
    int     n5dhi,
    char    subroutine[],               /* name of function                 (INPUT) */
    char    message[],                  /* message to the user              (INPUT) */
    double  *****data_array,            /* data(x,y,z,face,material         (INPUT) */
    int     ptr_flag,                   /* =1 if data_array is a 4d array   */
                                        /* and the 5th dimension is a pointer*/
                                        /* address, I'm sure this is confusing*/
    int     ghostcells       )          /* include ghostcell data in printout(INPUT)*/       
{ 
    char c[2];
    int i, j, k, f,
        xLo,    xHi,
        yLo,    yHi,
        zLo,    zHi;
/*__________________________________
* double check inputs
*___________________________________*/
    assert ( xLoLimit - ghostcells >= 0 && xHiLimit + ghostcells <= X_MAX_LIM);
    assert ( yLoLimit - ghostcells >= 0 && yHiLimit + ghostcells <= Y_MAX_LIM);
    assert ( zLoLimit - ghostcells >= 0 && zHiLimit + ghostcells <= Z_MAX_LIM);
    assert ( n5dlo >=1 && n5dhi <= N_MATERIAL);
    strcpy(c,"  ");   
    xLo = xLoLimit - ghostcells;    xHi = xHiLimit + ghostcells;
    yLo = yLoLimit - ghostcells;    yHi = yHiLimit + ghostcells;
 /*    zLo = zLoLimit - ghostcells;    zHi = zHiLimit + ghostcells; */
    zLo =   zLoLimit;               zHi = zHiLimit;
       
/* ______________________________
  Now print the string
  print a new line if the returnchar
   is found.
______________________________  */                
    fprintf(stderr,"\n ______________________________________________\n");
    fprintf(stderr,"%s\n",subroutine);
    fprintf(stderr,"%s\n",message);
    fprintf(stderr,"\n");
    for(k = zLo; k <= zHi; k++)
    {
        for(j = yHi; j >= yLo; j--)
        {
        for(f = n4dlo; f <=n4dhi; f++)
                {
            for(i = xLo; i <= xHi; i++)
            {
                
                    if (f == (int)TOP)      strcpy(c,"T ");
                    if (f == (int)BOTTOM)   strcpy(c,"B ");
                    if (f == (int)RIGHT)    strcpy(c,"R ");
                    if (f == (int)LEFT)     strcpy(c,"L ");
                    if (f == (int)FRONT)    strcpy(c,"F ");
                    if (f == (int)BACK)     strcpy(c,"BK");
                    if (ptr_flag != 1)
                        fprintf(stderr,"[%d,%d,%d,%s,%d]= %4.3lf  ",
                        i,j,k,c,n5dlo, data_array[i][j][k][f][n5dlo]);
                    else if (ptr_flag == 1)  
                        fprintf(stderr,"[%d,%d,%d,%s,%d]= %4.3lf  ",
                        i,j,k,c,n5dlo, *data_array[i][j][k][f]);
                

                }
                fprintf(stderr,"\n");
            }
            fprintf(stderr,"\n");
        }
        fprintf(stderr,"\n");
    }
    fprintf(stderr,"______________________________________________\n");
       
 }
/*STOP_DOC*/
/* 
 ======================================================================*/
 #include "parameters.h"
 #include "functionDeclare.h"
 #include <assert.h>
/*
 Function:  printData_6d--DEBUG: Write a 6D array to stderr.
 Filename: commonFunctions.c

 Purpose:  Print to stderr the array 
 Special Note:
            This function assumes that the array data is a pointer array
            meaning
                *data[i][j][k][face][m] = value
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       02/22/99   Written   
_______________________________________________________________________ */

void    printData_6d(                         
    int     xLoLimit,                   /* x-array lower limit              */
    int     yLoLimit,                   /* y-array lower limit              */
    int     zLoLimit,                   /* z-array lower limit              */
    int     xHiLimit,                   /* x-array upper limit              */
    int     yHiLimit,                   /* y-array upper limit              */
    int     zHiLimit,                   /* z-array upper limit              */
    int     n4dlo,                      /* 4th dimension lower limit        */ 
    int     n4dhi,                      /* 4th dimension upper limit        */
    int     n5dlo,                      /* 5th dimension lower limit        */
    int     n5dhi,                      /* 5th dimension upper limit        */
    char    subroutine[],               /* name of funcion                  */
    char    message[],                  /* message to the user              */
    double  ******data_array,           /* *data[i][j][k][f][m] = value     */
    int     ghostcells       )          /* Include ghostcells when printing */        
{ 
    char c[2];
    int i, j, k, f, m,
        xLo,    xHi,
        yLo,    yHi,
        zLo,    zHi;
/*__________________________________
* double check inputs
*___________________________________*/
    assert ( xLoLimit - ghostcells >= 0 && xHiLimit + ghostcells <= X_MAX_LIM);
    assert ( yLoLimit - ghostcells >= 0 && yHiLimit + ghostcells <= Y_MAX_LIM);
    assert ( zLoLimit - ghostcells >= 0 && zHiLimit + ghostcells <= Z_MAX_LIM);
    assert ( n5dlo >=1 && n5dhi <= N_MATERIAL);
    strcpy(c,"  ");   
    xLo = xLoLimit - ghostcells;    xHi = xHiLimit + ghostcells;
    yLo = yLoLimit - ghostcells;    yHi = yHiLimit + ghostcells;
 /*    zLo = zLoLimit - ghostcells;    zHi = zHiLimit + ghostcells; */
    zLo =   zLoLimit;               zHi = zHiLimit;
/*__________________________________
*   HARDWIRE FOR NOW
*___________________________________*/   
    m = n5dlo;    
/* ______________________________
  Now print the string
  print a new line if the returnchar
   is found.
______________________________  */                
    fprintf(stderr,"\n ______________________________________________\n");
    fprintf(stderr,"%s\n",subroutine);
    fprintf(stderr,"%s\n",message);
    fprintf(stderr,"\n");
    
    for(k = zLo; k <= zHi; k++)
    {
        for(j = yHi; j >= yLo; j--)
        {
        for(f = n4dlo; f <=n4dhi; f++)
                {
            for(i = xLo; i <= xHi; i++)
            {
                
                    if (f == (int)TOP)      strcpy(c,"T ");
                    if (f == (int)BOTTOM)   strcpy(c,"B ");
                    if (f == (int)RIGHT)    strcpy(c,"R ");
                    if (f == (int)LEFT)     strcpy(c,"L ");
                    if (f == (int)FRONT)    strcpy(c,"F ");
                    if (f == (int)BACK)     strcpy(c,"BK");
                    fprintf(stderr,"[%d,%d,%d,%s,%d]= %4.3lf  ",
                    i,j,k,c,n5dlo, *data_array[i][j][k][f][m]);
                }
                fprintf(stderr,"\n");
            }
            fprintf(stderr,"\n");
        }
        fprintf(stderr,"\n");
    }
    fprintf(stderr,"______________________________________________\n");
       
 }
/*STOP_DOC*/
/* 
 ======================================================================*/
 #include "switches.h"
 #include "parameters.h"
 #include "functionDeclare.h"
 #include <assert.h>
/* 
 Function:  printData_4d--DEBUG: Write a 4D array to stderr.
 Filename:  commonFunctions.c

 Purpose:  Print to stderr a cell-centered, multimaterial array

 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       02/22/99   Written   
_______________________________________________________________________ */

void    printData_4d(  
          int     xLoLimit,             /* x-array lower limit              */
          int     yLoLimit,             /* y-array lower limit              */
          int     zLoLimit,             /* z-array lower limit              */
          int     xHiLimit,             /* x-array upper limit              */
          int     yHiLimit,             /* y-array upper limit              */
          int     zHiLimit,             /* z-array upper limit              */
          int     n4dlo,                /* 4th dimension lower limit        */
          int     n4dhi,                /* 4th dimension upper limit        */
          char    subroutine[],         /* name of the function             (INPUT) */
          char    message[],            /* message to the user              (INPUT) */
          double  ****data_array      ) /* data(i,j,k,m)                    (INPUT) */
        
{ 
    int i, j, k, m;
/*__________________________________
* HARDWIRED
*___________________________________*/
    m=n4dlo;
/*__________________________________
* double check inputs
*___________________________________*/
    assert ( xLoLimit >= 0 && xHiLimit < X_MAX_LIM);
    assert ( yLoLimit >= 0 && yHiLimit < Y_MAX_LIM);
    assert ( zLoLimit >= 0 && zHiLimit < Z_MAX_LIM);       
/* ______________________________
  Now print the string
  print a new line if the returnchar
   is found.
______________________________  */                
    fprintf(stderr,"\n ______________________________________________\n");
    fprintf(stderr,"%s\n",subroutine);
    fprintf(stderr,"%s\n",message);
    fprintf(stderr,"\n");
    for(k = zLoLimit; k <= zHiLimit; k++)
    {
        for(j = yHiLimit; j >= yLoLimit; j--)
        {
            for(m = n4dlo; m <= n4dhi; m++)
            {
                 for(i = xLoLimit; i <= xHiLimit; i++)
                {    
                    #if (switchDebug_printData_4d == 1)
                    fprintf(stderr,"[%d,%d,%d,%d]= %4.3lf  ",
                      i,j,k,m, data_array[m][i][j][k]);
                    #endif 
                    
                    #if (switchDebug_printData_4d == 2)                      
                    fprintf(stderr,"[%d,%d,%d,%d]= %6.5lf  ",
                      i,j,k,m, data_array[m][i][j][k]);
                    #endif
                }
               
                fprintf(stderr,"\n");
            }
            fprintf(stderr,"\n");
        }
        fprintf(stderr,"\n");
    }
    fprintf(stderr," ______________________________________________\n");
/*__________________________________
*   Quite fullwarn remarks 
*___________________________________*/      
    QUITE_FULLWARN(data_array); 
 }
/*STOP_DOC*/
/* 
 ======================================================================*/
 #include "parameters.h"
 #include "functionDeclare.h"
 #include <assert.h>
/*
 Function:  printData_3d--DEBUG: Write a 3D array to stderr.
 Filename: commonFunctions.c

 Purpose:  Print to stderr a cell-centered, single material

 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       02/22/99   Written   
_______________________________________________________________________ */

void    printData_3d(  
        int     xLoLimit,               /* x-array lower limit              */
        int     yLoLimit,               /* y-array lower limit              */
        int     zLoLimit,               /* z-array lower limit              */
        int     xHiLimit,               /* x-array upper limit              */
        int     yHiLimit,               /* y-array upper limit              */
        int     zHiLimit,               /* z-array upper limit              */
        char    subroutine[],           /* name of the function             */
        char    message[],              /* message to user                  */
        double  ***data_array   )       /* data(x,y,z)                      */
{ 
    int i, j, k;

/*__________________________________
* double check inputs
*___________________________________*/
    assert ( xLoLimit >= 0 && xHiLimit < X_MAX_LIM);
    assert ( yLoLimit >= 0 && yHiLimit < Y_MAX_LIM);
    assert ( zLoLimit >= 0 && zHiLimit < Z_MAX_LIM);       
/* ______________________________
  Now print the string
  print a new line if the returnchar
   is found.
______________________________  */                
    fprintf(stderr,"\n ______________________________________________\n");
    fprintf(stderr,"%s\n",subroutine);
    fprintf(stderr,"%s\n",message);
    fprintf(stderr,"\n");
    for(k = zLoLimit; k <= zHiLimit; k++)
    {
        for(j = yHiLimit; j >= yLoLimit; j--)
        {
            for(i = xLoLimit; i <= xHiLimit; i++)
            {
               fprintf(stderr,"[%d,%d,%d] = %4.3lf  ",
                      i,j,k, data_array[i][j][k]);
               
               /*  fprintf(stderr,"\n"); */
            }
            fprintf(stderr,"\n");
        }
        fprintf(stderr,"\n");
    }
    fprintf(stderr," ______________________________________________\n");
       
 }
/*STOP_DOC*/


/* 
 ======================================================================*/
 #include "parameters.h"
 #include "functionDeclare.h"
 #include <assert.h>
/*
 Function:  printData_1d--DEBUG: Write a 1D array to stderr.
 Filename: commonFunctions.c

 Purpose:  Print to stderr a cell-centered, single material

 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       03/7/00   Written   
_______________________________________________________________________ */

void    printData_1d(  
        int     xLoLimit,               /* x-array lower limit              */
        int     xHiLimit,               /* x-array upper limit              */
        char    subroutine[],           /* name of the function             */
        char    message[],              /* message to user                  */
        double  *data_array   )         /* data(x)                          */
{ 
    int i;

/*__________________________________
* double check inputs
*___________________________________*/
    assert ( xLoLimit >= 0 && xHiLimit < X_MAX_LIM);     
/* ______________________________
  Now print the string
  print a new line if the returnchar
   is found.
______________________________  */                
    fprintf(stderr,"\n ______________________________________________\n");
    fprintf(stderr,"%s\n",subroutine);
    fprintf(stderr,"%s\n",message);
    fprintf(stderr,"\n");
    for(i = xLoLimit; i <= xHiLimit; i++)
    {
       fprintf(stderr,"[%d] = %6.5lf  ",
              i, data_array[i]);          
    }
    fprintf(stderr,"\n______________________________________________\n");
       
 }
/*STOP_DOC*/

/* 
 ======================================================================*/
 #include "parameters.h"
 #include "functionDeclare.h"
 #include <assert.h>
 #include <math.h>
 #define    DIFFERENCE 1.0e-3
/*
 Function:  print_5d_where_computations_have_taken_place--DEBUG: Function that writes to stderr where a computation has taken place.
 Filename:  commonFunctions.c

 Purpose:  print to stderr only the entries in the data_array
    that are equal to 1.  This is mainly used in debugging the code
    specifically where computations have taken place.
    In a test array just set the value of data_array to 1.0
    and this funcion will print out that location 

 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       10/10/99   Written   
_______________________________________________________________________ */
void    print_5d_where_computations_have_taken_place(                         
    int     xLo,                        /* x-array lower               */
    int     yLo,                        /* y-array lower               */
    int     zLo,                        /* z-array lower               */
    int     xHi,                        /* x-array upper               */
    int     yHi,                        /* y-array upper               */
    int     zHi,                        /* z-array upper               */
    int     n4dlo,                      
    int     n4dhi,
    int     n5dlo, 
    int     n5dhi,
    char    subroutine[],               /* name of function             (INPUT) */
    char    message[],                  /* message to the user          (INPUT) */
    double  *****data_array,            /* data(i,j,k,f,m)              (INPUT) */
    int     ghostcells       )          /* =1 include ghostcelldata     (INPUT) */
        
{ 
    char    c[2];
    int     i, j, k, f;

/*__________________________________
*   Define limits
*___________________________________*/
    strcpy(c,"  ");
    if (ghostcells == 1)
    {   
        xLo = GC_LO(xLo);    xHi = GC_HI(xHi);
        yLo = GC_LO(yLo);    yHi = GC_HI(yHi);
 /*     zLo = GC_LO(zLo);    zHi = GC_HI(zHi); */
        zLo =   zLo;               zHi = zHi;
    }
/*__________________________________
* double check inputs
*___________________________________*/
    assert ( xLo >= 0 && xHi <= X_MAX_LIM);
    assert ( yLo >= 0 && yHi <= Y_MAX_LIM);
    assert ( zLo >= 0 && zHi <= Z_MAX_LIM);
    assert ( n5dlo >=1 && n5dhi <= N_MATERIAL);

       
/* ______________________________
  Now print the string
  print a new line if the returnchar
   is found.
______________________________  */                
    fprintf(stderr,"\n ______________________________________________\n");
    fprintf(stderr,"%s\n",subroutine);
    fprintf(stderr,"%s\n",message);
    fprintf(stderr,"\n");
    for(k = zLo; k <= zHi; k++)
    {
        for(j = yHi; j >= yLo; j--)
        {
            for(f = n4dlo; f <=n4dhi; f++)
            {
                for(i = xLo; i <= xHi; i++)
                {
                
                    if (f == (int)TOP)      strcpy(c,"T ");
                    if (f == (int)BOTTOM)   strcpy(c,"B ");
                    if (f == (int)RIGHT)    strcpy(c,"R ");
                    if (f == (int)LEFT)     strcpy(c,"L ");
                    if (f == (int)FRONT)    strcpy(c,"F ");
                    if (f == (int)BACK)     strcpy(c,"BK");
                    if ( fabs(data_array[i][j][k][f][n5dlo] - YES) <= DIFFERENCE  )
                        fprintf(stderr,"[%d,%d,%d,%s,%d]  ",i,j,k,c,n5dlo);
                    else 
                        fprintf(stderr,"              ");
                }
                fprintf(stderr,"\n");
            }
            fprintf(stderr,"\n");
        }
        /* fprintf(stderr,"\n"); */
    }
    fprintf(stderr,"______________________________________________\n");
       
 }
/*STOP_DOC*/ 
 
 /* 
 ======================================================================*/
 #include "parameters.h"
 #include "functionDeclare.h"
 #include <math.h>
 #include <assert.h>
 #define    DIFFERENCE 1.0e-3
/*
 Function:  print_4d_where_computations_have_taken_place--DEBUG: Function that writes to stderr where a computation has taken place.
 Filename:  commonFunctions.c

 Purpose:  print to stderr only the entries in the data_array
    that are equal to 1.  This is mainly used in debugging the code
    specifically where computations have taken place.
    In a test array just set the value of data_array to 1.0
    and this funcion will print out that location 

 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       10/10/99   Written   
_______________________________________________________________________ */

void    print_4d_where_computations_have_taken_place(                         
    int     xLo,                        /* x-array lower                */
    int     yLo,                        /* y-array lower                */
    int     zLo,                        /* z-array lower                */
    int     xHi,                        /* x-array upper                */
    int     yHi,                        /* y-array upper                */
    int     zHi,                        /* z-array upper                */
    int     n4dlo,                      /* 4th dimension lower limit    (INPUT) */
    int     n4dhi,
    char    subroutine[],
    char    message[],
    double  ****data_array,
    int     ghostcells       )        
{ 
    int     i, j, k, m;
/*__________________________________
*   Define limits
*___________________________________*/
    if (ghostcells == 1)
    {   
        xLo = GC_LO(xLo);    xHi = GC_HI(xHi);
        yLo = GC_LO(yLo);    yHi = GC_HI(yHi);
 /*     zLo = GC_LO(zLo);    zHi = GC_HI(zHi); */
        zLo =   zLo;               zHi = zHi;
    }
/*__________________________________
* double check inputs
*___________________________________*/
    assert ( xLo >= 0 && xHi <= X_MAX_LIM);
    assert ( yLo >= 0 && yHi <= Y_MAX_LIM);
    assert ( zLo >= 0 && zHi <= Z_MAX_LIM);

       
/* ______________________________
  Now print the string
  print a new line if the returnchar
   is found.
______________________________  */                
    fprintf(stderr,"\n ______________________________________________\n");
    fprintf(stderr,"%s\n",subroutine);
    fprintf(stderr,"%s\n",message);
    fprintf(stderr,"\n");
    for(k = zLo; k <= zHi; k++)
    {
        for(j = yHi; j >= yLo; j--)
        {
            for(m = n4dlo; m <= n4dhi; m++)
            {
                 for(i = xLo; i <= xHi; i++)
                {    
                    if ( fabs(data_array[m][i][j][k] - YES) <= DIFFERENCE  )
                        fprintf(stderr,"[%d,%d,%d,%d]  ",i,j,k,m);
                    else 
                        fprintf(stderr,"                  ");
                        
                }
               
                fprintf(stderr,"\n");
            }
            fprintf(stderr,"\n");
        }
        fprintf(stderr,"\n");
    }
    fprintf(stderr," ______________________________________________\n");
 }
/*STOP_DOC*/

 
/* 
 ======================================================================*/
 #include "parameters.h"
 #include "functionDeclare.h"
 #include <math.h>
 #include <assert.h>
 #define    DIFFERENCE 1.0e-3
/*
 Function:  print_3d_where_computations_have_taken_place--DEBUG: Function that writes to stderr where a computation has taken place.
 Filename:  commonFunctions.c

 Purpose:  print to stderr only the entries in the data_array
    that are equal to 1.  This is mainly used in debugging the code
    specifically where computations have taken place.
    In a test array just set the value of data_array to 1.0
    and this funcion will print out that location 

 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       10/10/99   Written   
_______________________________________________________________________ */

void    print_3d_where_computations_have_taken_place(                         
    int     xLo,                        /* x-array lower limit              */
    int     yLo,                        /* y-array lower limit              */
    int     zLo,                        /* z-array lower limit              */
    int     xHi,                        /* x-array upper limit              */
    int     yHi,                        /* y-array upper limit              */
    int     zHi,                        /* z-array upper limit              */
    char    subroutine[],               /* name of function                 (INPUT) */
    char    message[],                  /* message to the user              (INPUT) */
    double  ***data_array,              /* data(i,j,k)                      (INPUT) */
    int     ghostcells       )        
{ 
    int     i, j, k;


/*__________________________________
*   Define limits
*___________________________________*/
    if (ghostcells == 1)
    {   
        xLo = GC_LO(xLo);    xHi = GC_HI(xHi);
        yLo = GC_LO(yLo);    yHi = GC_HI(yHi);
 /*     zLo = GC_LO(zLo);    zHi = GC_HI(zHi); */
        zLo =   zLo;               zHi = zHi;
    }
/*__________________________________
* double check inputs
*___________________________________*/
    assert ( xLo >= 0 && xHi <= X_MAX_LIM);
    assert ( yLo >= 0 && yHi <= Y_MAX_LIM);
    assert ( zLo >= 0 && zHi <= Z_MAX_LIM);
       
/* ______________________________
  Now print the string
  print a new line if the returnchar
   is found.
______________________________  */                
    fprintf(stderr,"\n ______________________________________________\n");
    fprintf(stderr,"%s\n",subroutine);
    fprintf(stderr,"%s\n",message);
    fprintf(stderr,"\n");
    for(k = zLo; k <= zHi; k++)
    {
        for(j = yHi; j >= yLo; j--)
        {
             for(i = xLo; i <= xHi; i++)
            {    
                if ( fabs(data_array[i][j][k] - YES) <= DIFFERENCE  )
                    fprintf(stderr,"[%d,%d,%d]  ",i,j,k);
                else 
                    fprintf(stderr,"                  ");
            }    
            fprintf(stderr,"\n");
        }
        fprintf(stderr,"\n");
    }
    fprintf(stderr," ______________________________________________\n");
 }
/*STOP_DOC*/ 
