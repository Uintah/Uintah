/* 
 ======================================================================*/
#include "functionDeclare.h"
#include "parameters.h"
#include "switches.h"
#include <assert.h>
#include <time.h>
#include <sys/types.h>
#include <math.h>

/*
 Filename: delta_press_CC.c
 Name: delta_press_CC()     

 Purpose:
   This function calculates the cell-centered, time n+1, change in the 
   equilbration pressure. 
   
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       03/24/99    

 IN/OUT args/commons     Units      Description
 ---------------         -----      ----------- 
 x,y,z LoLimit           int        lower array limits
 x,y,z HiLimit           int        upper array limits
 Vol_L_CC           ****double      Lagragian cell-centered volume
 Vol_CC              ***double      Cell-centered volume
 u,v,wvel_FC        ****doublt      x,y,z face-centered velocity components
 del(X,Y,Z)             double      distance between cell centers
 delt                   double      delta time
 nMaterials             int         Material 
  
 Prerequisites:
 
 ---------------------------------------------------------------------  */
void delta_press_CC(    xLoLimit,       yLoLimit,       zLoLimit,
                        xHiLimit,       yHiLimit,       zHiLimit,
                        Vol_CC,         rho_CC,         speedSound,
                        delt,           delX,           delY,           delZ,
                        nMaterials)
                
    int     xLoLimit,                   /* x-array lower limit              */
            yLoLimit,                   /* y-array lower limit              */
            zLoLimit,                   /* z-array lower limit              */
            xHiLimit,                   /* x-array upper limit              */
            yHiLimit,                   /* y-array upper limit              */
            zHiLimit,                   /* z-array upper limit              */
            nMaterials;
            
    double  
            delX,                       /* distance/cell, xdir              */
            delY,                       /* distance/cell, ydir              */
            delZ,                       /* distance/cell, zdir              */
            delt,                       /* delta t                          */
            ***Vol_CC,                  /* cell-centered volume             */
                                        /* (i,j,k)                          */
            ****rho_CC,                 /* Cell-centered density            */
                                        /* (x, y, z, material )             */
            ****speedSound;             /* speed of sound (x,y,z, material) */                      

{
    int i, j, k,m;                      /*   loop indices  locators         */            
                          
    
    double  topface, bottomface,
            rightface, leftface,
            frontface, backface;        /* temp symbols to represent terms  */
    
    time_t start,secs;                  /* timing variables                */
    start = time(NULL); 

/*__________________________________
* Check that the inputs are reasonable
*___________________________________*/
    assert ( xLoLimit > 0 && xHiLimit < X_MAX_LIM);
    assert ( yLoLimit > 0 && yHiLimit < Y_MAX_LIM);
    assert ( zLoLimit > 0 && zHiLimit < Z_MAX_LIM);
    assert ( delt > 0 );
    assert ( delX > 0 );
    assert ( delY > 0 );
    assert ( delZ > 0 );

/*______________________________________________________________________
*
*_______________________________________________________________________*/
    for ( i = xLoLimit; i <= xHiLimit; i++)
    {
        for ( j = yLoLimit; j <= yHiLimit; j++)
        {
            for ( k = zLoLimit; k <= zHiLimit; k++)
            {            

                /*__________________________________
                * top and bottom face contributions
                *___________________________________*/
               topface      = delX*delZ*vvel_FC[i][j][k][TOP][m];
               bottomface   = delX*delZ*vvel_FC[i][j][k][BOTTOM][m];
                /*__________________________________
                * left and right face contributions
                *___________________________________*/
               leftface     = delY*delZ*uvel_FC[i][j][k][RIGHT][m];
               rightface    = delY*delZ*uvel_FC[i][j][k][LEFT][m];
                /*__________________________________
                * front and back face contributions
                *___________________________________*/
               frontface    = delX*delY*wvel_FC[i][j][k][FRONT][m];
               backface     = delX*delY*wvel_FC[i][j][k][BACK][m];
                /*__________________________________
                * change cell index and signs on
                * pressure terms
                *___________________________________*/
                Vol_L_CC[i][j][k][m] = Vol_CC[i][j][k] + delt*
                (topface + bottomface + leftface + rightface 
                 + frontface + backface );

            }
        }
    }
/*______________________________________________________________________
*   DEBUGGING AND STOP WATCH INFORMATION
*_______________________________________________________________________*/
/*__________________________________
* No printout debugging and timing info
*___________________________________*/ 
#if switchDebug_vol_L_FC
    printData_3d(   xLoLimit,       yLoLimit,       zLoLimit,
                       xHiLimit,       yHiLimit,       zHiLimit,
                       "lagrangian_vol,Vol_L_CC,       Vol_L_CC,
                        m);
                       
#endif 
       
#if sw_vel_face
    stopwatch("lagragian volume",start);
#endif

}
