/* 
 ======================================================================*/
#include <math.h>
#include <assert.h>
#include <stdio.h>
#include "nrutil+.h"
#include "parameters.h"
#include "switches.h"
#include "macros.h"
/*
 Function:  convective_viscous_term
 Filename:  convective_viscous_term.c
 Purpose:  
    This function calculates the terms (F) and(G)which corresponds to the viscous
    convective terms and bodyforce terms in the X and Y momentum equation.  This comes
    directly from pg 34-35 of reference (2)
 
 References:
    Casulli, V. and Greenspan, D, Pressure Method for the Numerical Solution
    of Transient, Compressible Fluid Flows, International Journal for Numerical
    Methods in Fluids, Vol. 4, 1001-1012, (1984)
    
(2) Bulgarelli, U., Casulli, V. and Greenspan, D., "Pressure Methods for the 
    Numerical Solution of Free Surface Fluid Flows, Pineridge Press
    (1984)
            
    Note that this code is only a testbed for the verification of the 
    iterative pressure solver.     
       
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       08/30/99

 ---------------------------------------------------------------------  */
 
void    convective_viscous_terms(
        int     xLoLimit,               /* x-array lower limit              */
        int     yLoLimit,               /* y-array lower limit              */
        int     zLoLimit,               /* z-array lower limit              */
        int     xHiLimit,               /* x-array upper limit              */
        int     yHiLimit,               /* y-array upper limit              */
        int     zHiLimit,               /* z-array upper limit              */
        double  delX,                   /* distance/cell, xdir              */
        double  delY,                   /* distance/cell, ydir              */
        double  delZ,                   /* distance/cell, zdir              */
        double  ******uvel_FC,          /* u-face-centered velocity         */
                                        /* uvel_FC(x,y,z,face,)             */
        double  ******vvel_FC,          /*  v-face-centered velocity        */
                                        /* vvel_FC(x,y,z, face,)            */
        double  ******wvel_FC,          /* w face-centered velocity         */
                                        /* wvel_FC(x,y,z,face,)             */
        double  delt,                   /* delta t                          */
        double  ****F,                  /* viscous, convective and body force*/
                                        /* term for the x-momentum eq       */
        double  ****G  )                /* viscous, convective and body force*/
                                        /* term for the y-momentum eq       */ 

{
    int     i, j, k,m;
    double  
            A, B, C, D,                 /* temporary term used for big eqs  */
            g,                          /* gravity                          */
            nu;                         /* viscosity                        */
        
/*__________________________________
* double check inputs
*___________________________________*/
    assert ( xLoLimit-1 >= 0 && xHiLimit+1 <= X_MAX_LIM);
    assert ( yLoLimit-1 >= 0 && yHiLimit+1 <= Y_MAX_LIM);
    assert ( zLoLimit-1 >= 0 && zHiLimit+1 <= Z_MAX_LIM);
    
 /*__________________________________
 *  HARDWIRE some variables
 *___________________________________*/
    g   = 9.81;
    nu  = 0.4;
    m   = 1;
/*______________________________________________________________________
*   Step 1 determine the face-centered
*   velocities for the initial interation that are normally not calculated
*   eq. 3.7
*_______________________________________________________________________*/   
    for ( k = zLoLimit; k <= zHiLimit; k++)
    {
        for ( j = yLoLimit; j <= yHiLimit; j++)
        {
            for ( i = xLoLimit; i <= xHiLimit; i++)
            {
                *uvel_FC[i][j][k][TOP][m]  = 
                            (*uvel_FC[i][j][k][RIGHT][m]     + *uvel_FC[i][j][k][LEFT][m]
                         +  *uvel_FC[i][j+1][k][RIGHT][m]    + *uvel_FC[i][j+1][k][LEFT][m] )/4.0;

                *vvel_FC[i][j][k][RIGHT][m]= 
                            (*vvel_FC[i][j][k][TOP][m]       + *vvel_FC[i][j][k][BOTTOM][m]
                         +  *vvel_FC[i+1][j][k][TOP][m]      + *vvel_FC[i+1][j][k][BOTTOM][m] )/4.0;                  
     
/*______________________________________________________________________
* Calculate (F), equation 3.6 of reference (2)
*_______________________________________________________________________*/
                if ( *uvel_FC[i][j][k][RIGHT][m] >= 0.0 )
                    A = *uvel_FC[i][j][k][RIGHT][m] * (*uvel_FC[i][j][k][RIGHT][m] - *uvel_FC[i][j][k][LEFT][m] )/delX;
                           
                else
                    A = *uvel_FC[i][j][k][RIGHT][m] * (*uvel_FC[i+1][j][k][RIGHT][m] - *uvel_FC[i][j][k][LEFT][m] )/delX;
                

                if ( *vvel_FC[i][j][k][RIGHT][m] >= 0.0 )
                    B = *vvel_FC[i][j][k][RIGHT][m] * (*uvel_FC[i][j][k][RIGHT][m] - *uvel_FC[i][j-1][k][RIGHT][m] )/delY;
                           
                else
                    B = *vvel_FC[i][j][k][RIGHT][m] * (*uvel_FC[i][j+1][k][RIGHT][m] - *uvel_FC[i][j][k][RIGHT][m] )/delY;
                

                C = *uvel_FC[i+1][j][k][RIGHT][m] - 2.0 * *uvel_FC[i][j][k][RIGHT][m] + *uvel_FC[i][j][k][LEFT][m];
                C = C/pow(delX,2);

                D = *uvel_FC[i][j+1][k][RIGHT][m] - 2.0 * *uvel_FC[i][j][k][RIGHT][m] + *uvel_FC[i][j-1][k][RIGHT][m];
                D = D/pow(delY,2);

                F[i][j][k][RIGHT] = (A + B - nu * ( C + D) );
    

/*__________________________________
*   Calculate G
*___________________________________*/

                if ( *uvel_FC[i][j+1][k][TOP][m] >= 0.0 )
                    A = *uvel_FC[i][j+1][k][TOP][m] * (*vvel_FC[i][j][k][TOP][m] - *vvel_FC[i-1][j][k][TOP][m] )/delX;
                           
                else
                    A = *uvel_FC[i][j+1][k][TOP][m] * (*vvel_FC[i+1][j][k][TOP][m] - *vvel_FC[i][j][k][TOP][m] )/delX;
                

                if ( *vvel_FC[i][j][k][TOP][m] >= 0.0 )
                    B = *vvel_FC[i][j][k][TOP][m] * (*vvel_FC[i][j][k][TOP][m] - *vvel_FC[i][j][k][BOTTOM][m] )/delY;
                           
                else
                    B = *vvel_FC[i][j][k][TOP][m] * (*vvel_FC[i][j+1][k][TOP][m] - *vvel_FC[i][j][k][TOP][m] )/delY;
                

                C = *vvel_FC[i+1][j][k][TOP][m] - 2.0 * *vvel_FC[i][j][k][TOP][m] + *vvel_FC[i-1][j][k][TOP][m];
                C = C/pow(delX,2);

                D = *vvel_FC[i][j+1][k][TOP][m] - 2.0 * *vvel_FC[i][j][k][TOP][m] + *vvel_FC[i][j][k][BOTTOM][m];
                D = D/pow(delY,2);

                G[i][j][k][TOP] = (A + B - nu * ( C + D) + g );
   
            }
        }
    }  
/*______________________________________________________________________
*   DEBUGGING PRINTOUTS
*_______________________________________________________________________*/  
#if switchDebug_convective_viscous_terms
    printData_4d(       GC_LO(xLoLimit),    yLoLimit,       zLoLimit,
                        GC_HI(xHiLimit),    yHiLimit,       zHiLimit,
                        RIGHT,              RIGHT,
                       "convective_viscous_terms",     
                       "F",                 F);    
    
    printData_4d(       GC_LO(xLoLimit),    yLoLimit,       zLoLimit,
                        GC_HI(xHiLimit),    yHiLimit,       zHiLimit,
                        TOP,                TOP,
                       "convective_viscous_terms",     
                       "G",                 G);    
#endif
/*__________________________________
*   Quite fullwarn remarks that is compiler
*   independent
*___________________________________*/
    QUITE_FULLWARN(delZ);                   QUITE_FULLWARN(*wvel_FC[0][0][0][1][1]);    
    QUITE_FULLWARN(delt);   
}    
          
