/* 
 ======================================================================*/
#include <math.h>
#include <stdio.h>
#include "macros.h"
#include "switches.h"
#include "nrutil+.h"
/*
 Function:  Inviscisd Burgers--MISC: compares some of the test problem resul
            with an exact solution 
 Filename:  Inviscid_burgers.c
  
 Purpose:
            To be filled in
  
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       03/02/99    

References:
Numerical Computation of Internal and External flows Vol 2 pg 200-??

 ---------------------------------------------------------------------  */
 void inviscid_burgers(
        int     qLoLimit,               /* x-array lower limit              */
        int     qHiLimit,               /* x-array upper limit              */
        double  u0,
        double  A,
        double  uL,                     /* left cell velocity               */
        double  uR,                     /* right cell velocity              */
        double  u1,
        double  u2,
        double  delQ,                   /* distance/cell, xdir              */
        double  x0,                      /* origin of disconinuity           */
        double  t,                      /* time                             */
        double  ****qvel_CC,            /* u-cell-centered velocity         */
        float  *data_array2)
{
    int     i;
    double  C, B, x, x1;
    double  ts, L;
    double  xLo, xHi, junk;
    double  qmax;
      
/*______________________________________________________________________
* Do some preliminary calculations that are needed for each test problem
*_______________________________________________________________________*/
    #if (testproblem == 1)
        C   = 0.5 * (uL + uR);
        fprintf(stderr,"Discontinuity location %g \n",(x0 + C*t));
    #endif       
    
    #if (testproblem == 2) 
        B   = 4.0 * A * x0/M_PI;
        xLo = u0*t;
        xHi = u0*t + sqrt(B*t);
        junk=  u0 + sqrt(B/t);
        fprintf(stderr, "B = %g, t = %g\n",B,t);
        fprintf(stderr, "Height of point should be sqrt(B/t) = %g\n",junk); 
        
        qmax= 0.0;
        for ( i = (qLoLimit); i <= (qHiLimit); i++)
        { 
            qmax= DMAX(qmax, qvel_CC[i][1][1][1]);
        }
        fprintf(stderr, "actual Height %g, ratio of exact/actual height=%g\n",qmax,junk/qmax);
        
    #endif
    
    #if (testproblem == 3) 
        L   = 5.0 * delQ;  
        x1  = x0 - L; 
        ts  = ( L)/(u1 - u2);
        fprintf(stderr,"tshock = %g, time = %g\n",ts, t);
    #endif
/*__________________________________
*   Now solve for the exact solution
*___________________________________*/
    for ( i = qLoLimit; i <= qHiLimit; i++)
    { 
         x = (double)(i-qLoLimit) * delQ;
        /*__________________________________
        * Initial shock discontinutiy  
        *___________________________________*/
        #if (testproblem == 1)

            if(x <= (x0 + C*t) )
                data_array2[i] = uL;
                
            if(x > (x0 + C*t) )
                data_array2[i] = uR;
        #endif
        /*__________________________________
        *   Sinusoidal wave profile
        *___________________________________*/
        #if (testproblem == 2) 
            data_array2[i] = 0.0;
            if( (xLo <= x) && (x < xHi ) )
                data_array2[i] = x/t;    
        #endif

         /*__________________________________
         *   Initial Linear Distribution
         *___________________________________*/
         #if (testproblem == 3)
            data_array2[i] = 0.0;
            if(t > ts)
            {
                 if( x < x1 + ((u1 + u2)/2.0)*t )
                     data_array2[i] = u1;

                  if( x > x1 + ((u1 + u2)/2.0)*t )
                     data_array2[i] = u2; 
            }        
         #endif
         /*__________________________________
         *   Expansion Wave
         *___________________________________*/
         #if (testproblem == 4)
            x = x -x0; 
            
            if(x/t < u1)
                data_array2[i] = u1;

            if( (u1<= x/t) && (x/t <= u2))
                data_array2[i] = x/t;
                
            if(x/t > u2)
                data_array2[i] = u2;
         #endif       

    }
/*__________________________________
*   Quite fullwarn stuff
*___________________________________*/
    u0 = u0;    uL = uL;    uR = uR;
    u1 = u1;    u2 = u2;    x0 = x0;
       
    C  = C;     B  = B;     x1 = x1;
    ts = ts;    L  = L;
    xLo= xLo;  xHi= xHi;    junk = junk;
    qmax= qmax;
    QUITE_FULLWARN(qvel_CC);
}
/*STOP_DOC*/ 
