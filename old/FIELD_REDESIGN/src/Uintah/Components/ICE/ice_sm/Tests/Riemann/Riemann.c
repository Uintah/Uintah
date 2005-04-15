
/* 
 ======================================================================*/
#include <math.h>
/*
 Function:  p2_p1_ratio--MISC: computes p2/p1
 Filename:  Riemann.c
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       02/16/00    
 
 This is function is used by the secant method.
 ---------------------------------------------------------------------  */
    double p2_p1_ratio(
            double  gamma,
            double  p4_p1,
            double  p2_p1_guess,
            double  a4,
            double  a1,
            double  u4,
            double  u1  )
{         
    double  gamma_ratio1,
            gamma_ratio2,
            sqroot,
            exponent,
            fraction,
            boxed_quantity;
/*__________________________________
*
*___________________________________*/            
            
        gamma_ratio1    = (gamma + 1.0)/( 2.0 * gamma);
        sqroot          = sqrt( gamma_ratio1 * (p2_p1_guess - 1.0) + 1.0 );
        fraction        = (p2_p1_guess - 1.0)/sqroot;

        boxed_quantity  = u4 - u1 - (a1/gamma) * fraction;
        
        gamma_ratio2    = (gamma - 1.0)/(2.0 * a4);
        exponent        = -2.0*gamma/(gamma - 1.0);
        return p4_p1 - p2_p1_guess * pow( (1.0 + gamma_ratio2 * boxed_quantity), exponent);
      
}

/* 
 ======================================================================*/
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#define CONVERGENCE 1e-6
#define MAX_ITER 50 
/*
 Function:  Solve_Riemann_problem--MISC: Computes the solution to the shock tube problme 
 Filename:  Riemann.c
  
 Purpose:
            Solves the shock tube problem

 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       02/16/00    
Implementation Notes:
    The notation comes from the reference
       
Reference:
    Computational GasDynamics by C.B. Laney, 1998, pg 73
 ---------------------------------------------------------------------  */
 
 void Solve_Riemann_problem(
        int     qLoLimit,                /* array lower limit                */
        int     qHiLimit,                /* upper limit                      */
        double  delQ,
        double  time,
        double  gamma,
        double  p1,                     /* pressure  right of the diaphram  */
        double  rho1,                   /* density                          */
        double  u1,                     /* velocity                         */
        double  a1,
        double  p4,                     /* pressure  Left of the diaphram   */
        double  rho4,                   /* density                          */
        double  u4,                     /* velocity                         */ 
        double  a4,
        double  *u_Rieman,              /* exact solution velocity          */
        double  *a_Rieman,              /* exact solution speed of sound    */
        double  *p_Rieman,              /* exact solution pressure          */
        double  *rho_Rieman,             /* exact solution density           */  
        double  *T_Rieman,
        double  R)                      /* ideal gas constant               */      
{
    int     i, iter;
            
    double  gamma_ratio1,               /* variables that make writing the eqs easy */
            gamma_ratio2,
            exponent,
            sqroot,
            fraction;
    double 
            origin,                     /* origin where the diaphram was initially  */
            S,                          /* shock velocity                   */  
            x,       
            xtemp,    
            xshock,                     /* location of the shock            */        
            xcontact,                   /* location of the contact          */      
            xexpansion_head,            /* Location of the expansion head   */
            xexpansion_tail;            /* location of the expansion tail   */

    double  p2, p3,                     /* quantities in the various locations*/
            u2, u3,
            a2, a3,
            rho2, rho3,
            p2_p1,
            p4_p1;
            
    double  p2_p1_guess_old,
            p2_p1_guess_new,
            p2_p1_guess,
            
            p2_p1_guess0,
            p2_p1_guess00;
            
    double  delta,                       /* used by the secant method        */
            fudge;
    char    should_I_write_output;            
/*START_DOC*/
/*__________________________________
*   Compute some stuff
*___________________________________*/
    p4_p1           = p4/p1;
    origin          = (double)((qHiLimit - qLoLimit + 1)/2.0 ) * delQ;
    
    p2_p1_guess0    = 0.5* p4_p1;
    p2_p1_guess00   = 0.05* p4_p1;
    iter            = 0;    
    fudge           = 0.99;
/*______________________________________________________________________
*   Use the secant method to solve for pressure ratio across the shock
*                   p2_p1
*   See Numerical Methods by Hornbeck, pg 71
*_______________________________________________________________________*/
/*__________________________________
*   Step 1 Compute the pressure ratio
*   across the shock
*   Need to add iterative loop
*___________________________________*/
    delta               = p2_p1_guess0 - p2_p1_guess00;
    p2_p1_guess         = p2_p1_guess0;
    p2_p1_guess_old     = p2_p1_ratio( gamma,  p4_p1, p2_p1_guess00, a4, a1, u4, u1  );
    while (fabs(delta) > CONVERGENCE && iter < MAX_ITER)
   {
        p2_p1_guess_new = p2_p1_ratio( gamma,  p4_p1, p2_p1_guess, a4, a1, u4, u1  );
        delta           = -p2_p1_guess_new/( (p2_p1_guess_new - p2_p1_guess_old)/delta );
        p2_p1_guess     = p2_p1_guess + delta;

        p2_p1_guess_old = p2_p1_guess_new;
        /* fprintf(stderr,"p2_p1_guess = %g delta %g \n",p2_p1_guess, delta); */
        iter ++;
    }
    p2_p1               = p2_p1_guess;
/*______________________________________________________________________
*   Now compute the properties
*   that are constant in each section
*_______________________________________________________________________*/
    gamma_ratio1        = (gamma + 1.0)/( 2.0 * gamma);
    sqroot              = sqrt( gamma_ratio1 * (p2_p1 - 1.0) + 1.0 );
    fraction            = (p2_p1 - 1.0)/sqroot;
    u2                  = u1 + (a1/gamma) * fraction;
 
    gamma_ratio1        = (gamma + 1.0)/(gamma - 1.0);
    fraction            = (gamma_ratio1 + p2_p1)/( 1.0 + gamma_ratio1 * p2_p1);
    a2                  = sqrt(pow(a1,2.0) * p2_p1 * fraction );
    p2                  = p2_p1 * p1;
    rho2                = gamma * p2/pow(a2,2);;
 
    /*___*/
    u3                  = u2;
    p3                  = p2;
    
    gamma_ratio1        = (gamma -1.0)/2.0;
    a3                  = gamma_ratio1*(u4 + a4/gamma_ratio1 - u3);
    rho3                = gamma * p3/pow(a3,2);
    
/*______________________________________________________________________
*   Step 2 Compute the shock and expansion locations all relative the orgin
*   Write the data to an array
*_______________________________________________________________________*/
    gamma_ratio1        = (gamma + 1.0)/( 2.0 * gamma);
    sqroot              = sqrt(gamma_ratio1 * (p2_p1 - 1.0) + 1.0);
    S                   = u1 + a1 * sqroot;
    
    xshock              = origin + (S               * time);
    xcontact            = origin + (u3              * time);
    xexpansion_head     = origin + ((u4 - a4)  * time);
    xexpansion_tail     = origin + ((u3 - a3)  * time);
    
/*__________________________________
*   Now write all of the data to the arrays
*___________________________________*/

    for( i = qLoLimit; i <= qHiLimit; i++)
    {
        x = (double) (i-qLoLimit) * delQ;
        /*__________________________________
        *   Region 1
        *___________________________________*/
        if (x >=xshock)
        {
            u_Rieman[i]     = u1; 
            a_Rieman[i]     = a1;
            p_Rieman[i]     = p1;
            rho_Rieman[i]   = rho1;
            T_Rieman[i]     = pow(a1,2)/(gamma * R);
        }
        /*__________________________________
        *   Region 2
        *___________________________________*/
        if ( (xcontact < x) && (x < xshock) )
        {
            u_Rieman[i]     = u2; 
            a_Rieman[i]     = a2;
            p_Rieman[i]     = p2;
            rho_Rieman[i]   = rho2;
            T_Rieman[i]     = pow(a2,2)/(gamma * R);
        }

        /*__________________________________
        *   Region 3
        *___________________________________*/
        if ( (xexpansion_tail <= x) && (x <= xcontact) )
        {
            u_Rieman[i]     = u3; 
            a_Rieman[i]     = a3;
            p_Rieman[i]     = p3;
            rho_Rieman[i]   = rho3;
            T_Rieman[i]     = pow(a3,2)/(gamma * R);
        }

        /*__________________________________
        *   Expansion fan Between 3 and 4
        *___________________________________*/
        if ( (xexpansion_head <= x) && (x < xexpansion_tail) )
        {
            xtemp           = (x - xexpansion_tail);
            exponent        = (2.0 * gamma)/( gamma - 1.0 );
            gamma_ratio2    = 2.0/(gamma + 1.0);
            u_Rieman[i]     =  gamma_ratio2 * ( xtemp/(time + 1.0e-100) + ((gamma - 1.0)/2.0) * u4 + a4);
            
            a_Rieman[i]     =  u_Rieman[i] - xtemp/(time + 1.0e-100);
            p_Rieman[i]     =  p4 * pow( (a_Rieman[i]/a4), exponent);
            
            exponent        = (2.0)/( gamma - 1.0 );
            rho_Rieman[i]   =  rho4 * pow( (a_Rieman[i]/a4), exponent);
            
            T_Rieman[i]     =  pow(a_Rieman[i],2)/(gamma * R);
        }

        /*__________________________________
        *   Region 4
        *___________________________________*/
        if (x <xexpansion_head)
        {
            u_Rieman[i]     = u4;
            a_Rieman[i]     = a4;
            p_Rieman[i]     = p4;
            rho_Rieman[i]   = rho4;
            T_Rieman[i]     = pow(a4,2)/(gamma*R);
        }
        
    }
    should_I_write_output = *getenv("SHOULD_I_WRITE_OUTPUT");
    if ( should_I_write_output == '1')
    {    
        fprintf(stderr,"________________________________________________\n"); 
        fprintf(stderr," p2_p1: %f\n",p2_p1);
        fprintf(stderr," u4: %f,\t  u3: %f,\t  u2: %f,\t  u1: %f\n",u4, u3, u2, u1);
        fprintf(stderr," a4: %f,\t  a3: %f,\t  a2: %f,\t  a1: %f\n",a4, a3, a2, a1);
        fprintf(stderr," p4: %f,\t  p3: %f,\t  p2: %f,\t  p1: %f\n",p4, p3, p2, p1); 
        fprintf(stderr," rho4: %f,\t  rho3: %f,\t  rho2: %f,\t  rho1: %f\n",rho4, rho3, rho2, rho1); 
        fprintf(stderr," shock Velocity: \t %f \n",S);
        fprintf(stderr," LHS expansion vel: \t %f \n",(u4 - a4) );
        fprintf(stderr," RHS expansion vel: \t %f \n",(u3 - a3) );    
        fprintf(stderr," Xlocations\n");
        fprintf(stderr,"%f,  %f,  %f,   %f\n",xexpansion_head, xexpansion_tail, xcontact, xshock);
        fprintf(stderr,"________________________________________________\n");   
    }    

}

/*STOP_DOC*/ 
