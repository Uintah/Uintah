
/* 
 ======================================================================*/
#include <math.h>
/*
 Function:  r_r2_function--MISC: computes radius ratio in equation 11.15
 Filename:  blastWave.c
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       03/31/00    
 This is function is used by the secant method to compute V
 Reference: Similarity and Dimensional Methods in Mechanics, Sedov, 1959 pg 219
 ---------------------------------------------------------------------  */
    double r_r2_function(
            double  gamma,
            double  nu,
            double  r,
            double  r2,
            double  V_guess)
{         
    double  gPlus, gMinus,  nuPlus,      /* Abbrevations                     */
            alpha1, alpha2,             /* exponents                        */
            A, B, C,                    /* Temporary stuff                  */
            T1, T2, T3,                 /* terms 1-3 in equation            */
            RHS, LHS;                   /* both sides of the equation       */
/*__________________________________
* define the exponents of eq 11.15
*___________________________________*/  
    gPlus   = gamma + 1.0;
    gMinus  = gamma - 1.0;
    nuPlus  = nu + 2.0;
    
    
    alpha2  = (1.0 - gamma )/ ( 2.0 * gMinus + nu );
    A       = ( nuPlus * gamma )/( 2.0 + nu * gMinus);
    B       = (2.0 * nu * (2.0 - gamma) )/( gamma * pow( nuPlus, 2) );
    C       = B - alpha2;
    alpha1  = A * C;
             
/*__________________________________
*   Now form r/r2 eq 11.15a
*___________________________________*/    
    A       = V_guess * ( ( nuPlus * gPlus )/4.0 );
    T1      = pow( A, (-2.0/nuPlus) );
    
    A       = (gPlus/gMinus) * ( V_guess * ( (nuPlus * gamma )/2.0 ) - 1.0 );
    T2      = pow( A, -alpha2 );
    
    A       = (nuPlus * gPlus )/( nuPlus * gPlus - 2.0 * (2.0 + nu * gMinus) );
    B       = 1.0 - V_guess * ( (2.0 + nu * gMinus)/2.0 );
    T3      = pow( A * B, -alpha1 ); 
    
    RHS     = T1 *T2 * T3; 
    LHS     = r/r2;             
 
    return RHS - LHS;
}

/* 
 ======================================================================*/
#include <math.h>
/*
 Function:  rho_rho2_function--MISC: computes the density ratio, equation 11.15 of reference
 Filename:  blastWave.c
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       03/31/00    
 Reference: Similarity and Dimensional Methods in Mechanics, Sedov, 1959 pg 219
 ---------------------------------------------------------------------  */
    double rho_rho2_function(
            double  gamma,
            double  nu,
            double  V)
{         
    double  gPlus, gMinus,  nuPlus,      /* Abbrevations                     */
            alpha1, alpha2, 
            alpha3, alpha4, alpha5,     /* exponents                        */
            A, B, C,                    /* Temporary stuff                  */
            T1, T2, T3,                 /* terms 1-3 in equation            */
            RHS;                        /* right hand sides of the equation */
/*__________________________________
* define the exponents of eq 11.15
*___________________________________*/  
    gPlus   = gamma + 1.0;
    gMinus  = gamma - 1.0;
    nuPlus  = nu + 2.0;
    
   alpha2  = (1.0 - gamma )/ ( 2.0 * gMinus + nu );
    A       = ( nuPlus * gamma )/( 2.0 + nu * gMinus);
    B       = (2.0 * nu * (2.0 - gamma) )/( gamma * pow( nuPlus, 2) );
    C       = B - alpha2;
    alpha1  = A * C;
        
    alpha3  = nu/( 2.0 * gMinus + nu );
    alpha4  = alpha1 * nuPlus/( 2.0 - gamma );
    alpha5  = 2.0/(gamma - 2.0);
             
/*__________________________________
*   Now form rho/rho2 eq 11.15a
*___________________________________*/       
    A       = (gPlus/gMinus) * ( V * ( (nuPlus * gamma )/2.0 ) - 1.0 );
    T1      = pow( A, alpha3 );

    A       = (gPlus/gMinus) * ( 1.0 - ( nuPlus/2.0 ) * V);
    T2      = pow( A, alpha5 );
    
    A       = (nuPlus * gPlus )/( nuPlus * gPlus - 2.0 * (2.0 + nu * gMinus) );
    B       = 1.0 - V * ( (2.0 + nu * gMinus)/2.0 );
    T3      = pow( A * B, alpha4 ); 
    
    RHS     = T1 *T2 * T3;             
 
    return RHS;    
}


/* 
 ======================================================================*/
#include <math.h>
/*
 Function:  press_press2_function--MISC: computes the pressure ratio equation 11.15
 Filename:  blastWave.c
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       03/31/00    
 
 Notaion taken directly from the reference.
 Reference: Similarity and Dimensional Methods in Mechanics, Sedov, 1959 pg 219
 ---------------------------------------------------------------------  */
    double press_press2_function(
            double  gamma,
            double  nu,
            double  V)
{         
    double  gPlus, gMinus,  nuPlus,      /* Abbrevations                     */
            alpha1, alpha2,             /* exponents                        */
            alpha4, alpha5,
            A, B, C,                    /* Temporary stuff                  */
            T1, T2, T3,                 /* terms 1-4 in equation            */
            RHS;                        /* right-hand-side of the equation  */
/*__________________________________
* define the exponents of eq 11.15
*___________________________________*/  
    gPlus   = gamma + 1.0;
    gMinus  = gamma - 1.0;
    nuPlus  = nu + 2.0;
    
    alpha2  = (1.0 - gamma )/ ( 2.0 * gMinus + nu );
    A       = ( nuPlus * gamma )/( 2.0 + nu * gMinus);
    B       = (2.0 * nu * (2.0 - gamma) )/( gamma * pow( nuPlus, 2) );
    C       = B - alpha2;
    alpha1  = A * C;
    alpha4  = alpha1 * nuPlus/( 2.0 - gamma );    
    alpha5  = 2.0 / ( gamma - 2.0 );
             
/*__________________________________
*   Now form r/r2 eq 11.15a
*___________________________________*/    
    A       = V * ( ( nuPlus * gPlus )/4.0 );
    T1      = pow( A, ( (2.0 * nu)/nuPlus) );
    
    A       = (gPlus/gMinus) * ( 1.0 - V * ( nuPlus/2.0 ) );
    T2      = pow( A, (alpha5 + 1.0) );
    
    A       = (nuPlus * gPlus )/( nuPlus * gPlus - 2.0 * (2.0 + nu * gMinus) );
    B       = 1.0 - V * ( (2.0 + nu * gMinus)/2.0 );
    T3      = pow( (A * B), (alpha4 - 2.0 * alpha1) ); 
    
    RHS     = T1 *T2 * T3;             
 
    return RHS;
}      

/* 
 ======================================================================*/
#include <math.h>
#include "parameters.h"
#include "functionDeclare.h"
#define CONVERGENCE 1e-5
#define MAX_ITER 50 
/*
 Function:  BlastWave_Find_hat_variables--MISC: Computes radius,
            density, velocity, pressure and temperature ratios
            in equation 11.15
            
 Filename:  blastWave.c
  
 Purpose:
            Solves the blast wave problem in 2-D

 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       04/01/00    
Implementation Notes:
    The notation used in this code comes directly from the reference
       
Reference:
     Reference: Similarity and Dimensional Methods in Mechanics, Sedov, 1959
Steps:
    - Use the secant method to solve for V in the r/r2 equation of eq 11.15
    - Use V to compute the velocity, density, pressure and temperature ratio.
 ---------------------------------------------------------------------  */ 
 void BlastWave_compute_eq_11_15(
        double  r,                      /* local radius                     */
        double  r2,                     /* radius of shock wave             */
        double  gamma,
        double  nu,
        double *r_r2,                   /* Radius Ratio                     */
        double *rho_rho2,               /* Density Ratio                    */
        double *vel_vel2,               /* Velocity Ratio                   */
        double *press_press2,           /* Pressure Ratio                   */
        double *Temp_Temp2 )            /* Temperature Ratio                */
{
    int     iter,
            changed_size_of_delta;
      
    double  gPlus,                      /* gamma + 1                        */ 
            nuPlus,           
            V,
            V_guess,
            V_guess_old,                /* old and new guesses for V        */
            V_guess_new,
            V0,                         /* initial guesses for V            */
            V00,
            delta;   
/*__________________________________
*   Initial guesses for V and 
*   abbrevations for gamma + 1 and nu + 2
*___________________________________*/
    gPlus   = gamma + 1.0;
    nuPlus  = nu + 2.0;
    iter    = 0;        
    V0      = 2.0/( nuPlus * gamma);        /* Lower Limit of V         */
    V00     = 4.0/( nuPlus * gPlus);        /* upper limit of V         */

/*START_DOC*/
/*__________________________________
* default value of V incase r = 0.0;
*___________________________________*/
    V   = 0.0;
/*______________________________________________________________________
*   Use the secant method to solve for V in the r/r2 equation of 11.15
*   See Numerical Methods by Hornbeck, pg 71
*   Do this only if r != 0.0
*_______________________________________________________________________*/
    if( r > SMALL_NUM)
    {
        delta            = V0 - V00;
        V_guess          = V0;
        V_guess_old      = r_r2_function( gamma, nu, r, r2, V00);

        while (fabs(delta) > CONVERGENCE && iter < MAX_ITER )
       {
            changed_size_of_delta   = 0;
            V_guess_new             = r_r2_function( gamma, nu, r, r2, V_guess);
            delta                   = -V_guess_new/( (V_guess_new - V_guess_old)/delta );
            /*__________________________________
            *   Limit the size of change of delta
            *___________________________________*/
            if ( (V_guess + delta) > V00 || (V_guess + delta) < V0 ) 
            {
                changed_size_of_delta= 1;
                delta               = delta/5.0;
                V_guess             = V_guess + delta;
            }
            
            if ( V_guess > V00  )
            {
                V_guess = V00;
                changed_size_of_delta= 1;
            }
            
            if ( V_guess < V0  ) 
            {
                V_guess = V0;
                changed_size_of_delta= 1;
            }            
            
            
            /*__________________________________
            *   If things aren't broken
            *___________________________________*/
            if ( (V_guess + delta) < V00 && (V_guess + delta) > V0 ) 
            {
                V_guess= V_guess + delta;
            }

            
            V_guess_old             = V_guess_new;
            /*
            
            fprintf(stderr,"V_guess = %g delta %g \t Had to limit delta %i\n",V_guess, delta,
            changed_size_of_delta);
            
            */
            iter ++;
            /*__________________________________
            *   Make sure that delta is a number
            *___________________________________*/
            if ( isnan(delta) ) delta = 2.0; 
        }
        /*__________________________________
        *   Bullett proof
        *___________________________________*/
        if (iter == MAX_ITER) 
            Message(1,"File: blastWave","Max. iterations when solving for V", "");

        V                = V_guess;
    }
/*______________________________________________________________________
*   Now compute the density, velocity and pressure ratio
*_______________________________________________________________________*/   
    *r_r2        = r/r2;
    
    *rho_rho2    = rho_rho2_function(gamma,nu,V);
    *press_press2= press_press2_function(gamma, nu, V);
    
    *vel_vel2    = ( ( nuPlus * gPlus)/4.0 ) * V *  *r_r2;
    *Temp_Temp2  = *press_press2/ *rho_rho2;
/*__________________________________
*   Printout some debugging stuff
*___________________________________*/
/*     fprintf(stderr, "V \t \t %g\n", *V);
    fprintf(stderr, "rhohat \t \t %g\n",*rhohat);
    fprintf(stderr, "press_hat\t %g\n", *press_hat); */
      
}


/* 
 ======================================================================*/
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#define CONVERGENCE 1e-6
#define MAX_ITER 50
/*
 Function:  Solve_Blast_Wave_problem--MISC: Computes flow variables for the 
            blast wave problem. 
 Filename:  blastWave.c
  
 Purpose:
            Solves the blast wave problem in 2-D.  The wave can be planar
            cylindrical or spherically symetric.

 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       04/01/00    
       
Implementation Notes:
    The notation used in this code comes directly from the reference
       
Reference:
      Reference: Similarity and Dimensional Methods in Mechanics, Sedov, 1959
                 pg 217 +
     
Steps:
    - Determine shock radius and shock velocity                     ( 11.4-11.6)
    - Compute the vel, Temp, Rho, Press just down stream of shock   (11.11)
    - Using the secant method solve for density, velocity, temperature
      ratios as described by eq. 11.15                    
    - Backout uvel_exact, vvel_exact press_exact, rho_exact 
 ---------------------------------------------------------------------  */
 
 void Solve_Blast_Wave_problem(
        int     xLoLimit,               /* x-array lower limit              */
        int     yLoLimit,               /* y-array lower limit              */
        int     zLoLimit,               /* z-array lower limit              */
        int     xHiLimit,               /* x-array upper limit              */
        int     yHiLimit,               /* y-array upper limit              */
        int     zHiLimit,               /* z-array upper limit              */
        double  delX,                   /* distance/cell, xdir              */
        double  delY,                   /* distance/cell, ydir              */
        double  delZ,                   /* distance/cell, zdir              */
        int     ignition_point_xdir,    /* x location of where E was added  */
        int     ignition_point_ydir,    /* y location of where E was added  */
        double  t,
        double  E,                      /* initial energy released at a point*/
        double  rho1,                   /* initial density                  */
        double  gamma,
        double  R,                      /* ideal gas constant               */
        double  uvel_initial,           /* initial values                   */
        double  vvel_initial,
        double  rho_initial,
        double  press_initial,
        double  Temp_initial,
        double  *r2,                    /* radius of blast wave             */
        double  ***uvel_exact,          /* exact vel. xdir                  */
        double  ***vvel_exact,          /* exact vel. ydir                  */
        double  ***rho_exact,           /* exact density                    */
        double  ***press_exact,         /* exact pressure                   */
        double  ***Temp_exact )          /* exact Temperature                */
        
{
        int     i,j,k, m;
 static int     BW_radius_cell_old,
                BW_radius_cell_new;
        
        double  x, y,
                vel_radial_blastWave,   /* exact radial velocity            */
                angle,                  /* angle used to extract x and y    */
                                        /* components of velocity           */
                gPlus,                  /* gamma + 1.0                      */
                gMinus,                 /* gamma - 1.0                      */
                nuPlus,                 /* nu + 2.0                         */
                r,                      /* current radius from ingition_pt  */
                strong_shock,           /* used to determine if it is a     */
                                        /* strong shock or not              */
                c;                      /* blast wave velocity              */
               
                /* immediately behind the shock         */
                
        double  press_2,                /* pressure down stream of shock   */
                vel_2,                  /* just downstream of the shock    */
                rho_2,                  /* rho                             */
                Temp_2,                 /* Temperature                     */
                rho_rho2,               /* Equation 11.15 ratios            */
                vel_vel2,
                press_press2,
                Temp_Temp2,
                r_r2;
        double  r_cell;                  /* diagonal length in cell         */
        

        int     nu;                      /* 1 = plane wave                  */
                                         /* 2 = cylindrical                 */
                                         /* 3 = spherical                   */
        
/*__________________________________
*   Abbrevations used in the equations
*___________________________________*/
    nu      = 2;
    m       = 1;            /* HARDWIRED */  
    gPlus   = gamma + 1.0;
    gMinus  = gamma - 1.0;
    nuPlus  = (double)nu + 2.0;
    r_cell  = sqrt( pow(delX,2) + pow(delY,2) );
/*__________________________________
* Compute shock radius and velocity
*___________________________________*/
    if ( nu == 3)                       /* spherical symmetry case  eq 11.4 */
    {
        *r2  =   pow( (E/rho1), 0.2) * pow( t, 0.4);
        c   =   (2.0/5.0) * pow( (E/rho1), 0.2) * pow( t, -0.6);
    }
    if ( nu == 2)                       /* cylindrical symmetry case eq 11.5*/
    {
        *r2  =   pow( (E/rho1), 0.25) * pow( t, 0.5);
        c   =   (1.0/2.0) * pow( (E/rho1), 0.25) * pow( t, -0.5);
    }    
    if ( nu == 1)                       /* plane symmetry case eq. 11.6     */
    {
        *r2  =   pow( (E/rho1), (1.0/3.0)) * pow( t, (2.0/3.0));
        c   =   (2.0/3.0) * pow( (E/rho1), (1.0/3.0)) * pow( t, -(1.0/3.0));
    }
    
/*__________________________________
*   Immediate downstream of the shock
*   Find the Temp,Press,Vel and Rho eq. 11.11
*___________________________________*/
    press_2 = 8.0 * E/( pow(nuPlus,2) * gPlus * pow(*r2,nu) );
    vel_2   = (4.0 / ( nuPlus * gPlus) ) *  pow( (E/rho1), 0.5) * (1.0/ pow( *r2, nu/2.0) );
    rho_2   = (gPlus / gMinus) * rho1;
    Temp_2  = press_2/(R * rho_2);    



/*__________________________________
*   Compute the exact solution only if 
*   a strong shock has been detected
*   otherwise set the exact solution
*   equal to the undisturbed state
*___________________________________*/   
    BW_radius_cell_new = *r2/delX;
    fprintf(stderr, "BW_radius Cell - BW_radius_cell_old %i\n", ( BW_radius_cell_new-BW_radius_cell_old) );
    if ( (BW_radius_cell_new - BW_radius_cell_old) == 1.1) BW_radius_cell_old = BW_radius_cell_new;
    
     
    for ( k = zLoLimit; k <= zHiLimit; k++)
    {
        for ( j = yLoLimit; j <= yHiLimit; j++)
        {
            for ( i = xLoLimit; i <= xHiLimit; i++)
            { 
            /*__________________________________
            *   Find the radius from the ignition point
            *___________________________________*/
                x   = (double)(i - ignition_point_xdir) * delX;
                y   = (double)(j - ignition_point_ydir) * delY;
                
                r   = sqrt( pow(x,2) + pow(y,2) ); 
                
                /*__________________________________
                *   outside of the blast wave set 
                *   the exact = *CC
                *___________________________________*/
                if ( ( r > *r2) )
                {
                    uvel_exact[i][j][k]     = uvel_initial;   
                    vvel_exact[i][j][k]     = vvel_initial;   
                    rho_exact[i][j][k]      = rho_initial;     
                    press_exact[i][j][k]    = press_initial;            
                    Temp_exact[i][j][k]     = Temp_initial;    
                }
                /*__________________________________
                *   Inside of the blast wave compute
                *   but not at the ignition point
                *___________________________________*/          
                if ( r < *r2 )
                {   
                    /* if (j == ignition_point_ydir) fprintf(stderr,"[%i][%i][%i]\n",i,j,k); */
                    
                    BlastWave_compute_eq_11_15(
                                        r,          *r2,                  
                                        gamma,      nu,
                                        &r_r2,      &rho_rho2,      &vel_vel2,            
                                        &press_press2,              &Temp_Temp2 );      
                    /*__________________________________
                    *   Now backout the exact values from
                    *   the ratios
                    *___________________________________*/
                    angle                   = atan(y/x);
                    angle                   = atan2(y,x);
                    vel_radial_blastWave    = vel_vel2 * vel_2;
                    
                    uvel_exact[i][j][k]     = (cos(angle))  * vel_radial_blastWave;
                    vvel_exact[i][j][k]     = (sin(angle))  * vel_radial_blastWave;

                    rho_exact[i][j][k]      = rho_rho2      * rho_2;             
                    press_exact[i][j][k]    = press_press2  * press_2;
                    Temp_exact[i][j][k]     = Temp_Temp2    * Temp_2;
 
                    /*__________________________________
                    *   Put an upper limit on the temperature
                    *   so you can plot it, expecially near
                    *    the origin
                    *___________________________________*/
                   /*  if (Temp_exact[i][j][k] > 10000) Temp_exact[i][j][k] = 10000; */
                }
                /*__________________________________
                * At the origin take the average of the
                * nearby cells
                *___________________________________*/
                if ( r < delX)
                {
                    rho_exact[i][j][k]      = (   rho_exact[ignition_point_xdir+1][j][k] 
                                                + rho_exact[ignition_point_xdir-1][j][k])/2.0;   
                                                         
                    uvel_exact[i][j][k]     = (   uvel_exact[ignition_point_xdir+1][j][k] 
                                                + uvel_exact[ignition_point_xdir-1][j][k])/2.0;
                                                
                    vvel_exact[i][j][k]     = (   vvel_exact[ignition_point_xdir+1][j][k] 
                                                + vvel_exact[ignition_point_xdir-1][j][k])/2.0;
                                                
                    press_exact[i][j][k]    = (   press_exact[ignition_point_xdir+1][j][k] 
                                                + press_exact[ignition_point_xdir-1][j][k])/2.0;
                                                
                    Temp_exact[i][j][k]     =(    Temp_exact[ignition_point_xdir+1][j][k] 
                                                + Temp_exact[ignition_point_xdir-1][j][k])/2.0;
                }
            }
        }
    }
    
/*__________________________________
*   Print debugging messages
*___________________________________*/
    fprintf(stderr, "rho_2 = \t %g \t vel_2 = \t %g\n", rho_2, vel_2);
    fprintf(stderr, "press_2 = \t %g \t Temp_2 = \t %g\n", press_2, Temp_2);
    fprintf(stderr, "r2 = \t %g\n",*r2); 

/*__________________________________
*   Quite Full warnings
*___________________________________*/
    delZ = delZ;
}

/*STOP_DOC*/ 
