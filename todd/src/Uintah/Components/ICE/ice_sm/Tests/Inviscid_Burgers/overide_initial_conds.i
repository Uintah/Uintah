/*______________________________________________________________________
*   This section of code sets up the different tests for a inviscid 
*   burgers equation.  See pgs 199 to 204 in "Numerical Computation
*   of Internal and external Flows Vol. 2"
*_______________________________________________________________________*/ 
    x0 = delX * (double)(xHiLimit - xLoLimit)/2;

#if (testproblem == 2)
    x0 = x0/2.0;
#endif    
    
     
    for ( k = GC_LO(zLoLimit); k <= GC_HI(zHiLimit); k++)
    {
        for ( j = GC_LO(yLoLimit); j <= GC_HI(yHiLimit); j++)
        {
            for ( i = (xLoLimit); i <= (xHiLimit); i++)
            { 
                x = (double)(i - xLoLimit) * delX;
#if (testproblem == 1)            
            /*__________________________________
            * Initial shock discontinutiy  
            *___________________________________*/
                uL  = 100.0;
                uR  = 0.0;
                /* CFL = 0.9; */
                /*__________________________________
                * LEFT chamber
                *___________________________________*/
                uvel_CC[m][i][j][k] = uL;
                rho_CC[m][i][j][k]  = 1.0;
                press_CC[m][i][j][k]= 1.0;
                /*__________________________________
                * RIGHT chamber
                *___________________________________*/   
                if( x > x0 )
               {
                    uvel_CC[m][i][j][k] = uR;
                    rho_CC[m][i][j][k]  = 1.0;
                    press_CC[m][i][j][k]= 1.0;
                }   
#endif   
#if (testproblem == 2)            
            /*__________________________________
            * Sinusoidal wave profile 
            *___________________________________*/
               u0   = 0.0;
              /*  CFL  = 0.9; */
               L    = 0;                   /* offset from left wall*/
               A    = u0 + 50;
               uvel_CC[m][i][j][k]   = u0;  
               if( (x < x0 + L) && (x > L )  )
               {
                    uvel_CC[m][i][j][k] = u0 + A * sin(M_PI*(x - L)/x0); 
                }
#endif 

#if (testproblem == 3)            
            /*__________________________________
            * Initial Linear Distribution
            * LEFT CHAMBER
            * u1 = 2.0 and u2 = 1.0
            *___________________________________*/
            u1  = 200.0;
            u2  = 100.0;
            /* CFL = 0.9; */
            
            L  = 5.0 * delX;
            x1 = x0 - L;
            x2 = x0;
            /*__________________________________
            * Linear Distribution
            *___________________________________*/
                 rho_CC[m][i][j][k]  = 1.0;
                 press_CC[m][i][j][k]= 1.0;
                 temp1               = (x - x1)*(1 / L);
                 temp2               = u1 + (u2 - u1 )*temp1;
                 uvel_CC[m][i][j][k] = temp2; 
            /*__________________________________
            *   Left Chamber
            *___________________________________*/            
                if( x <= x1 )
               {
                    uvel_CC[m][i][j][k] = u1;
                    rho_CC[m][i][j][k]  = 1.0;
                    press_CC[m][i][j][k]= 1.0;
                } 
 
            /*__________________________________
            * Right chamber
            *___________________________________*/
                if( x >= x2 )
               {
                    uvel_CC[m][i][j][k] = u2;
                    rho_CC[m][i][j][k]  = 1.0;
                    press_CC[m][i][j][k]= 1.0;
                } 
#endif  

#if (testproblem == 4)                     
            /*__________________________________
            * Expansion Wave
            *___________________________________*/
                u1  = 100.0;
                u2  = 200.0;
                /* CFL = 0.9; */
                /*__________________________________
                * LEFT chamber
                *___________________________________*/
                uvel_CC[m][i][j][k] = u1;
                rho_CC[m][i][j][k]  = 1.0;
                press_CC[m][i][j][k]= 1.0;
                /*__________________________________
                * RIGHT chamber
                *___________________________________*/   
                if( x > x0 )
               {
                    uvel_CC[m][i][j][k] = u2;
                    rho_CC[m][i][j][k]  = 1.0;
                    press_CC[m][i][j][k]= 1.0;
                }     
#endif  

#if (testproblem == 5)                     
            /*__________________________________
            * Triangle
            *___________________________________*/
                u1  = 100.0;
                L   = 10 * delX;
                uvel_CC[m][i][j][k] = 0.0;
                /*__________________________________
                * LEFT of center
                *___________________________________*/
                if( (x > x0 - L) && (x <=x0))
               {
                    uvel_CC[m][i][j][k] = u1 * (x - (x0 - L))/L;
                    rho_CC[m][i][j][k]  = 1.0;
                    press_CC[m][i][j][k]= 1.0;
                }
                /*__________________________________
                * RIGHT of center
                *___________________________________*/   
                if( (x < x0 + L) && (x >x0) ) 
               {
                    uvel_CC[m][i][j][k] = u1 * ( (x0 + L) - x )/L;
                    rho_CC[m][i][j][k]  = 1.0;
                    press_CC[m][i][j][k]= 1.0;
                }     
#endif 
                   
            }
        }
    }
    

    
