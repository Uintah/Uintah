/*______________________________________________________________________
*   This section of code sets up the different tests for a Riemann Problem
*   See pg 204 in "Numerical Computation
*   of Internal and external Flows Vol. 2"
*_______________________________________________________________________*/ 
    for ( k = GC_LO(zLoLimit); k <= GC_HI(zHiLimit); k++)
    {
        for ( j = GC_LO(yLoLimit); j <= GC_HI(yHiLimit); j++)
        {
            for ( i = GC_LO(xLoLimit); i <= GC_HI(xHiLimit); i++)
            { 

#if (testproblem == 1)            
            /*__________________________________
            * Initial shock discontinutiy  
            *___________________________________*/
                /*__________________________________
                * LEFT chamber
                *___________________________________*/
              /*   uvel_CC[m][i][j][k] = u4; */
                uvel_CC[m][i][j][k]     = u4;
                rho_CC[m][i][j][k]      = rho4;
                press_CC[m][i][j][k]    = p4;
                Temp_CC[m][i][j][k]     = p4/(R[m] * rho4);
                speedSound[m][i][j][k]  = sqrt(1.4*press_CC[m][i][j][k]/rho_CC[m][i][j][k]);
                /*__________________________________
                * RIGHT chamber
                *___________________________________*/   
                if( i > ( (int)(qHiLimit - qLoLimit+1)/2 ) )
               {
                    /* uvel_CC[m][i][j][k] = u1; */
                    uvel_CC[m][i][j][k]     = u1;
                    rho_CC[m][i][j][k]      = rho1;
                    press_CC[m][i][j][k]    = p1;
                    Temp_CC[m][i][j][k]     = p1/(R[m] * rho1);
                    speedSound[m][i][j][k] = sqrt(1.4*press_CC[m][i][j][k]/rho_CC[m][i][j][k]);
                }   
#endif   
                   
            }
        }
    }
    

    
