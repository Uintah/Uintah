/*______________________________________________________________________
*   This chunk of code is used to test the advection of a square
*    wave
*_______________________________________________________________________*/ 
    for ( k = GC_LO(zLoLimit); k <= GC_HI(zHiLimit); k++)
    {
        for ( j = GC_LO(yLoLimit); j <= GC_HI(yHiLimit); j++)
        {
            for ( i = GC_LO(xLoLimit); i <= GC_HI(xHiLimit); i++)
            { 
                   Temp_CC[i][j][k][m]   = 1.0;                           

               if( i > xLoLimit + 1 && i < xLoLimit + 10 &&j >yLoLimit + 1 && j < yLoLimit + 10)
               {
                   Temp_CC[i][j][k][m]   = 2.0;
                }
                
/*                if( i > xLoLimit + 5 && i < xLoLimit + 15 &&j <yHiLimit - 5 && j > yHiLimit -15)
               {
                   rho_CC[i][j][k][m]   = 2.0;
                }
                
                if( i < xHiLimit - 5 && i > xHiLimit - 15 &&j <yHiLimit - 5 && j > yHiLimit -15)
               {
                   rho_CC[i][j][k][m]   = 2.0;
                }
                if( i < xHiLimit - 5 && i > xHiLimit - 15 &&j >yLoLimit + 5 && j < yLoLimit + 15)
               {
                   rho_CC[i][j][k][m]   = 2.0;
                }

               

                x_FC[i][j][k][2]= x_CC[i][j][k] + 0.5;
                x_FC[i][j][k][1]= x_CC[i][j][k] - 0.5;
     */
                
            }
        }
    }
