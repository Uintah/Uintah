/*______________________________________________________________________
*   This chunk of code is used to test the advection of a square
*    wave
*_______________________________________________________________________*/ 
#if 1
    for ( k = GC_LO(zLoLimit); k <= GC_HI(zHiLimit); k++)
    {
        for ( j = GC_LO(yLoLimit); j <= GC_HI(yHiLimit); j++)
        {
            for ( i = GC_LO(xLoLimit); i <= GC_HI(xHiLimit); i++)
            { 
                   Temp_CC[m][i][j][k]   = 1.0;                           
         
               if(     i > ( (int)(xHiLimit - xLoLimit)/2 - 3 ) 
                    && i < ( (int)(xHiLimit - xLoLimit)/2 + 3 ) 
                    && j > ( (int)(yHiLimit - yLoLimit)/2 - 3 ) 
                    && j < ( (int)(yHiLimit - yLoLimit)/2 + 3 )  )
               {
                   Temp_CC[m][i][j][k]   = 2.0;
                }
                
/*                if( i > xLoLimit + 5 && i < xLoLimit + 15 &&j <yHiLimit - 5 && j > yHiLimit -15)
               {
                   rho_CC[m][i][j][k]   = 2.0;
                }
                
                if( i < xHiLimit - 5 && i > xHiLimit - 15 &&j <yHiLimit - 5 && j > yHiLimit -15)
               {
                   rho_CC[m][i][j][k]   = 2.0;
                }
                if( i < xHiLimit - 5 && i > xHiLimit - 15 &&j >yLoLimit + 5 && j < yLoLimit + 15)
               {
                   rho_CC[m][i][j][k]   = 2.0;
                }

               

                x_FC[i][j][k][2]= x_CC[i][j][k] + 0.5;
                x_FC[i][j][k][1]= x_CC[i][j][k] - 0.5;
     */
                
            }
        }
    }
    

    
#endif
