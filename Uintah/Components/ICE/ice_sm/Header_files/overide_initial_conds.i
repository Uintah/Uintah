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
        /*        
	 uvel_CC[m][i][j][k] = (double)i;
	 vvel_CC[m][i][j][k] = (double)j;
         vvel_CC[m][i][j][k] = 0.;	
	*/
                
            }
        }
    }
    

    
#endif
