
/*  
======================================================================*/
#include <math.h>
#include <stdio.h>
#include "switches.h" 
#include "parameters.h"
#include "macros.h"
#include "functionDeclare.h"
/*
 Function:  generateGrid--MISC: Compute the x,y,z, coordinates of the cell-centeres and face-centeres
 Filename:  grid.c
 Purpose: Generate the x,y,z, coordinates of the cell-centered and face-centered 

 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       02/22/99   
       
----------------------------------------------------------------------- */

void    generateGrid(   
            int         xLoLimit,
            int         yLoLimit,
            int         zLoLimit,
            int         xHiLimit,
            int         yHiLimit,
            int         zHiLimit,

            double      delX,           /* cell width                       (INPUT) */
            double      delY,           /* cell width in y dir.             (INPUT) */
            double      delZ,           /* cell width in z dir.             (INPUT) */
            double      ***x_CC,        /* x-coord. cell-center             (OUTPUT)*/
            double      ***y_CC,        /* y-coord. cell-center             (OUTPUT)*/
            double      ***z_CC,        /* z-coord. cell-center             (OUTPUT)*/
            double      ***Vol_CC,      /* cell volume cell-center          (OUTPUT)*/
                        /* treated as pointers *(*)_FC(i,j,k,face)                  */
            double      *****x_FC,      /* x-coordinate face-center         (OUTPUT)*/  
            double      *****y_FC,      /* y-coordinate face-center         (OUTPUT)*/
            double      *****z_FC  )    /* z-coordinate face-center         (OUTPUT)*/
{
        int         i,j,k,
                    xLo, xHi,           /* array limits including ghost cells */
                    yLo, yHi,
                    zLo, zHi;
/* _______________________________________________________________________
 calculate cell centered distances

                               _______ ________
                              |       |       |
                              | I,J,K |       |
                              |       |       |--- y_CC
                              |_______|_______|
                                   \      /
                                   x_CC

_______________________________________________________________________ */
        xLo = GC_LO(xLoLimit);
        xHi = GC_HI(xHiLimit);
        yLo = GC_LO(yLoLimit);
        yHi = GC_HI(yHiLimit);
        zLo = GC_LO(zLoLimit);
        zHi = GC_HI(zHiLimit);
/*__________________________________
* bullet proofing
*___________________________________*/
        if( xLo < 0 || xLo > X_MAX_LIM)
            Message(1, "grid.c", "generateGrid", "xLo < 0 || xLo > X_MAX_LIM");      
        if( xHi < 0 || xHi > X_MAX_LIM)
            Message(1, "grid.c", "generateGrid", "xHi < 0 || xHi > X_MAX_LIM");
        if( yLo < 0 || yLo > X_MAX_LIM)
            Message(1, "grid.c", "generateGrid", "yLo < 0 || yLo > Y_MAX_LIM");      
        if( yHi < 0 || yHi > Y_MAX_LIM)
            Message(1, "grid.c", "generateGrid", "yHi < 0 || yHi > Y_MAX_LIM");
        if( zLo < 0 || zLo > Z_MAX_LIM)
            Message(1, "grid.c", "generateGrid", "zLo < 0 || zLo > Z_MAX_LIM");      
        if( zHi < 0 || zHi > Z_MAX_LIM)
            Message(1, "grid.c", "generateGrid", "zHi < 0 || zHi > Z_MAX_LIM");                    
        for( k = zLo; k<=zHi; k++)
        {
            for( j = yLo; j <=yHi; j++)
            {
                for( i = xLo; i<= xHi; i++)
                {
                
                    x_CC[i][j][k]   = (double)i*delX - delX/2.0;
                    y_CC[i][j][k]   = (double)j*delY - delY/2.0;
                    z_CC[i][j][k]   = (double)k*delZ - delZ/2.0;
                   
                    Vol_CC[i][j][k] = delX*delY*delZ;

                }
            }
        }

/* _______________________________________________________________________
 calculate face center coordinates
______________________________

                                 ________ 
                                /  1    /|
                               /_______/ |
                              |       | ______(3)
                       (4)____| I,J,K |  |     
                              |       | /      
                              |_______|/
                                  |               (6) = back face
                                 (2)              (5) = front face

FUTURE IMPROVEMENT:                                 
  I'm being stupid here since you only need to calculate the right
  and top and front face coordinates.  The address of 
  x_FC[i][j][k][RIGHT] = X_FC[i][j][k][LEFT]
_______________________________________________________________________ */

        for( k = zLo; k<= zHi; k++)
        {
            for( j = yLo; j <=yHi; j++)
            {
                for( i = xLo; i<=xHi; i++)
                {
                    
/*__________________________________
* x-cell face coordinates
*___________________________________*/
                    *x_FC[i][j][k][TOP]      = x_CC[i][j][k];
                    *x_FC[i][j][k][BOTTOM]   = x_CC[i][j][k];
                    *x_FC[i][j][k][RIGHT]    = x_CC[i][j][k] + delX/2.0;
                    *x_FC[i][j][k][LEFT]     = x_CC[i][j][k] - delX/2.0;
                    *x_FC[i][j][k][FRONT]    = x_CC[i][j][k];
                    *x_FC[i][j][k][BACK]     = x_CC[i][j][k];
/*__________________________________
* y-cell face coordinates
*___________________________________*/                     
                    *y_FC[i][j][k][TOP]      = y_CC[i][j][k] + delY/2.0;
                    *y_FC[i][j][k][BOTTOM]   = y_CC[i][j][k] - delY/2.0; 
                    *y_FC[i][j][k][RIGHT]    = y_CC[i][j][k];
                    *y_FC[i][j][k][LEFT]     = y_CC[i][j][k];
                    *y_FC[i][j][k][FRONT]    = y_CC[i][j][k];
                    *y_FC[i][j][k][BACK]     = y_CC[i][j][k];
/*__________________________________
* z-cell face coordinates
*___________________________________*/
                    *z_FC[i][j][k][TOP]      = z_CC[i][j][k];
                    *z_FC[i][j][k][BOTTOM]   = z_CC[i][j][k];
                    *z_FC[i][j][k][RIGHT]    = z_CC[i][j][k];
                    *z_FC[i][j][k][LEFT]     = z_CC[i][j][k];
                    *z_FC[i][j][k][FRONT]    = z_CC[i][j][k] + delZ/2.0;
                    *z_FC[i][j][k][BACK]     = z_CC[i][j][k] - delZ/2.0;
                }
            }
        } 

/*__________________________________
*
*___________________________________*/
#if switchDebug_grid
        fprintf(stderr,"\n Finished with generageGrid\n");
#endif       
}
/*STOP_DOC*/
