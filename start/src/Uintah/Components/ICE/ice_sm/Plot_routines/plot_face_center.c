 
/* ======================================================================*/
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include "cpgplot.h"
#include "parameters.h"
#include "nrutil+.h"
#include "functionDeclare.h"
#include "macros.h"
/* 
 Function:  plot_face_centered_data--VISUALIZATION: generates the face-centered contour plots.
 Filename:  plot_face_center.c

 Purpose:  This routine draws a cell grid pattern with the color of the 
 line on a particular face corresponding to a particular value
 
 This

 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       10/06/99   Written 
         

_______________________________________________________________________ */

   void    plot_face_centered_data(  
    int     xLoLimit,                   /* x_axis origin                */
    int     xHiLimit,                   /* x_axis limit                 */
    int     yLoLimit,                   /* y_axis origin                */
    int     yHiLimit,                   /* y_axis limit                 */
    int     zLoLimit,
    int     zHiLimit,                                                   
    double  delX,                       /* delta X                      */
    double  delY,                       /* delta Y                      */
    double  ******data,                 /* *data(x,y,z,face,material)   */
    char    x_label[],                  /* label along the x axis       */
    char    y_label[],                  /* label along the y axis       */
    char    graph_label[],              /* label at top of plot         */
    int     outline_ghostcells,          /* 1= outline the ghostcells    */            
    int     max_sub_win,                /* max number of sub windows to */
                                        /* generate                     */
    char    file_basename[],            /* basename of the output file  */
    int     filetype,                   /* type of file to select       */
                                        /* 1 = gif, 2 = ps, 3 = xwd     */
    int     m)
{
    int     i,j,k, 
            color,
            error_data;                 /* error flag for NAN or INF    */
static int
            n_sub_win;                  /* counter for the number of sub*/
                                        /* windows                      */
    float   xLo,        xHi,        /* max and min values           */
            yHi,        yLo,
            data_max,   data_min,
            x,          y,
            offset; 
    char    stay_or_go;

/*__________________________________
*   Initialize Variables
*___________________________________*/
    offset  = 0.25;
/*__________________________________
*   Should I plot during this pass
*___________________________________*/
    stay_or_go = *getenv("PGPLOT_PLOTTING_ON_OFF");
    if (stay_or_go == '0') return;


/*__________________________________
*   FRAGMENT FOR TESTING
*___________________________________*/
/*     for ( k = zLoLimit; k <= zHiLimit; k++)
    {
        for ( j = yLoLimit; j <= yHiLimit; j++)
        {
            for ( i = xLoLimit; i <= xHiLimit; i++)
            {
                
                *data[i][j][k][LEFT][m]     = (float)xHiLimit - i;
                *data[i][j][k][RIGHT][m]    = (float)xHiLimit - i - 1;
                *data[i][j][k][BOTTOM][m]   = (float)yHiLimit - j;
                *data[i][j][k][TOP][m]      = (float)yHiLimit - j - 1;
                  
            }
        }
    } */

/*______________________________________________________________________
*   When dumping to the screen
*_______________________________________________________________________*/    
    if (filetype == 0)
    {
    
        
        plot_open_window_screen(max_sub_win, &n_sub_win);
        
        /*__________________________________
        *   Generate the color spectrum
        *___________________________________*/       
        plot_color_spectrum(); 

        /*__________________________________
        * Begin buffering the output
        *___________________________________*/
        cpgbbuf();
        
        
        xHi = (float)xHiLimit+1;
        yHi = (float)yHiLimit+1;
        xLo = (float)xLoLimit;
        yLo = (float)yLoLimit;

        plot_generate_axis( x_label,        y_label,        graph_label,
                            &xLo,           &xHi,         
                            &yLo,           &yHi, 
                            &error_data);
                            
        plot_scaling_FC( 
                            xLoLimit,       yLoLimit,       zLoLimit,                  
                            xHiLimit,       yHiLimit,       zHiLimit,                  
                            data,           m,                         
                            &data_min,      &data_max );
/*______________________________________________________________________
*   Draw the face centered values in all of the cells except the 
*   top and right face of the domain
*_______________________________________________________________________*/
        for ( k = zLoLimit; k <= zHiLimit; k++)
        {
            for ( j = yLoLimit; j <= yHiLimit; j++)
            { 
                for ( i = xLoLimit; i <= xHiLimit; i++)
                {  
                    x       = (float) i;    
                    y       = (float) j;
                    cpgmove(x,y);
                    /*__________________________________
                    *   Draw horizontal line
                    *___________________________________*/
                    x       = (float) i+1;
                    y       = (float) j;
                    color   = (int)(NUM_COLORS)*(*data[i][j][k][BOTTOM][m] - data_min)/
                                (data_max - data_min + SMALL_NUM);
                    cpgsci(color);                          /* Define the color         */
                    cpgdraw(x,y);                           /* Draw a horiziontal line  */
                    x       = (float) i;    
                    y       = (float) j;
                    cpgmove(x,y);                           /* Move back                */
                    
                    /*__________________________________
                    *   Draw vertical line
                    *___________________________________*/
                    x       = (float) i;    
                    y       = (float) j+1;
                    color   = (int)(NUM_COLORS) *(*data[i][j][k][LEFT][m] - data_min)/
                                (data_max - data_min + SMALL_NUM);
                    cpgsci(color);                          /* Define the color         */
                    cpgdraw(x,y);                            /* Draw a vertical line     */
                           /* move back down to i,j    */  
                }
            }
        }


/*______________________________________________________________________
*   Draw the face centered values on the top and right face of the domain
*_______________________________________________________________________*/
        
        for ( k = zLoLimit; k <= zHiLimit; k++)
        {
            for ( j = yHiLimit; j <= yHiLimit; j++)
            { 
                for ( i = xLoLimit; i <= xHiLimit; i++)
                {  
                    x       = (float) i;    
                    y       = (float) j + 1;
                    cpgmove(x,y); 
                    /*__________________________________
                    *   Top Face
                    *___________________________________*/
                    x       = (float) i+1;
                    y       = (float) j+1;
                    color   = (int)(NUM_COLORS)*(*data[i][j][k][TOP][m] - data_min)/
                                (data_max - data_min + SMALL_NUM);
                    cpgsci(color);                          /* Define the color         */
                    cpgdraw(x,y);                           /* Draw a horiziontal line  */
                }
            }
        }
        for ( k = zLoLimit; k <= zHiLimit; k++)
        {
            for ( j = yLoLimit; j <= yHiLimit; j++)
            { 
                for ( i = xHiLimit; i <= xHiLimit; i++)
                { 
                    x       = (float) i + 1;    
                    y       = (float) j;
                    cpgmove(x,y);
                    /*__________________________________
                    *   Right Face
                    *___________________________________*/
                    x       = (float) i + 1;    
                    y       = (float) j+1;
                    color   = (int)(NUM_COLORS) *(*data[i][j][k][RIGHT][m] - data_min)/
                                (data_max - data_min + SMALL_NUM);
                    cpgsci(color);                          /* Define the color         */
                    cpgdraw(x,y);                           /* Draw a vertical line     */

                }
            }
        }

        /*__________________________________
        *   Generate a legend
        *___________________________________*/
        plot_legend(        data_max,       data_min, 
                            xHi,          yHi);
        /*__________________________________
        *   Draw lines around the ghost cells
        *___________________________________*/
        cpgsci(1);
        if(outline_ghostcells == 1)
        {       
            cpgmove(xLoLimit+N_GHOSTCELLS   + offset,yLoLimit+N_GHOSTCELLS      + offset);
            cpgdraw(xHiLimit+1-N_GHOSTCELLS - offset,yLoLimit+N_GHOSTCELLS      + offset);
            cpgdraw(xHiLimit+1-N_GHOSTCELLS - offset,yHiLimit+1-N_GHOSTCELLS    - offset);
            cpgdraw(xLoLimit+N_GHOSTCELLS   + offset,yHiLimit+1-N_GHOSTCELLS    - offset);
            cpgdraw(xLoLimit+N_GHOSTCELLS   + offset,yLoLimit+N_GHOSTCELLS      + offset);
        }
        
        /*__________________________________
        * End buffering the output
        *___________________________________*/
        cpgebuf();
        /*__________________________________
        *Close the windows
        *___________________________________*/
        
        if(n_sub_win == max_sub_win)
        {
            cpgclos();
            n_sub_win = 0;
        }
        

                                                               
    }
/*__________________________________
*   Quite fullwarn remarks in a way that
*   is compiler independent
*___________________________________*/
    QUITE_FULLWARN(delX);                       QUITE_FULLWARN(delY);
    QUITE_FULLWARN(file_basename);
}
/*STOP_DOC*/

