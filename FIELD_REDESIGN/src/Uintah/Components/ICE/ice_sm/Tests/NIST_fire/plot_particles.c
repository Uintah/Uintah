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
 Function:  plot_particles--VISUALIZATION: generates the colored particles.
 Filename:  plot_particles.c

 Purpose:  This routine draws a colored circle with the filled in color 
 
 This

 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       05/18/00   Written 
         

_______________________________________________________________________ */

   void    plot_particles(  
    int     xLoLimit,                   /* x_axis origin                */
    int     xHiLimit,                   /* x_axis limit                 */
    int     yLoLimit,                   /* y_axis origin                */
    int     yHiLimit,                   /* y_axis limit                 */                                                   
    double  delX,                       /* delta X                      */
    double  delY,                       /* delta Y                      */
    double  *x_pos,                     /* x position of particle       */
    double  *y_pos,                     /* y position of particle       */
    double  *data,                      /* particle data                */
    char    x_label[],                  /* label along the x axis       */
    char    y_label[],                  /* label along the y axis       */
    char    graph_label[],              /* label at top of plot         */
    int     outline_ghostcells,         /* 1= outline the ghostcells    */            
    int     max_sub_win,                 /* max number of sub windows to */
                                        /* generate                     */
    int     nParticles,                 /* number of particles of plot  */
    char    file_basename[],            /* basename of the output file  */
    int     filetype)                   /* type of file to select       */
                                        /* 1 = gif, 2 = ps, 3 = xwd     */
{
    int     n,
            color,                      /* color of the particle        */
            shape,                      /* shape of the particle        */
            error_data;                 /* error flag for NAN or INF    */
static int
            n_sub_win,                  /* counter for the number of sub*/
                                        /* windows                      */
            n_sub_win_cursor,           /*  related to the cursor counter*/
            n_sub_win_file;             /* number of subwins for file   */                              
                                        
    float   xLo,        xHi,            /* max and min values           */
            yHi,        yLo,
            *data_float,
            data_max,   data_min,       /* max and min values of data   */
            x_pos_temp, y_pos_temp,     /* temp. vars for x & y position*/
            offset; 
    char    stay_or_go;
/*______________________________________________________________________
*    P  R  E  L  I  M  I  N  A  R  Y  S
*_______________________________________________________________________*/
    data_float = vector_nr(1, nParticles);
    offset  = 0.25;
    shape   = -9;
/*__________________________________
*   Should I plot during this pass
*___________________________________*/
    stay_or_go = *getenv("PGPLOT_PLOTTING_ON_OFF");
    if (stay_or_go == '0') return;
   

/*______________________________________________________________________
*   - Open a file or window
*  - Generate the color spectrum
*_______________________________________________________________________*/    
    if (filetype == 0)  plot_open_window_screen(max_sub_win, &n_sub_win);
    if (filetype > 0)   plot_open_window_file(max_sub_win, &n_sub_win_file, file_basename, filetype);   
    plot_color_spectrum(); 

    /*__________________________________
    *   - Begin buffering the output
    *   - Generate an axis
    *   - outline the ghostcells
    *___________________________________*/
    cpgbbuf();

    xHi = (float)xHiLimit+1;
    yHi = (float)yHiLimit+1;
    xLo = (float)xLoLimit-1;
    yLo = (float)yLoLimit-1;

    plot_generate_axis( x_label,        y_label,        graph_label,
                        &xLo,           &xHi,         
                        &yLo,           &yHi, 
                        &error_data);

    cpgsci(1);
    if(outline_ghostcells == 1)
    {       
        cpgmove(xLoLimit                - offset,yLoLimit                   - offset);
        cpgdraw(xHiLimit+1-N_GHOSTCELLS + offset,yLoLimit                   - offset);
        cpgdraw(xHiLimit+1-N_GHOSTCELLS + offset,yHiLimit+1-N_GHOSTCELLS    + offset);
        cpgdraw(xLoLimit                - offset,yHiLimit+1-N_GHOSTCELLS    + offset);
        cpgdraw(xLoLimit                - offset,yLoLimit                   - offset);
    }
/*______________________________________________________________________
*    P  A  R  T  I  C  L  E     P  L  O  T  T  I  N  G 
*   - recast data into a float array
*   - generate a legend
*   - find the max or min of the data
*   - plot the data
*_______________________________________________________________________*/ 
    for(n = 1; n<= nParticles; n++)
        data_float[n] = (float)data[n];

    plot_scaling(   data_float,     nParticles,                        
                    &data_min,      &data_max      );

#if 0                  
    for( n = 1; n <= nParticles; n++)
    {
        data_max = FMAX(data_max, data[n]);
        data_min = FMIN(data_min, data[n]);
    } 
#endif 
    
    plot_legend(        data_max,       data_min, 
                        xHi,            yHi); 
                        
                            
    for( n = 1; n <= nParticles; n++)
    {                 
        color = 2 + ( NUM_COLORS - 2) * (data[n] - data_min)/(data_max - data_min + SMALL_NUM);
        cpgsci(color);
        x_pos_temp = (float)x_pos[n]/delX;
        y_pos_temp = (float)y_pos[n]/delY;
        cpgpt1(x_pos_temp,y_pos_temp, shape);
    }
/*______________________________________________________________________
*    C  L  E  A  N     U  P     A  N  D     E  X  I  T 
*   - end buffering
*   - if you want to examine the cursor position
*   - Header description
*   - Outline ghostcells
*   - close all the sub windows
*   - free locally defined arrays
*_______________________________________________________________________*/
    cpgebuf();

   /*__________________________________
    *  Plot_cursor_position
    *___________________________________*/
    if (filetype == 0 ) plot_cursor_position(max_sub_win, &n_sub_win_cursor); 

    /*__________________________________
    *       MISC
    * Graph description
    *___________________________________*/
    if(n_sub_win == 1 || n_sub_win_file == 1 )
    {
        cpgsclp(0);                                      /* turn off clipping*/
        cpgsci(1);
        cpgsch(1.0);
        cpgmtxt("T\0", 3.0, 0.0, 0.0, GRAPHDESC);
        cpgmtxt("T\0", 2.0, 0.0, 0.0, GRAPHDESC2);
        cpgmtxt("B\0", 3.5, 0.5, 0.5, GRAPHDESC3);
        cpgmtxt("B\0", 4.5, 0.5, 0.5, GRAPHDESC4);
        cpgscf(1);
        cpgsclp(1);                                      /* turn on clipping*/
    }
    /*__________________________________
    *   Outline Ghostcells
    *___________________________________*/
    if(outline_ghostcells == 1)
    {        
        cpgmove(xLo+N_GHOSTCELLS,yLo+N_GHOSTCELLS);
        cpgdraw(xHi-N_GHOSTCELLS,yLo+N_GHOSTCELLS);
        cpgdraw(xHi-N_GHOSTCELLS,yHi-N_GHOSTCELLS);
        cpgdraw(xLo+N_GHOSTCELLS,yHi-N_GHOSTCELLS);
        cpgdraw(xLo+N_GHOSTCELLS,yLo+N_GHOSTCELLS);
    }
          
    /*__________________________________
    *Close the windows
    *___________________________________*/

    if(n_sub_win == max_sub_win && filetype == 0)
    {
        cpgclos();
        n_sub_win           = 0;
        n_sub_win_cursor    = 0;
    }
    if(n_sub_win_file == max_sub_win && filetype > 0)
    {
        cpgclos();
        n_sub_win_file      = 0;
    }    
    
    
    
    free_vector_nr(data_float, 1, nParticles);   
/*__________________________________
*   Quite fullwarn remarks in a way that
*   is compiler independent
*___________________________________*/
    QUITE_FULLWARN(delX);                       QUITE_FULLWARN(delY);
}
/*STOP_DOC*/
