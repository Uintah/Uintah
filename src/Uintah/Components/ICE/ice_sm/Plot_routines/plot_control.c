 
/* ======================================================================*/
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include "cpgplot.h"
#include "parameters.h"
#include "nrutil+.h"
#include "functionDeclare.h"
/* 
 Function:  plot_control--VISUALZATION: Main controller for the plotting routines.
 Filename:  plot_common.c

 Purpose:  main plot controlling program

 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       05/12/99   Written   
_______________________________________________________________________ */
   void    plot(  
    const float   *data,                /* data vector                  */
    const float   *data2,               /* second set of data used for  */
                                        /* vector plots and the x-axis  */
                                        /* for scatter plots            */ 
    int     max_len, 
    double  delX,                       /* delta x                      */
    int     xLoLimit,                   /* x_axis origin                */
    int     xHiLimit,                   /* x_axis limit                 */
    int     yLoLimit,                   /* y_axis origin                */
    int     yHiLimit,                   /* y_axis limit                 */
    char    x_label[],                  /* label along the x axis       */
    char    y_label[],                  /* label along the y axis       */
    char    graph_label[],              /* label at top of plot         */
    int     plot_type,                  /* switch to determine which type of*/
                                        /* plot to generate                 */
    int     outline_ghostcells,         /* 1= outline the ghostcells    */
    int     max_sub_win,                /* max number of sub windows to */
                                        /* generate                     */
    char    file_basename[],            /* basename of the output file  */
    int     filetype)                   /* type of file to select       */
                                        /* 1 = gif, 2 = ps, 3 = xwd     */
{

    int     i,  
            color,
            symbol,                                                       
            error,          
            error_x, 
            error_data;                 /* error flag for NAN or INF    */

static int
            n_sub_win,                  /* counter for the number of sub*/
            n_sub_win_cursor,           /* windows                      */
            n_sub_win_file;             /* number of subwins for file   */
                 
    float   xLo,          xHi,          /* max and min values           */
            yHi,          yLo,
            data_min,       data_max;
    float   *x_data,
            symbol_size; 
            
    char    stay_or_go;
    
   
/*__________________________________
*   Should I plot during this pass
*___________________________________*/
    stay_or_go = *getenv("PGPLOT_PLOTTING_ON_OFF");
/*     printf ("PGPLOT_PLOTTING: %c \n",stay_or_go); */
    if (stay_or_go == '0') return;
/*__________________________________
*   Bulletproofing
*___________________________________*/
    if(max_len <=0)
    {
        Message(1,"File: Plot_Control.c","Function Plot",
        "The length of the input data array is <=zero, now exitig");
    }     
                    
    /*__________________________________
    *  Find data_min, data_max
    *___________________________________*/                                
    error_data  = plot_scaling( data,       max_len,        &data_min, &data_max);  
    
/*______________________________________________________________________
*   Now start the the plotting 
*_______________________________________________________________________*/    
    if (filetype == 0)  plot_open_window_screen(max_sub_win, &n_sub_win);
    if (filetype > 0)   plot_open_window_file(max_sub_win, &n_sub_win_file, file_basename, filetype);

    /*__________________________________
    *   Generate the color spectrum
    *___________________________________*/       
    plot_color_spectrum(); 
    cpgsclp(0);                                      /* turn off clipping*/
    /*__________________________________
    *  Contour plot
    *___________________________________*/  
    if(plot_type == 1)
    { 
        cpgbbuf();
        xHi       = (float)xHiLimit;
        yHi       = (float)yHiLimit;
        xLo       = (float)xLoLimit;
        yLo       = (float)yLoLimit;

        plot_generate_axis( x_label,        y_label,        graph_label,
                            &xLo,           &xHi,         
                            &yLo,           &yHi, 
                            &error_data);

    #if (contourplot_type  == 1)                              
        plot_contour(       xLo,            xHi,       
                            yLo,            yHi,
                            data_min,       data_max,       data);
    #endif
    #if (contourplot_type  == 2)                                    
        plot_contour_checkerboard(       
                            xLo,            xHi,       
                            yLo,            yHi,
                            data_min,       data_max,       data);
    #endif

        plot_legend(        data_max,       data_min, 
                            xHi,            yHi);
        cpgebuf();

    }
   /*__________________________________
   * Line plot
   *___________________________________*/
    if((plot_type == 2) || (plot_type == 4))
    {

        color           = NUM_COLORS;
        symbol          = -9;
        symbol_size     = 0.75;
        /*__________________________________
        *   Generate x data
        *___________________________________*/
        x_data      = vector_nr(1,  (max_len));
        x_data[0]   = (float)xLoLimit - 1;   
        x_data[1]   = (float)xLoLimit;

        for( i =2; i<=max_len; i++)
        {
            x_data[i] = x_data[i-1] + 1;
        }
        /*__________________________________
        *   search for NAN or INF numbers
        *___________________________________*/
        error_x     = plot_scaling( x_data,     max_len,     &xLo,    &xHi);
        error_data  = plot_scaling(   data,     max_len,     &yLo,    &yHi); 
        error       = IMAX(error_x, error_data);

        plot_generate_axis( 
                            x_label,        y_label,        graph_label,
                            &xLo,           &xHi,         
                            &yLo,           &yHi, 
                            &error_data );
        cpgbox("G\0",0.0, 0, "G\0", 0.0, 0);
        plot_2d_scatter(    x_data,         data,           max_len, 
                            color,          symbol,         symbol_size);
        /*__________________________________
        *   To overlay a second set of data
        *___________________________________*/
        if (plot_type == 4)
        { 
            symbol      = -9;
            color = (int)color/2;                   
            plot_2d_scatter(x_data,         data2,          max_len,     
                            color,          symbol,         symbol_size);
        }

        free_vector_nr(x_data, 1, max_len);

    }
    /*__________________________________
    *   Vector Plot
    *___________________________________*/
    if( plot_type == 3 )
    {
        cpgbbuf();

/*`==========TESTING==========*/ 
        xHi       = (float)xHiLimit + 1;
        yHi       = (float)yHiLimit + 1;

/*==========TESTING==========`*/

        xLo       = (float)xLoLimit;
        yLo       = (float)yLoLimit;

        plot_generate_axis( x_label,        y_label,        graph_label,
                            &xLo,          &xHi,         
                            &yLo,          &yHi, 
                            &error_data);   
        xHi = xHi - 1.0;    /* you need this to acount for cell-centered data */
        yHi = yHi - 1.0;         
        plot_vector_2D(     xLo,            yLo,
                            xHi,            yHi,
                            max_len,
                            data,           data2); 
        cpgsclp(1);                                      /* turn on clipping*/
        cpgebuf();     
     }


    /*__________________________________
    *  Plot_cursor_position
    *  This doesn't work in the pse
    *___________________________________*/
   /*  if (filetype == 0 ) plot_cursor_position(max_sub_win, &n_sub_win_cursor);  */

/*______________________________________________________________________
*   Now write a discription and outline the ghostcells and 
*   if it's time then close the file
*_______________________________________________________________________*/    
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
/*__________________________________
*   Quite fullwarn remarks in a way that
*   is compiler independent
*___________________________________*/
    error = error;
     
}
/*STOP_DOC*/
