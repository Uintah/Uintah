/* ======================================================================*/
#include <stdio.h>
#include "cpgplot.h"
#include "parameters.h"
#include "nrutil+.h"
#include "functionDeclare.h"

#define     SELECT_WINDOW(a, b, c) (a ==5)? b : c - b + 1 
/* 
 Function: plot_open_window_screen
 File Name: plot_common.c

 Purpose: Window manager, opens the number of window and makes sure
 that the output goes to the right window.
 Steps:
 ------
    1) If it is the first time through the main loop then count the max. 
    number of main windows that are opened (max_main_win).
    2) Open an window and select select it
    3) If there are subwindows for a main window then don't open a new 
    window.
    4) For subsequent passes through the routine select the correct window
    using win_index so that the figures don't bounce around.
    
Note that the windows will be closed outside of this routine

 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       05/12/99   Written   

 IN args/commons        Units      Description
 ---------------        -----      -----------

_______________________________________________________________________ */

    void plot_open_window_screen(
    int        max_sub_win,             /* max number of sub_windows    */
    int        *n_sub_win,              /* counter for the number of sub*/
                                        /* windows                      */
    int        *plot_ID       )         /* Used to identify the main win*/

{
    int     i,
            open_new_windows;           /* flag for opening new windows */
            
   static  int 
            first_time_through ,        /* if this is the first time    */ 
            n_windows,                  /* counter for the number of open*/
                                        /* windows                      */
            max_main_win,               /* max number of windows opened */
            win_indx,                   /* window index                 */
            subcycle,
            ID,                         /* ID of device                 */
            plot_ID_old;
            
/*__________________________________
*   TESTING
*___________________________________*/
    char    answer[10];
    int     answer_len = sizeof(answer);
    float  completed_one_loop;
        
    cpgqinf("STATE", answer, &answer_len);
    printf("cpgqinf state; %s\n", answer);
    
    cpgqch(&completed_one_loop);
    printf ("completed_one_loop: %f \n",completed_one_loop);
    
/*______________________________________________________________________
*
*_______________________________________________________________________*/
    open_new_windows = 0;
    
    
    
/*______________________________________________________________________
*   Open the main window and the sub windows
*______________________________________________________________________
*  determine if this is the first time through
*___________________________________*/

    if ( max_main_win == 0) first_time_through ++;
    
    /*__________________________________
    *  After we've been throught the entire
    * algorithm once then set some flags
    * to indicate that we've already opened
    * all the windows that we need to.
    *___________________________________*/
    if ( *plot_ID == 1 && max_sub_win == 0) 
        {
            first_time_through ++;
            open_new_windows = 1;
/*             if(subcycle == 4)
                subcycle = 0;
            subcycle ++; */
        }
    if ( *plot_ID == 1 && max_sub_win > 0 && *n_sub_win == 0) 
        {
            first_time_through ++;
            open_new_windows = 1;
/*             if(subcycle == 4)
                subcycle = 0;
            subcycle ++; */
        }
/*__________________________________
*   First pass through 
*___________________________________*/    
    if ( first_time_through == 1 )
    {
        
        /*__________________________________
        * No subwindows then just open a new window
        *___________________________________*/
        if ( max_sub_win == 0 ) 
        {
            max_main_win ++;
            ID = cpgopen("/XSERVE\0");
            *plot_ID = max_main_win;
            cpgeras();
        }    
              
        /*__________________________________
        * If there are sub windows then set
        *   the sub window layout and open a new
        * window.
        *___________________________________*/
         if( max_sub_win > 0)
         {
            *n_sub_win = *n_sub_win + 1;
            if (*n_sub_win == 1 )
            {
                ID = cpgopen("/XSERVE\0");
                max_main_win ++;
                if (max_sub_win ==2) cpgsubp(1,2);
                if (max_sub_win >=3) cpgsubp(2,2);
                *plot_ID = max_main_win;
                cpgeras();
            }
        }
        /*__________________________________
        * select the window to plot in
        *___________________________________*/    
        cpgslct(ID);  
              
    }
   
/*__________________________________
*   All subsequent passes through
*   this routine
*___________________________________*/       
    if ( first_time_through >1 )
    {
        /*__________________________________
        *   reset some counters
        *   and open a new set of windows
        *___________________________________*/
        if( open_new_windows == 1 )
        {           
            n_windows   =   0;
            *n_sub_win  =   0;
            for (i = 1; i<= max_main_win; i++) cpgopen("/XSERVE\0");       
        }
         
        /*__________________________________
        *   If there are no subwindows then 
        *   we need to find the right window
        *   to dump the graphics in so that
        *   the plots stay in one window
        *___________________________________*/
        if ( max_sub_win == 0 ) 
        {
            n_windows   ++;   
            win_indx = max_main_win - n_windows + 1;    
            cpgslct(win_indx);
        } 
                     
        /*__________________________________
        *   If there are sub windows then set
        *   the sub window layout adn 
        *   find the right window to dump the 
        *   graphics in .
        *___________________________________*/
         if( max_sub_win > 0)
         {
            *n_sub_win = *n_sub_win + 1;

            if (*n_sub_win == 1 )
            {
                n_windows   ++; 
                win_indx = max_main_win - n_windows + 1; 
                cpgslct(win_indx); 
                    
                if (max_sub_win ==2) cpgsubp(1,2);
                if (max_sub_win >=3) cpgsubp(2,2);
            }
        }
    }
    plot_ID_old = *plot_ID;
/*     fprintf(stderr, "win_indx %i, subcycle %i, first time through %i \n", 
            win_indx, subcycle, first_time_through); */      
}


/* ======================================================================*/
#include <stdio.h>
#include <string.h>
#include "cpgplot.h"
#include "parameters.h"
#include "nrutil+.h"
#include "functionDeclare.h"


/* 
 Function: plot_open_window_file
 File Name: plot_common.c

 Purpose: Window manager, opens a file
 Steps:
 ------
    1) If it is the first time through the main loop then count the max. 
    number of main windows that are opened (max_main_win).
    2) Open an window and select
    3) If there are subwindows for a main window then don't open a new 
    window.

 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       05/12/99   Written   

 IN args/commons        Units      Description
 ---------------        -----      -----------

_______________________________________________________________________ */

    void plot_open_window_file(
    
    int     max_sub_win,                /* max number of sub_windows    */

    int     *n_sub_win,                 /* counter for the number of sub*/
                                        /* windows                      */              
    char    basename[],                 /* basename of the file         */
    int     filetype            )       /* type of file to select       */
                                        /* 1 = gif, 2 = ps, 3 = xwd     */

{
   static  int 
            n_open_files,               /* number of open windows       */
            first_time_through,         /* if this is the first time    */ 
                    
            n_windows,                  /* counter for the number of open*/
                                        /* windows                      */
            max_main_file,              /* max number of windows opened */
            counter,                    /* file name counter            */
            ID;                         /* device identifier            */
            
    char    filename[50],
            indx[3],                    /* indice of the file name      */
            type[10];                   /* typ of plot                  */
/*______________________________________________________________________
*   
*_______________________________________________________________________*/
    if (filetype == 1 )
        strcpy(type,".gif/GIF\0");
    if (filetype == 2 )
        strcpy(type,".ps/CPS\0");
    if (filetype == 3 )
        strcpy(type,".xwd/WD\0");
/*______________________________________________________________________
*   Open the main window and the sub windows
*_______________________________________________________________________
*   determine if this is the first time
*   through since cpgclos was called
*___________________________________*/ 
    if (first_time_through == 0 )cpgqid(&n_windows);
    
    cpgqid(&n_open_files);
    if (n_open_files == n_windows ) first_time_through ++;
/*__________________________________
*   First pass through 
*___________________________________*/    
    if ( first_time_through == 1 )
    {     
        if ( max_sub_win == 0 ) 
        {
            max_main_file ++;

            /*__________________________________
            * Determine the filename and open the 
            * file
            *___________________________________*/
            counter ++;
            sprintf(indx, "%d",counter);
            strcpy(filename,filepath);
            strcat(filename,indx);
            strcat(filename,basename);
            strcat(filename,type);
            ID = cpgopen(filename);
        }    
              
        /*__________________________________
        *   If there are sub windows then set
        *   the sub window layout and open the file
        *___________________________________*/
         if( max_sub_win > 0)
         {
            *n_sub_win = *n_sub_win + 1;
            if (*n_sub_win == 1 )
            {
               max_main_file ++;
                /*__________________________________
                * Determine the filename and open the 
                * file
                *___________________________________*/
                counter ++;
                sprintf(indx, "%d",counter);
                strcpy(filename,filepath);
                strcat(filename,indx);
                strcat(filename,basename);
                strcat(filename,type);
                ID = cpgopen(filename);
                /*__________________________________
                * Change the orientation of the plots here
                *___________________________________*/
                if (max_sub_win ==2) cpgsubp(1,2);
                if (max_sub_win >=3) cpgsubp(2,2);
            }
        }                
    }
    
/*__________________________________
*   All subsequent passes through
*___________________________________*/       
    if ( first_time_through >1 )
    {
        /*__________________________________
        *   reset some counters
        *___________________________________*/
        if( n_open_files == n_windows  )
        {           
            counter ++;
            *n_sub_win   =   0;      
        }
        /*__________________________________
        *   No subwindows
        *___________________________________*/ 
        if ( max_sub_win == 0 ) 
        {
            /*__________________________________
            *   Determine the filename and open the 
            *   file
            *___________________________________*/
            strcpy(filename,filepath);
            sprintf(indx, "%d",counter);
            strcat(filename,indx);
            strcat(filename,basename);
            strcat(filename,type);
            ID = cpgopen(filename);      
        } 
                     
        /*__________________________________
        *   If there are sub windows then set
        *   the sub window layout
        *___________________________________*/
         if( max_sub_win > 0)
         {
            *n_sub_win = *n_sub_win + 1;

            if (*n_sub_win == 1 )
            {
                /*__________________________________
                *   Determine the filename and open the 
                *   file
                *___________________________________*/
                sprintf(indx, "%d",counter);
                strcpy(filename,filepath);
                strcat(filename,indx);
                strcat(filename,basename);
                strcat(filename,type);
                cpgopen(filename);   
                if (max_sub_win ==2) cpgsubp(1,2);
                if (max_sub_win >=3) cpgsubp(2,2);
            }
        }
    }      
    cpgslct(ID);
}


/* ======================================================================*/
#include <stdio.h>
#include "cpgplot.h"
#include "parameters.h"
#include "nrutil+.h"
#include "functionDeclare.h"

/* 
 Function: plot_generate_axis
 File Name: plot_common.c

 Purpose: generate the plot axis and label

 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       05/12/99   Written   

 IN args/commons        Units      Description
 ---------------        -----      -----------
 label                  char        label to put on plot
 x_min, x_max           float       max and min of x-axis
 y_min, y_max           float       max and min of y-axis 
 
  steps:
    1)  Determine the size of the characters that should be used in the 
        axis labels and the text
    2)  Generate the axis and label the plot
    3)  If one of the numbers is NAN or INF then print a warning message
_______________________________________________________________________ */

    void plot_generate_axis(  
    
    char    x_label[],                  /* label along x axis           */
    char    y_label[],                  /* label along y axis           */
    char    graph_label[],              /* label along the top of plot  */    
    float   *x_min, 
    float   *x_max,
    float   *y_min,
    float   *y_max,
    int     *error          )           /* if NAN or INF has been found */
{
    int     unit = 2;                   /* units of measure 2= mm       */
            
    float   x1,x2,y1, y2;
    float   char_size_axis,
            char_size_text,
            c_x, c_y, delx, dely;
/*__________________________________
* Determine the size of the text
*___________________________________*/
    cpgqvp(unit,&x1, &x2, &y1, &y2);

    delx            = x2 - x1;
    dely            = y2 - y1;
    
    c_x             = 60.0/delx;
    c_y             = 60.0/dely;
    char_size_axis  = FMAX(1.0, c_x);
    char_size_axis  = FMAX(char_size_axis, c_y);
    
    c_x             = 80.0/delx;
    c_y             = 80.0/dely;    
    char_size_text  = FMAX(1.0, c_x);
    char_size_text  = FMAX(char_size_text, c_y);

/*     fprintf(stderr,"x1 %f, x2 %f, y1 %f y2 %f \n",x1, x2, y1, y2); */
/*__________________________________
* Generate axis and label plot
*___________________________________*/
    cpgsch(char_size_axis);
    cpgenv(*x_min, *x_max, *y_min, *y_max, 0, show_grid);
    cpgsch(char_size_text);
    cpglab(x_label, y_label,    graph_label);
/*     cpgmtxt("B\0",2.0, 0.5, 0.5, label); */
/*__________________________________
*   testing labels
*___________________________________*/
/*     cpglab("test of labeling\0","yaxis of labeling\0","this is a tthes of the american roadadfadsfadfcacst tystem let us see how well it wraps aroun the workd again and again and again\0"); */
    cpgsch(1.0); 
    
/*__________________________________
*   If NAN or INF has been detected
*   in the data array then warn the user
*___________________________________*/   
    if (*error == 1) 
    {
        cpgsch(char_size_text);
        cpgmtxt("T\0",2.5, 0.5, 0.5, "NAN or INF has been detected\0");
        cpgsch(1.0);
    }
}


/* ======================================================================*/
#include <stdio.h>
#include <math.h>
#include "cpgplot.h"
#include "parameters.h"
#include "nrutil+.h"
#include "functionDeclare.h"

/* 
 Function: plot_legend
 File Name: plot_common.c

 Purpose: generate a color wedge on top of a contour plot
            add labels along with the data max and min values

 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       05/12/99   Written   

 IN args/commons        Units      Description
 ---------------        -----      -----------
 label                  char        label to put on plot
 x_min, x_max           float       max and min of x-axis
 y_min, y_max           float       max and min of y-axis 
 
  steps:
_______________________________________________________________________ */

void    plot_legend( 
    float   data_max,   
    float   data_min,
    float   x_max,
    float   y_max           )
{
    int     i, 
            unit = 2;                   /* units of measure 2= mm       */     
            
    float   x1,     x2,     y1,         y2;
    float   char_size_axis,
            char_size_text,
            text,
            height,
            offset,
            c_x,    c_y,    
            delx,   dely;
    char    label[15];
/*__________________________________
* Determine the size of the text
*___________________________________*/
    cpgqvp(unit,&x1, &x2, &y1, &y2);

    delx            = x2 - x1;
    dely            = y2 - y1;
    
    c_x             = 70.0/delx;
    c_y             = 70.0/dely;
    char_size_axis  = FMAX(0.0, c_x);
    char_size_axis  = FMAX(char_size_axis, c_y);
    
    c_x             = 80.0/delx;
    c_y             = 80.0/dely;    
    char_size_text  = FMAX(1.0, c_x);
    char_size_text  = FMAX(char_size_text, c_y);
    
/*__________________________________
*   Set th Wedge function and color range
*___________________________________*/
    cpgscir(1,NUM_COLORS);
/*__________________________________
*   Define the color wedge
*___________________________________*/
    cpgsch(char_size_axis);
    cpgsci(0);
    cpgwedg("RI\0",0.0, char_size_text, 0.0, 1.0, " \0");
/*__________________________________
*   Generate the label for the wedge
*   10 labels per wedge
*___________________________________*/
    cpgsch(char_size_axis);
    cpgstbg(0);
    cpgsci(1);
/*__________________________________
*   Now generate the location of where 
*   to place the label and the labels 
*   themselves
*___________________________________*/
     offset      = 0.0454550*fabs(data_max - data_min);
    data_min    = data_min + offset;
    data_max    = data_max - offset;
    
    for( i = 0; i <=10; i++)
    {
        height  = (float)i/10.0;
        text    = data_min + (float) (i) * (data_max - data_min)/10.0;
        sprintf(label, "%.4g",text); 
         
        cpgptxt(x_max*1.02, y_max*height, 30.0, 0.0,label);  
    } 
        
}


/*
 ======================================================================*/

 #include <assert.h>
 #include <math.h>
 #include <stdio.h>
 #include "parameters.h"
 #include "functionDeclare.h"
 #define DIFFERENCE 1.0e-3
/*
 Function: plot_scaling
 File Name: plot_common.c

 Purpose:  Find the max and min values for a vector

 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       05/12/99   Written   

 IN args/commons        Units      Description
 ---------------        -----      -----------
 data_array             *float  
_______________________________________________________________________ */

int plot_scaling( 
    const float  *data_array,                        
    int    max_len,                        
    float  *y_min, 
    float   *y_max      )
        
{ 
    float   
            offset;
            
    int     i,
            error;
/*__________________________________
* initialize
*___________________________________*/
    *y_min = 1.0e10;
    *y_max = 0.0;
    error  = 0;
       
/*______________________________
  Now find the max and min of the 
  array
______________________________  */                
    for(i = 1; i <= max_len; i++)
    {
       if (isnan(data_array[i] ) == 1) error = 1;
        
       if ( (data_array[i]) >= *y_max) 
       {
            *y_max = data_array[i];
       }
       if ( (data_array[i]) <= *y_min)
       { 
            *y_min = data_array[i];  
       }
    }
/*__________________________________
*   if y_min = y_max
*___________________________________*/
    offset = 0.05*fabs(*y_max - *y_min);
    *y_min = *y_min - offset;
    *y_max = *y_max + offset;
    
/*__________________________________
* if y_min or y_max =0.0
*___________________________________*/
    if( fabs(*y_min) <= DIFFERENCE ) *y_min = -0.1;
    if( fabs(*y_max) <= DIFFERENCE ) *y_max =  0.1;
    
    return error;         
 } 
 
 
/* 
 ======================================================================*/
#include "cpgplot.h"
#include "parameters.h"
#include "functionDeclare.h"
/* 
 Function: plot_color_spectrum
 File Name: plot_common.c

 Purpose:  This routine generates a color spectrum and color index that
is used to to mark the individual particles.

       (Red, green, blue)
      (1, 0, 0)  to (1, 1, 0)  loop 1     red
      (1, 1, 0) to  (0, 1, 0)  loop 2
      (0, 1, 0) to  (0, 0, 1)  loop 3
      (0, 0, 1) to  (0, 0, 0)  loop 4     blue
      An index is assigned for each different color.  Note that the range 
      the index is 2 < index < num_color.  The indices 0, 1 have special
      meaning and should not be touched.
      
      Your data should be scaled as shown below

      color = ((NUM_COLORS-1)*varialble/max(variable)) + 2

      Input: NUM_COLORS    I      Number of colors in the spectrum 

 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       05/12/99   Written   

 IN args/commons        Units      Description
 ---------------        -----      -----------
 NUM_COLORS             int         parameters.h
_______________________________________________________________________ */

void plot_color_spectrum(void)
{
       int      i,  color_index, n_colors;
       float    color, increment;
/*______________________________________________________________________
*   Initialize some coloriables
*_______________________________________________________________________*/

                 
                 color = 0.0;
                 n_colors = (int)NUM_COLORS/4;
                 color_index = (n_colors*4 ) + 2;
                 increment = 1.0/(float)n_colors;
/*__________________________________
*   Loop 1
*___________________________________*/                                 
                 for ( i=1; i<=n_colors; i++)
                 {
                     color_index = color_index - 1;
                     color = color + increment;
                     if (color > 1.0) color = 1.0;
                     cpgscr(color_index,1.0,color,0.0);
                     
/*                      print*, color_index,1.0, color, 0.0 */
                 }
/*__________________________________
*   Loop 2
*___________________________________*/                 
                 color = 1.0;
                 for ( i=1; i<=n_colors; i++)
                 {
                     color_index = color_index - 1;
                     color = color - increment;
                     
                     if (color < 0.0) color = 0.0;
                     cpgscr(color_index,color,1.0,0.0);
                     
/*                       print*, color_index,color, 1.0, 0.0 */
                 }
/*__________________________________
*   Loop 3
*___________________________________*/ 
                
                 color = 0.0;
                 for ( i=1; i<=n_colors; i++)
                 {
                     color_index = color_index - 1;
                     color = color + increment;
                     if (color > 1.0) color = 1.0;
                     cpgscr(color_index,0.0,1.0,color);
                     
/*                      print*, color_index,0.0, 1.0, color */
                 }
/*__________________________________
*   Loop 4
*___________________________________*/ 
                
                 color = 1.0;
                 for ( i=1; i<=n_colors; i++)
                 {
                     color_index = color_index - 1;
                     color = color - increment;
                     if (color < 0.0) color = 0.0;
                     cpgscr(color_index,0.0,color,1.0);
                     
/*                      print*, color_index,0.0, 1.0, color */
                 }
/*__________________________________
* redefine black (0) and white (1)
*___________________________________*/
            cpgscr(1,0.0,0.0,0.0);
            cpgscr(0, 1.0, 1.0, 1.0);    
}


 
 
 /*______________________________________________________________________
 *  EXTRA CODE TO BE EVENTUALLYTHROWN OUTS
 *_______________________________________________________________________*/
 
 
 /*  */
 
 
 /* 
 ======================================================================*/
 #include <assert.h>
 #include "parameters.h"
#include "functionDeclare.h"
/* 
 Function: plot_scaling_CC
 File Name: plot_common.c

 Purpose:  Find the max and min values for a single material array

 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       05/12/99   Written   

 IN args/commons        Units      Description
 ---------------        -----      -----------
 data_array             ***double  (x,y,z)
 m                      int         material  
_______________________________________________________________________ */

void plot_scaling_CC( 

    int     xLoLimit,                   /* x-array lower limit              */
    int     yLoLimit,                   /* y-array lower limit              */
    int     zLoLimit,                   /* z-array lower limit              */
    int     xHiLimit,                   /* x-array upper limit              */
    int     yHiLimit,                   /* y-array upper limit              */
    int     zHiLimit,                   /* z-array upper limit              */            
    double  ***data_array,
    float   *y_min,
    float   *y_max          )
        
{ 
    int i, j, k;
/*__________________________________
* initialize and hardwire
*___________________________________*/
    *y_min = 1.0e10;
    *y_max = 0.0;
/*__________________________________
* double check inputs
*___________________________________*/
    assert ( xLoLimit >= 0 && xHiLimit < X_MAX_LIM);
    assert ( yLoLimit >= 0 && yHiLimit < Y_MAX_LIM);
    assert ( zLoLimit >= 0 && zHiLimit < Z_MAX_LIM);       
/* ______________________________
  Now find the max and min of the 
  array
______________________________  */                
    for(k = zLoLimit; k <= zHiLimit; k++)
    {
        for(j = yHiLimit; j >= yLoLimit; j--)
        {
            for(i = xLoLimit; i <= xHiLimit; i++)
            {
               if ( data_array[i][j][k] >= *y_max) {
                    *y_max = (float) data_array[i][j][k];
                }
               if ( data_array[i][j][k] <= *y_min) {
                    *y_min = (float) data_array[i][j][k];
                }  
               
            }
        }
    }
       
 }
 
 
/* 
 ======================================================================*/
 #include "parameters.h"
 #include <assert.h>
#include "functionDeclare.h"
/* 
 Function: plot_scaling_CC_MM
 File Name: plot_common.c

 Purpose:  Find the max and min values for a multi-material array

 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       05/12/99   Written   

 IN args/commons        Units      Description
 ---------------        -----      -----------
 data_array             ****double  (x,y,z,material)
 m                      int         material  
_______________________________________________________________________ */

void plot_scaling_CC_MM( 
                        
    int     xLoLimit,                   /* x-array lower limit              */
    int     yLoLimit,                   /* y-array lower limit              */
    int     zLoLimit,                   /* z-array lower limit              */
    int     xHiLimit,                   /* x-array upper limit              */
    int     yHiLimit,                   /* y-array upper limit              */
    int     zHiLimit,                   /* z-array upper limit              */ 
    double  ****data_array,
    int     m,                          /* material                         */
    float   *y_min, 
    float   *y_max          )
        
{ 
    int i, j, k;
/*__________________________________
* initialize and hardwire
*___________________________________*/
    m=1;
    *y_min = 1.0e10;
    *y_max = 0.0;
/*__________________________________
* double check inputs
*___________________________________*/
    assert ( xLoLimit >= 0 && xHiLimit < X_MAX_LIM);
    assert ( yLoLimit >= 0 && yHiLimit < Y_MAX_LIM);
    assert ( zLoLimit >= 0 && zHiLimit < Z_MAX_LIM);       
/* ______________________________
  Now find the max and min of the 
  array
______________________________  */                
    for(k = zLoLimit; k <= zHiLimit; k++)
    {
        for(j = yHiLimit; j >= yLoLimit; j--)
        {
            for(i = xLoLimit; i <= xHiLimit; i++)
            {
               if ( data_array[m][i][j][k] >= *y_max) 
               {
                    *y_max = data_array[m][i][j][k];
               }
               if ( data_array[m][i][j][k] <= *y_min)
               { 
                    *y_min = (float)data_array[m][i][j][k];  
               }
            }
        }
    }
       
 }

