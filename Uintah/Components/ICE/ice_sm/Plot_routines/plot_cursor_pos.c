/* ======================================================================*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cpgplot.h"
#include "parameters.h"
#include "nrutil+.h"
#include "functionDeclare.h"
/* 
 Function:  plot_find_cursor_pos--VISUALIZATION: Allows user to get data from the window by clicking the mouse.
 Filename:  plot_cursor_pos.c

 Purpose:   This function allows the user to select points on a plot and it 
            returns the x and y coordinates of that plot.  This is usefull
            in debugging.  For lack of a better term I call it cursor data.
 Steps:
 ------
    1) The first time through the main loop have the user select the windows
       that cursor data is desired.  The array "get_cursor_data" contains
       the flags that are used to determine which windows are to be investigated
       with the cursor.
   
    2) For subsequent passes through the routine select the correct window
       the user simply selects points on the window and the routing prints
       to stderr the x y coordinates.
    

 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       3/08/99   Written   
       
WARNING 
    Don't call any of the plotting routine inside of an iterative loop
       
_______________________________________________________________________ */

    void plot_cursor_position(
    int        max_sub_win,             /* max number of sub_windows                */
    int        *n_sub_win )             /* counter for the number of sub            */
                                        /* windows                                  */

{
    int     x_int, y_int;               /* position x and y converted to an integer */
                
    float   x, y; 
                                           
   static  int 
            first_time_through,         /* if this is the first time                */ 
            n_windows,                  /* number of open windows counter           */
            max_main_win,               /* max number of windows opened             */
            get_cursor_data[12];        /* array of switches for getting cursor     */
                                        /* data for that particular window          */
            
    char    am_I_here,
            ans;                        /* mouse input or answer                    */ 

     double junk;       
         
/*______________________________________________________________________
*   MAIN CODE
*_______________________________________________________________________*/
    
/*__________________________________
*   determine if this is the first time through
*___________________________________*/ 
    am_I_here= *getenv("PGPLOT_I_AM_HERE");        
    if ( am_I_here== '0' )  first_time_through = 1;
    if ( am_I_here== '1' )  first_time_through = 2;
      
/*__________________________________
*   First pass through 
*___________________________________*/ 
    if ( first_time_through == 1 )
    {
        /*__________________________________
        * No subwindows then just ask user
        * if they want that window interactive
        *___________________________________*/
        if ( max_sub_win == 0 ) 
        {
            max_main_win ++;    
                
            cpgsci(NUM_COLORS);
            cpgsch(1.25 );
            cpgscf(1);
            cpgmtxt("T\0", 0.0, 0.5, 0.5, "Press Right mouse button to make this window interactive"); 
            cpgsci(1);  
            cpgcurs(&x,&y,&ans);

            if(ans == 'X') 
            {
                get_cursor_data[max_main_win] = YES;
                /* fprintf(stderr, "ans %s,   array data[%i]= %i\n",&ans, max_main_win,get_cursor_data[max_main_win]); */
            } 
        }   
        /*__________________________________
        * If there are sub windows then ask
        * the user if they want it interactive
        * store the answer in get_cursor_data
        *___________________________________*/
         if( max_sub_win > 0)
         {
            *n_sub_win = *n_sub_win + 1;
            if (*n_sub_win == max_sub_win )
            {
                max_main_win ++;
                cpgsci(NUM_COLORS);
                cpgsch(1.25);
                cpgscf(1);
                cpgmtxt("B\0", 2.0, 0.5, 0.5, "Press Right mouse button to make this window interactive");
                cpgsci(1);  
                cpgcurs(&x,&y,&ans);

                if(ans == 'X') 
                {
                    get_cursor_data[max_main_win] = YES;
                    /* fprintf(stderr, "ans %s,   array data[%i]= %i\n",&ans, max_main_win,get_cursor_data[max_main_win]);  */
                }
            }
        } 
    
   }    
/*__________________________________
*   All subsequent passes through
*   this routine
*___________________________________*/       
    if ( first_time_through >1 )
    {
        /*__________________________________
        *   If there are no subwindows 
        *___________________________________*/
        if ( max_sub_win == 0 ) 
        {
            /*__________________________________
            * Find the window index
            *___________________________________*/
            n_windows   ++;
            /*__________________________________
            * If get cursor data = yes then get it
            *___________________________________*/    
            if(get_cursor_data[n_windows] == YES)
            {
                fprintf(stderr,"\n Click left to see data and right to exit\n");
                ans = 'n';
                /*__________________________________
                * Until the right mouse is clicked
                * get the x y coordinates and 
                * convert them into (int) and 
                * print it to stderr.
                *___________________________________*/
                while( ans != 'X'  )
                {
                    
                    cpgcurs(&x,&y,&ans);
                    /*__________________________________
                    * Now round off the x y data into ints
                    *___________________________________*/
                    if( modf((double)x, &junk) < 0.5)
                        x_int =(int) x;
                        
                    if( modf((double)x, &junk) >= 0.5)
                        x_int =(int) x + 1;

                    if( modf((double)y, &junk) < 0.5)
                        y_int =(int) y;
                        
                    if( modf((double)y, &junk) >= 0.5)
                        y_int =(int) y + 1;
                    fprintf(stderr,"xpos %f \t %i \t ypos %f \t %i\n",x, x_int, y, y_int);
                    /* fprintf(stderr, "character %s\n",&ch); */
                }
            }
        } 
                     
        /*__________________________________
        *   If there are sub windows 
        *___________________________________*/
         if( max_sub_win > 0 )
         {
            /*__________________________________
            * Find the window index
            *___________________________________*/
            *n_sub_win = *n_sub_win + 1;
            if (*n_sub_win == 1 )
            {
                n_windows   ++;
            }
            /*__________________________________
            * If get cursor data = yes then get it
            *___________________________________*/
            if(get_cursor_data[n_windows] == YES)
            {
                fprintf(stderr,"\n Click left to see data and right to exit\n");
                /*__________________________________
                * Until the right mouse is clicked
                * get the x y coordinates and 
                * convert them into (int) and 
                * print it to stderr.
                *___________________________________*/
                while( ans != 'X' )
                {
                    cpgcurs(&x,&y,&ans);
                    
                    /*__________________________________
                    * Now round off the x y data into ints
                    *___________________________________*/
                    if( modf((double)x, &junk) < 0.5)
                        x_int =(int) x;
                        
                    if( modf((double)x, &junk) >= 0.5)
                        x_int =(int) x + 1;

                    if( modf((double)y, &junk) < 0.5)
                        y_int =(int) y;
                        
                    if( modf((double)y, &junk) >= 0.5)
                        y_int =(int) y + 1;
                    fprintf(stderr,"xpos %f \t %i \t ypos %f \t %i\n",x, x_int, y, y_int);
                    /* fprintf(stderr, "character %s\n",&ch); */
                }
            } 
        }
    }
    /*__________________________________
    *   reset the number of windows count
    *___________________________________*/
    if (n_windows == max_main_win && *n_sub_win == max_sub_win) 
    {
        n_windows   = 0;
    }
   /*  fprintf(stderr, "win_indx %i max_main_win %i \n", win_indx, max_main_win);    */   
}
/*STOP_DOC*/
