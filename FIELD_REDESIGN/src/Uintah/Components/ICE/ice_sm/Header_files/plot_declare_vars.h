
/* 
=====================================================================*/
#include "cpgplot.h"



#define PLOT plot(data_array1, data_array2, max_len,delX, x_axis_origin,x_axis, y_axis_origin,y_axis, \
x_label, y_label,graph_label,plot_type,outline_ghostcells,Number_sub_plots,file_basename, outputfile_type) 

/*
 FILE NAME:  plot_declare_vars.h 
 Purpose:   This file defines all of the variables necessary for plotting
 To use is simply add #include "plot_declare_vars.h" before the first line of code in the 
 function     
__________________________________________________________________________*/
/*__________________________________
*   Plotting variables
*___________________________________*/ 
    int     
        plot_type,
        Number_sub_plots,
        outputfile_type,
        x_axis_origin,                  /* origin of x and y axis           */
        y_axis_origin,
        x_axis,                         /* upper limit if x axis            */
        y_axis,
        max_len,
        outline_ghostcells = 0;         /* = 1 if you want to outline GC    */
static int  first_pass = 0;             /* How many time through function   */    

    char
        x_label[30],                    /* x axis label                     */
        y_label[30],                    /* y axis label                     */
        graph_label[50],                /* label at top of the plot         */
        file_basename[30],
        stay_or_go;                     /* execute a section of code or not */
float
        *data_array1,                   /* 1-D vector containing data       */
        *data_array2;                   /* 1-D vector containing data       */

static float
        *time_axis;                     /* axis used for time               */
    
/*__________________________________
*   KEEP FULL WARN QUITE    
*___________________________________*/
    time_axis   = time_axis;
    data_array2 = data_array2;
    stay_or_go  = stay_or_go;
    first_pass  = first_pass;
