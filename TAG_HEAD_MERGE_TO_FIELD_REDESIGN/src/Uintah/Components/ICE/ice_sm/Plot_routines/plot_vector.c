/* 
 ======================================================================*/
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include "cpgplot.h"
#include "parameters.h"
#include "nrutil+.h"
#include "functionDeclare.h"

/* 
 Function:  plot_vector--VISUALIZATION: Generates a 2D vector plot of input data.
 Filename:  plot_vector.c

 Purpose:  Plot a vector plot of the input field

 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       09/2/99   Written   
_______________________________________________________________________ */
void    plot_vector_2D(
  const int     xLoLimit,
  const int     yLoLimit,
  const int     xHiLimit,               /* x-array upper limit              */
  const int     yHiLimit,               /* y-array upper limit              */
  const int     max_len,                /* max length of data arrays        */
  const float   *data_array1,           /* data vector                      */
  const float   *data_array2)           /* second set of data used for      */
                                        /* vector plots                     */
{
    int     i,
            unit = 2,                   /* units of measure 2 = mm          */
            xtemp, ytemp;
    float *TR;    
    float   vector_scale,               /* scale for the vector             */
            vel_vector,
            max_vector_len,             /* length of longest vector         */
            max_velocity,               /* maximum velocity                 */
            A, B,                       /* Temp varables                    */
            char_size_text,             /*scale text according to win size  */
            delx, dely,                 /* window width and height          */
            x1, x2, y1, y2,             /* window corners                   */
            c_x, c_y;                   /* char size in x and y dirs        */
    char    label[10]; 
       
/*______________________________________________________________________
*   Hardwire some vector plot parameters
*_______________________________________________________________________*/
    TR      = vector_nr(0,6);
    TR[0]   = xLoLimit- 0.5;
    TR[1]   = 1.0;
    TR[2]   = 0.0;
    TR[3]   = yLoLimit- 0.5;
    TR[4]   = 0.0;
    TR[5]   = 1.0;
    
    
    vector_scale    = 1;
    max_velocity    = 0.0;
    max_vector_len  = 1.0;

/*__________________________________
*   Find the max velocity, and the 
*   vector scale
*___________________________________*/
    for ( i = 1; i <= max_len; i++)
    {
        
        A               = pow(data_array1[i], 2);
        B               = pow(data_array2[i], 2);
        vel_vector      = sqrt( A + B );
        max_velocity    = FMAX(max_velocity, vel_vector);   
    }
/*__________________________________
*   Change the shape of the arrows and the
* size of the plot data
*___________________________________*/  
    cpgsch(0.7);
    cpgsah(1,40.0, 0.0);    
    
    vector_scale = max_vector_len/max_velocity;    

/*__________________________________
*   Generate the vector field
*___________________________________*/
    xtemp = xHiLimit - xLoLimit + 1;
    ytemp = yHiLimit - yLoLimit + 1;
    cpgvect(        &data_array1[1],&data_array2[1], 
                    xHiLimit,      yHiLimit,
                    1,              xtemp,
                    1,              ytemp,
                    vector_scale,   1, 
                    TR,             0.0);                     
/*__________________________________
* Determine the size of the text
*___________________________________*/
    cpgqvp(unit,&x1, &x2, &y1, &y2);

    delx            = x2 - x1;
    dely            = y2 - y1;
    
    c_x             = 70.0/delx;
    c_y             = 70.0/dely;

    c_x             = 80.0/delx;
    c_y             = 80.0/dely;    
    char_size_text  = FMAX(1.0, c_x);
    char_size_text  = FMAX(char_size_text, c_y);
/*__________________________________
*   Add labels
*___________________________________*/    
    cpgsch(char_size_text);
    cpgmtxt("L\0", 2.5, 0.0, 0.0, "Max.velocity\0");
    sprintf(label, "%5.5g",max_velocity);
    cpgmtxt("L\0", 1.5, 0.0, 0.0, label);
    
    
    
    free_vector_nr( TR, 0, 6);
}
/*STOP_DOC*/    
