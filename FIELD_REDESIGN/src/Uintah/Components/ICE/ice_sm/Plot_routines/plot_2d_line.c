 /* 
 ======================================================================*/
#include "cpgplot.h"
#include "functionDeclare.h"
#include "parameters.h"
/* 
 Function:  plot_2d_scatter--Visualizaton: Generates a 2D line plot.
 Filename:  plot_2d_line
 Purpose:
 This subroutine generates a 2-d scatter plot of the input variables.

 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       05/12/99   Written   

 IN args/commons        Units      Description
 ---------------        -----      -----------
 
 The length of the arrays is max_len
_______________________________________________________________________ */
    void plot_2d_scatter(
        const float     *x_data,
        const float     *y_data,
        int             max_len,
        int             color,
        int             symbol,
        float           symbol_size    )
{                  

/*______________________________________________________________________
*   Generate the plot window #1
*_______________________________________________________________________*/  
    if ( max_len != 1) 
    {        
        cpgsci(color);
        cpgsch(symbol_size);
        cpgpt(max_len,    &x_data[1], &y_data[1],symbol);
        cpgline(max_len,  &x_data[1], &y_data[1]);
        cpgsci(1);
    }  
} 
/*STOP_DOC*/ 





 /* 
 ======================================================================*/
#include <math.h>
#include "parameters.h"
/* 
 Function:  plot_dbl_flt--VISUALIZATION: converts a vector of doubles to floats.
 Filename:  plot_2d_line
 Purpose:
 Converts a 1-D double array into a 1D float array

 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       05/12/99   Written   

 IN args/commons        Units      Description
 ---------------        -----      -----------
 dbl_1                  double    4d array
 flt_1                  float     1-d array
 max_len                  int       maximum length of array
 
Note this function assumes that the arrays starting index is 1
_______________________________________________________________________ */
    void plot_dbl_flt(                        
        int     max_len,
        double  ****dbl_1,
        double  ****dbl_2,
        float   *flt_1,
        float   *flt_2 )
               
{       
     int  i;

       for (i = 1; i <= max_len; i++)
       {   
          flt_1[i] = (float)dbl_1[i+N_GHOSTCELLS][N_GHOSTCELLS][N_GHOSTCELLS][1];
          flt_2[i] = (float)dbl_2[i+N_GHOSTCELLS][N_GHOSTCELLS][N_GHOSTCELLS][1];
       }
}
/*STOP_DOC*/

