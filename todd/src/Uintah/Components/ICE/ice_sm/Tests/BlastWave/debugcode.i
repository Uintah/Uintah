
/*______________________________________________________________________
*   This file contains all of the debugging and ploting code for all of the
*   functions.
*   main.c
*
*   Plotting options:
*       outputfile_type     0 dump to the screen
*                           2   Postscript file
*       plot_type           1   Contour plot
*                           2   2D line
*
*_______________________________________________________________________*/

#if switchInclude_main_custom
   
    #if (switchDebug_main_custom == 1)           
       /*__________________________________
        * Define plotting variables 
        *___________________________________*/
        m                   = 1; 
        plot_type           = 1;
        Number_sub_plots    = 4;
        strcpy(file_basename,"");
        outputfile_type     = 0;
        x_axis_origin       = (xLoLimit);
        y_axis_origin       = (yLoLimit);
        x_axis              = (xHiLimit);
        y_axis              = (yHiLimit);  
        outline_ghostcells  = 0;
        strcpy(x_label,"cell\0");
        strcpy(y_label,"cell\0");                      
         
/*__________________________________
*   MAIN WINDOW 1
*   outputfile_type = 2 postscript
*___________________________________*/
        outputfile_type     = 0;
        Number_sub_plots    = 4;  
        outline_ghostcells  = 0;        
        strcpy(graph_label,"Main Program rho_CC\0");
        plot_type           = 1;
        /*__________________________________
        * Variable 1
        *___________________________________*/
        strcpy(y_label,"[kg/m^3]\0");
        data_array1    = convert_darray_4d_to_vector(
                                rho_CC,
                                (xLoLimit),             (xHiLimit),         (yLoLimit),
                                (yHiLimit),             (zLoLimit),          (zHiLimit),    
                                m,                      m,
                                &max_len);
        PLOT;

        free_vector_nr(    data_array1,       1, max_len);
        
        /*__________________________________
        *   Variable 2
        *___________________________________*/
        plot_type = 1;
        strcpy(y_label,"rho/Rho_s\0");
        strcpy(graph_label,"Main Program rho/rho_s\0");
        data_array1    = convert_darray_3d_to_vector(
                                rho_exact,
                                (xLoLimit),             (xHiLimit),         (yLoLimit),
                                (yHiLimit),             (zLoLimit),          (zHiLimit),    
                                &max_len); 
        PLOT;
         free_vector_nr(    data_array1,       1, max_len);
        /*__________________________________
        *   Variable 3
        *___________________________________*/
        strcpy(y_label,"K\0");
        strcpy(graph_label,"Main Program Temp_CC\0");
        data_array1    = convert_darray_4d_to_vector(
                                Temp_CC,
                                (xLoLimit),             (xHiLimit),         (yLoLimit),
                                (yHiLimit),             (zLoLimit),          (zHiLimit),    
                                m,                      m,
                                &max_len);
        PLOT;
         free_vector_nr(    data_array1,       1, max_len);
         
        /*__________________________________
        *   Variable 4
        *___________________________________*/
        strcpy(y_label,"T/T_s");
        strcpy(graph_label,"Main Program T/Ts\0");
        data_array1    = convert_darray_3d_to_vector(
                                Temp_exact,
                                (xLoLimit),             (xHiLimit),         (yLoLimit),
                                (yHiLimit),             (zLoLimit),          (zHiLimit),    
                                &max_len);
        PLOT;
         free_vector_nr(    data_array1,       1, max_len);
/*__________________________________
*   MAIN WINDOW 2
*___________________________________*/
        x_axis_origin       = (xLoLimit);
        y_axis_origin       = (yLoLimit);
        x_axis              = (xHiLimit);
        y_axis              = (yHiLimit);   
        outputfile_type     = 0;
        Number_sub_plots    = 4;  
        /*__________________________________
        *   Variable 1
        *___________________________________*/
        plot_type = 1 ;
        strcpy(graph_label,"Main Program vvel_CC\0");
        data_array1    = convert_darray_4d_to_vector(
                                vvel_CC,
                                (xLoLimit),             (xHiLimit),         (yLoLimit),
                                (yHiLimit),             (zLoLimit),          (zHiLimit),  
                                m,                      m,
                                &max_len); 
        PLOT;
         free_vector_nr(    data_array1,       1, max_len); 
         
        /*__________________________________
        *   Variable 2
        *___________________________________*/
        plot_type = 1 ;
        strcpy(graph_label,"Main Program vvel_exact\0");
        data_array1    = convert_darray_3d_to_vector(
                                vvel_exact,
                                (xLoLimit),             (xHiLimit),         (yLoLimit),
                                (yHiLimit),             (zLoLimit),          (zHiLimit),  
                                &max_len); 
        PLOT;
         free_vector_nr(    data_array1,       1, max_len);    
        /*__________________________________
        * Variable 3
        *___________________________________*/
        plot_type = 1;
        strcpy(graph_label,"Main Program uvel_CC\0");
        data_array1    = convert_darray_4d_to_vector(
                                uvel_CC,
                                (xLoLimit),             (xHiLimit),         (yLoLimit),
                                (yHiLimit),             (zLoLimit),          (zHiLimit),   
                                m,                      m,
                                &max_len); 
        PLOT;
         free_vector_nr(    data_array1,       1, max_len);
         
       /*__________________________________
        * Variable 4
        *___________________________________*/
        plot_type = 1;
        strcpy(graph_label,"Main Program uvel_exact\0");
        data_array1    = convert_darray_3d_to_vector(
                                uvel_exact,
                                (xLoLimit),             (xHiLimit),         (yLoLimit),
                                (yHiLimit),             (zLoLimit),          (zHiLimit),   
                                &max_len); 
        PLOT;
         free_vector_nr(    data_array1,       1, max_len);

        /*__________________________________
        *  Clean up the plotting windows
        *___________________________________*/ 
        first_pass = 1;
        
        
/*__________________________________
*   MAIN WINDOW 3
*   outputfile_type = 2 postscript
*___________________________________*/
        x_axis_origin       = (xLoLimit);
        y_axis_origin       = (yLoLimit);
        x_axis              = (xHiLimit);
        y_axis              = (yHiLimit); 
        outputfile_type     = 0;
        Number_sub_plots    = 2;  
        outline_ghostcells  = 0;        
        strcpy(graph_label,"Main Program press_CC\0");
        plot_type           = 1;
        /*__________________________________
        * Variable 1
        *___________________________________*/
        strcpy(y_label,"[Pa]\0");
        data_array1    = convert_darray_4d_to_vector(
                                press_CC,
                                (xLoLimit),             (xHiLimit),         (yLoLimit),
                                (yHiLimit),             (zLoLimit),          (zHiLimit),    
                                m,                      m,
                                &max_len);
        PLOT;

        free_vector_nr(    data_array1,       1, max_len);
        
        /*__________________________________
        *   Variable 2
        *___________________________________*/
        plot_type = 1;
        strcpy(y_label,"\0");
        strcpy(graph_label,"Main Program press_press2\0");
        data_array1    = convert_darray_3d_to_vector(
                                press_exact,
                                (xLoLimit),             (xHiLimit),         (yLoLimit),
                                (yHiLimit),             (zLoLimit),          (zHiLimit),    
                                &max_len); 
        PLOT;
         free_vector_nr(    data_array1,       1, max_len);

/*__________________________________
*   MAIN WINDOW 4
*   Line plots of the exact solution
*___________________________________*/ 
        x_axis_origin       = (xLoLimit);
        y_axis_origin       = (yLoLimit);
        x_axis              = (xLoLimit);
        y_axis              = (yHiLimit);   
        outputfile_type     = 0;
        Number_sub_plots    = 4;  
         
        /*__________________________________
        *   Variable 1
        *___________________________________*/
        data_array2     = vector_nr(1, X_MAX_LIM);
        plot_type = 2 +2 ;
        strcpy(graph_label,"Main Program uvel_exact along base\0");
        data_array1    = convert_darray_4d_to_vector(
                                uvel_CC,
                                (xLoLimit),             (xHiLimit),         (ignition_pt_ydir),
                                (ignition_pt_ydir),     (zLoLimit),          (zHiLimit),  
                                m,                      m,
                                &max_len); 
        
        data_array2    = convert_darray_3d_to_vector(
                                uvel_exact,
                                (xLoLimit),             (xHiLimit),         (ignition_pt_ydir),
                                (ignition_pt_ydir),     (zLoLimit),          (zHiLimit),  
                                &max_len); 
        PLOT;
         free_vector_nr(    data_array1,       1, max_len);    
         free_vector_nr(    data_array2,       1, X_MAX_LIM);
       /*__________________________________
        * Variable 2
        *___________________________________*/
        data_array2     = vector_nr(1, X_MAX_LIM);
        plot_type = 2 +2;
        strcpy(graph_label,"Main Program rho\0");
        data_array1    = convert_darray_4d_to_vector(
                                rho_CC,
                                (xLoLimit),             (xHiLimit),         (ignition_pt_ydir),
                                (ignition_pt_ydir),     (zLoLimit),          (zHiLimit),  
                                m,                      m,
                                &max_len); 
        data_array2    = convert_darray_3d_to_vector(
                                rho_exact,
                                (xLoLimit),             (xHiLimit),         (ignition_pt_ydir),
                                (ignition_pt_ydir),     (zLoLimit),          (zHiLimit),  
                                &max_len); 
        PLOT;
         free_vector_nr(    data_array1,       1, max_len);
         free_vector_nr(    data_array2,       1, X_MAX_LIM);                                
       /*__________________________________
        * Variable 3
        *___________________________________*/
        data_array2     = vector_nr(1, X_MAX_LIM);
        plot_type = 2 +2;
        strcpy(graph_label,"Main Program press\0");
        data_array1    = convert_darray_4d_to_vector(
                                press_CC,
                                (xLoLimit),             (xHiLimit),         (ignition_pt_ydir),
                                (ignition_pt_ydir),     (zLoLimit),          (zHiLimit),  
                                m,                      m,
                                &max_len); 
        data_array2    = convert_darray_3d_to_vector(
                                press_exact,
                                (xLoLimit),             (xHiLimit),         (ignition_pt_ydir),
                                (ignition_pt_ydir),     (zLoLimit),          (zHiLimit),  
                                &max_len); 
        PLOT;
         free_vector_nr(    data_array1,       1, max_len);
         free_vector_nr(    data_array2,       1, X_MAX_LIM);
        /*__________________________________
        * Variable 4
        *___________________________________*/
        data_array2     = vector_nr(1, X_MAX_LIM);
        plot_type = 2 +2;
        strcpy(graph_label,"Main Program Temp\0");
        data_array1    = convert_darray_4d_to_vector(
                                Temp_CC,
                                (xLoLimit),             (xHiLimit),         (ignition_pt_ydir),
                                (ignition_pt_ydir),     (zLoLimit),          (zHiLimit),  
                                m,                      m,
                                &max_len); 
        data_array2    = convert_darray_3d_to_vector(
                                Temp_exact,
                                (xLoLimit),             (xHiLimit),         (ignition_pt_ydir),
                                (ignition_pt_ydir),     (zLoLimit),          (zHiLimit),  
                                &max_len); 
        PLOT;
         free_vector_nr(    data_array1,       1, max_len);
         free_vector_nr(    data_array2,       1, X_MAX_LIM);
    /*__________________________________
    *  vector plot
    *___________________________________*/
    for (m = 1; m <= nMaterials; m++)
    {
        x_axis_origin       = (xLoLimit);
        y_axis_origin       = (yLoLimit);
        x_axis              = (xHiLimit);
        y_axis              = (yHiLimit); 
        plot_type           = 3;
        Number_sub_plots    = 0;
        sprintf(graph_label, "Main Program Velocity Vector Plot, Mat. %d \0",m);
        data_array1    = convert_darray_4d_to_vector(
                                uvel_CC,
                                (xLoLimit),         (xHiLimit),      (yLoLimit),
                                (yHiLimit),         (zLoLimit),      (zHiLimit),
                                m,               m,
                                &max_len); 

        data_array2    = convert_darray_4d_to_vector(
                                vvel_CC,
                                (xLoLimit),         (xHiLimit),      (yLoLimit),
                                (yHiLimit),         (zLoLimit),       (zHiLimit),
                                m,               m,
                                &max_len); 
  
        PLOT;
        free_vector_nr(    data_array1,       1, max_len);
        free_vector_nr(    data_array2,       1, max_len);                      
    }
    
    cpgend();       /* close all pgplot windows, do this at the very end*/
    #endif  
#endif





/*______________________________________________________________________*/


#include "../../Header_files/debugcode.i"
