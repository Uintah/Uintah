
/*______________________________________________________________________
*   This file contains all of the debugging and ploting code for all of the
*   functions.
*   main.c
*_______________________________________________________________________*/

#if switchInclude_main_custom
    #if (switchDebug_main_custom == 1)           
       /*__________________________________
        * Define plotting variables 
        *___________________________________*/
        m                   = 1;
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
*___________________________________*/           
        strcpy(graph_label,"Main Program vol_L_CC\0");
        plot_type           = 2;
        Number_sub_plots    = 3;
        data_array1    = convert_darray_4d_to_vector(
                                Vol_L_CC,
                                (xLoLimit),       (xHiLimit),       (yLoLimit),
                                (yHiLimit),       (zLoLimit),       (zHiLimit),
                                m,                      m,
                                &max_len); 
        PLOT;
         free_vector_nr(    data_array1,       1, max_len); 
        /*__________________________________
        *   Variable 2
        *___________________________________*/
        strcpy(graph_label,"Main Program Temp_CC\0");
        plot_type           = 2;
        data_array1    = convert_darray_4d_to_vector(
                                Temp_CC,
                                (xLoLimit),       (xHiLimit),       (yLoLimit),
                                (yHiLimit),       (zLoLimit),       (zHiLimit),
                                m,                      m,
                                &max_len); 
        PLOT;
         free_vector_nr(    data_array1,       1, max_len); 
        /*__________________________________
        *   Variable 3
        *___________________________________*/
        plot_type = 2 ;
        strcpy(graph_label,"Main Program uvel_FC\0");
        data_array1    = convert_darray_6d_to_vector(
                                uvel_FC,
                                (xLoLimit),             (xHiLimit),     (yLoLimit),
                                (yHiLimit),             (zLoLimit),          (zHiLimit),
/**/                            LEFT,                   RIGHT,    
                                m,                      m,
                                &max_len); 
        PLOT;
         free_vector_nr(    data_array1,       1, max_len); 
/*_________________________________
*   MAIN WINDOW 2
*___________________________________*/
        strcpy(graph_label,"Main Program u_Rieman\0");
        plot_type           = 2;
        Number_sub_plots    = 4;
        outline_ghostcells  = 1;        
        /* ______________________________
        *  convert the multidimensional arrays 
        *   into a float vector
        *__________________________________*/ 
        data_array1 = vector_nr(1, X_MAX_LIM); 
        max_len     = xHiLimit;
        for ( i =1 ; i <= xHiLimit; i++)
            data_array1[i] = (float)u_Rieman[i];                  
        PLOT;

        free_vector_nr(    data_array1,       1, X_MAX_LIM);
       /*__________________________________
       *   Variable 2
       *___________________________________*/
        data_array1 = vector_nr(1, X_MAX_LIM);
        strcpy(graph_label,"a_Rieman\0"); 
        for ( i =1 ; i <= xHiLimit; i++)
            data_array1[i] = (float)a_Rieman[i];                  
        PLOT;

        free_vector_nr(    data_array1,       1, X_MAX_LIM);
        /*__________________________________
        *   Variable 3
        *___________________________________*/
        data_array1 = vector_nr(1, X_MAX_LIM);
        strcpy(graph_label,"p_Rieman\0"); 
        for ( i =1 ; i <= xHiLimit; i++)
            data_array1[i] = (float)p_Rieman[i];                  
        PLOT;

        free_vector_nr(    data_array1,       1, X_MAX_LIM);
        
        /*__________________________________
        *   Variable 4
        *___________________________________*/
        data_array1 = vector_nr(1, X_MAX_LIM);
        strcpy(graph_label,"rho_Rieman\0"); 
        for ( i =1 ; i <= xHiLimit; i++)
            data_array1[i] = (float)rho_Rieman[i];                  
        PLOT;
        free_vector_nr(    data_array1,       1, X_MAX_LIM);
         
/*__________________________________
*   MAIN WINDOW 3
*___________________________________*/
        strcpy(graph_label,"Main Program rho_CC\0");
        x_axis_origin       = (xLoLimit);
        y_axis_origin       = (yLoLimit);
        x_axis              = (xHiLimit);
        y_axis              = (yHiLimit); 
        plot_type           = 2;
        Number_sub_plots    = 4;
        outline_ghostcells  = 1;        
        /* ______________________________
        *  convert the multidimensional arrays 
        *   into a float vector
        *__________________________________*/  
        data_array2     = vector_nr(1, X_MAX_LIM); 
        plot_type       = 2 + 2;    
        data_array1     = convert_darray_4d_to_vector(
                                rho_CC,
                                (xLoLimit),             (xHiLimit),         (yLoLimit),
                                (yHiLimit),             (zLoLimit),          (zHiLimit),    
                                m,                      m,
                                &max_len);
                              
                              

        j = 0;               
        for ( i =xLoLimit ; i <= xHiLimit; i++)
        {   j++;
            data_array2[j] = (float)rho_Rieman[j];
        } 
                    
        PLOT;

        free_vector_nr(    data_array2,       1, X_MAX_LIM);
        free_vector_nr(    data_array1,       1, max_len);
        /*__________________________________
        *   Variable 2
        *___________________________________*/
        strcpy(graph_label,"Main Program Press_CC\0");
        data_array2     = vector_nr(1, X_MAX_LIM); 
        plot_type       = 2 + 2;
        data_array1     = convert_darray_4d_to_vector(
                                press_CC,
                                (xLoLimit),             (xHiLimit),         (yLoLimit),
                                (yHiLimit),             (zLoLimit),          (zHiLimit),    
                                m,                      m,
                                &max_len); 
        j = 0;               
        for ( i =xLoLimit ; i <= xHiLimit; i++)
        {   j++;
            data_array2[j] = (float)p_Rieman[j];
        }
        PLOT;
        
        free_vector_nr(    data_array1,       1, max_len);
        free_vector_nr(    data_array2,       1, X_MAX_LIM);
        /*__________________________________
        *   Variable 3
        *___________________________________*/
        strcpy(graph_label,"Main Program uvel_CC\0");
        data_array2     = vector_nr(1, X_MAX_LIM); 
        plot_type       = 2 + 2;
        data_array1     = convert_darray_4d_to_vector(
                                uvel_CC,
                                (xLoLimit),             (xHiLimit),         (yLoLimit),
                                (yHiLimit),             (zLoLimit),          (zHiLimit),    
                                m,                      m,
                                &max_len); 
        j = 0;               
        for ( i =xLoLimit ; i <= xHiLimit; i++)
        {   j++;
            data_array2[j] = (float)u_Rieman[j];
        }
        PLOT;
        
        free_vector_nr(    data_array1,       1, max_len);
        free_vector_nr(    data_array2,       1, X_MAX_LIM);
               
 
        /*__________________________________
        *   Variable 4
        *___________________________________*/         
        strcpy(graph_label,"Main Program Speed_Sound_CC\0");
        data_array2     = vector_nr(1, X_MAX_LIM); 
        plot_type       = 2 + 2;
        data_array1     = convert_darray_4d_to_vector(
                                speedSound,
                                (xLoLimit),             (xHiLimit),         (yLoLimit),
                                (yHiLimit),             (zLoLimit),          (zHiLimit),    
                                m,                      m,
                                &max_len); 
        j = 0;               
        for ( i =xLoLimit ; i <= xHiLimit; i++)
        {   j++;
            data_array2[j] = (float)a_Rieman[j];
        }
        PLOT;
        free_vector_nr(    data_array1,       1, max_len);
        free_vector_nr(    data_array2,       1, X_MAX_LIM);


        cpgend(); 
    #endif  
#endif


/*______________________________________________________________________*/
#include "../../Header_files/debugcode.i"
