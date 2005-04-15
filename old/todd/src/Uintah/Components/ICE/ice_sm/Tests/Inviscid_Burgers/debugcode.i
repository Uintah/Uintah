
/*______________________________________________________________________
*   This file contains all of the debugging and ploting code for all of the
*   functions.
*   main.c
*_______________________________________________________________________*/
#if switchInclude_main_1

        
        /*__________________________________
        * Define plotting variables 
        *___________________________________*/
        plot_type           = 1;
        Number_sub_plots    = 4;
        strcpy(file_basename,"");
        outputfile_type     = 0;
        x_axis_origin       = GC_LO(xLoLimit);
        y_axis_origin       = GC_LO(yLoLimit);
        x_axis              = GC_HI(xHiLimit);
        y_axis              = GC_HI(yHiLimit);
        outline_ghostcells  = 1;
        strcpy(x_label,"cell\0");
        strcpy(y_label,"cell\0");
        strcpy(graph_label,"Main Program INPUTS PRESS_CC\0");
        
        /* ______________________________
        *  convert the multidimensional arrays 
        *   into a float vector
        *__________________________________*/                    
        data_array1    = convert_darray_4d_to_vector(
                                press_CC,
                                GC_LO(xLoLimit),       GC_HI(xHiLimit),           GC_LO(yLoLimit),
                                GC_HI(yHiLimit),       (zLoLimit),                (zHiLimit),
                                m,               m,
                                &max_len); 
        PLOT;

        free_vector_nr(    data_array1,       1, max_len);
/*__________________________________
*   Variable 2
*___________________________________*/         
        strcpy(graph_label,"Main Program INPUTS TEMP_CC\0");
        data_array1    = convert_darray_4d_to_vector(
                                Temp_CC,
                                GC_LO(xLoLimit),       GC_HI(xHiLimit),           GC_LO(yLoLimit),
                                GC_HI(yHiLimit),       (zLoLimit),                (zHiLimit),
                                m,               m,
                                &max_len); 
        PLOT;
         free_vector_nr(    data_array1,       1, max_len);
/*__________________________________
*   Variable 3
*___________________________________*/   
        plot_type = 2;      
        strcpy(graph_label,"Main Program INPUTS uvel_CC\0");
        data_array1    = convert_darray_4d_to_vector(
                                uvel_CC,
                                GC_LO(xLoLimit),       GC_HI(xHiLimit),          (yLoLimit),
                                (yHiLimit),       (zLoLimit),                   (zHiLimit),
                                m,               m,
                                &max_len); 
        PLOT;
         free_vector_nr(    data_array1,       1, max_len); 
         plot_type = 1;
/*__________________________________
*   Variable 4
*___________________________________*/         
        strcpy(graph_label,"Main Program INPUTS vvel_CC\0");
        data_array1    = convert_darray_4d_to_vector(
                                vvel_CC,
                                GC_LO(xLoLimit),       GC_HI(xHiLimit),           GC_LO(yLoLimit),
                                GC_HI(yHiLimit),       (zLoLimit),                 (zHiLimit),
                                m,               m,
                                &max_len); 
        PLOT;
         free_vector_nr(    data_array1,       1, max_len);                              
        /*__________________________________
        *  Clean up the plotting windows
        *___________________________________*/ 
        getchar();
        
       
#endif


/*______________________________________________________________________
*   BOTTOM OF MAIN
*_______________________________________________________________________*/

#if switchInclude_main
    
    #if (switchDebug_main == 2)
        printData_4d(   xLoLimit,       yLoLimit,       zLoLimit,
                        xHiLimit,       yHiLimit,       zHiLimit,
                        m,              m,
                       "Main.c",        "Vol_L_CC",     Vol_L_CC); 
 
         printData_4d(   xLoLimit,      yLoLimit,       zLoLimit,
                        xHiLimit,       yHiLimit,       zHiLimit,
                        m,              m,
                       "Main.c",        "rho_CC",       rho_CC); 
                                             
        printData_4d(   xLoLimit,       yLoLimit,       zLoLimit,
                        xHiLimit,       yHiLimit,       zHiLimit,
                        m,              m,
                       "Main.c",        "Temp_CC",       Temp_CC); 
                       
        printData_4d(   xLoLimit,       yLoLimit,       zLoLimit,
                        xHiLimit,       yHiLimit,       zHiLimit,
                        m,              m,
                       "Main.c",        "uvel_CC",       uvel_CC); 
                       
        printData_4d(   xLoLimit,       yLoLimit,       zLoLimit,
                        xHiLimit,       yHiLimit,       zHiLimit,
                        m,              m,
                       "Main.c",        "vvel_CC",       vvel_CC); 
                       
        printData_4d(   xLoLimit,       yLoLimit,       zLoLimit,
                        xHiLimit,       yHiLimit,       zHiLimit,
                        m,              m,
                       "Main.c",        "xmom_CC",       xmom_CC); 

        printData_4d(   xLoLimit,       yLoLimit,       zLoLimit,
                        xHiLimit,       yHiLimit,       zHiLimit,
                        m,              m,
                       "Main.c",        "ymom_CC",       xmom_CC); 
                         
        printData_4d(   xLoLimit,       yLoLimit,       zLoLimit,
                        xHiLimit,       yHiLimit,       zHiLimit,
                        m,              m,
                       "Main.c",        "delPress_CC",  delPress_CC);   
                       
        printData_4d(   xLoLimit,       yLoLimit,       zLoLimit,
                        xHiLimit,       yHiLimit,       zHiLimit,
                        m,              m,
                       "Main.c",        "Press_CC",  delPress_CC);   
    #endif
    #if (switchDebug_main == 1)           
       /*__________________________________
        * Define plotting variables 
        *___________________________________*/
        plot_type           = 1;
        Number_sub_plots    = 4;
        strcpy(file_basename,"");
        outputfile_type     = 0;
        x_axis_origin       = GC_LO(xLoLimit);
        y_axis_origin       = GC_LO(yLoLimit);
        x_axis              = GC_HI(xHiLimit);
        y_axis              = GC_HI(yHiLimit);  
        outline_ghostcells  = 0;
        strcpy(x_label,"cell\0");
        strcpy(y_label,"cell\0");                       
/*__________________________________
*   MAIN WINDOW 1
*___________________________________*/           
/*         strcpy(graph_label,"Main Program vol_L_CC\0");
        plot_type           = 2;
        Number_sub_plots    = 0;
        data_array1    = convert_darray_4d_to_vector(
                                Vol_L_CC,
                                (xLoLimit),       (xHiLimit),       (yLoLimit),
                                (yHiLimit),       (zLoLimit),       (zHiLimit),
                                m,                      m,
                                &max_len); 
        PLOT;
         free_vector_nr(    data_array1,       1, max_len);  */
         
/*__________________________________
*   MAIN WINDOW 2
*   This makes a postscript file
*___________________________________*/
        outputfile_type     = 2;
        Number_sub_plots    = 4;  
        outline_ghostcells  = 0;        
        strcpy(graph_label,"Main Program rho_CC\0");
        plot_type           = 2;
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
        *   Variable 3
        *___________________________________*/
        strcpy(y_label,"[m/s]\0");
        strcpy(graph_label,"Main Program Speed_of_Sound\0");
        data_array1    = convert_darray_4d_to_vector(
                                speedSound,
                                (xLoLimit),             (xHiLimit),         (yLoLimit),
                                (yHiLimit),             (zLoLimit),          (zHiLimit),    
                                m,                      m,
                                &max_len);
        PLOT;
         free_vector_nr(    data_array1,       1, max_len);
        /*__________________________________
        *   Variable 4
        *___________________________________*/
        plot_type = 2 +2;
        strcpy(y_label,"[m/s]\0");
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
*   MAIN WINDOW 3
*___________________________________*/  
        outputfile_type     = 0;
        Number_sub_plots    = 2;  
        /*__________________________________
        *   Variable 1
        *___________________________________*/
        plot_type = 2 ;
        strcpy(graph_label,"Main Program uvel_FC\0");
        data_array1    = convert_darray_6d_to_vector(
                                uvel_FC,
                                (xLoLimit),             (xHiLimit),         (yLoLimit),
                                (yHiLimit),             (zLoLimit),          (zHiLimit),
                                LEFT,                   RIGHT,    
                                m,                      m,
                                &max_len); 
        PLOT;
         free_vector_nr(    data_array1,       1, max_len);     
        /*__________________________________
        * Variable 2
        *___________________________________*/
        plot_type = 2 + 2;
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
        *  Clean up the plotting windows
        *___________________________________*/ 
        first_pass = 1;

    #endif  
#endif





/*______________________________________________________________________
*            LAGRANGIAN_VOL
*_______________________________________________________________________*/
#if switchInclude_lagrangian_vol
    /*printData_4d(   xLoLimit,       yLoLimit,       zLoLimit,
                    xHiLimit,       yHiLimit,       zHiLimit,
                    m,              m,
                       "lagrangian_vol","Vol_L_CC",       Vol_L_CC); */
                       
        /*__________________________________
        * Define plotting variables 
        *___________________________________*/
        m                   = 1;
        plot_type           = 2;
        Number_sub_plots    = 2;
        strcpy(file_basename,"");
        outputfile_type     = 0;
        outline_ghostcells  = 0;
        x_axis_origin       = (xLoLimit);
        y_axis_origin       = (yLoLimit);
        x_axis              = (xHiLimit);
        y_axis              = (yHiLimit);
        strcpy(x_label,"x\0");
        strcpy(y_label,"y\0");
        strcpy(graph_label,"Inside Lagrangian_vol Vol_L_CC\0");
        
        /* ______________________________
        *  convert the multidimensional arrays 
        *   into a float vector
        *__________________________________*/                    
        data_array1    = convert_darray_4d_to_vector(
                                Vol_L_CC,
                                (xLoLimit),       (xHiLimit),       (yLoLimit),
                                (yHiLimit),       (zLoLimit),       (zHiLimit),
                                m,              m,
                                &max_len);
        PLOT;                               
        /*__________________________________
        *  Clean up the plotting windows
        *___________________________________*/ 
        free_vector_nr(    data_array1,       1, max_len);
        
        /* ______________________________
        *  convert the multidimensional arrays 
        *   into a float vector
        *__________________________________*/ 
        strcpy(graph_label,"Div U Inside Lagrangian_vol\0");                   
        data_array1    = convert_darray_4d_to_vector(
                                plot_1,
                                (xLoLimit),       (xHiLimit),       (yLoLimit),
                                (yHiLimit),       (zLoLimit),       (zHiLimit),
                                m,                 m,
                                &max_len);
        PLOT;                               
        /*__________________________________
        *  Clean up the plotting windows
        *___________________________________*/ 
        free_vector_nr(    data_array1,       1, max_len);
                       
#endif 




/*______________________________________________________________________
*           GRADIENT_LIMITER FUNCTION
*_______________________________________________________________________*/

#if switchInclude_Advect_gradient_limiter
    /*__________________________________
    * Define plotting variables 
    *___________________________________*/
    plot_type           = 1;
    Number_sub_plots    = 0;
    strcpy(file_basename,"");
    outputfile_type     = 0;
    outline_ghostcells  = 1;
    x_axis_origin       = GC_LO(xLoLimit);
    y_axis_origin       = GC_LO(yLoLimit);
    x_axis              = GC_HI(xHiLimit);
    y_axis              = GC_HI(yHiLimit);
    strcpy(x_label,"x\0");
    strcpy(y_label,"y\0");
    strcpy(graph_label,"gradient Limiter\0");   
/* ______________________________
*  convert the multidimensional arrays 
*   into a float vector
*__________________________________*/ 
         data_array1    = convert_darray_3d_to_vector(
                                test,
                                GC_LO(xLoLimit),        GC_HI(xHiLimit),    GC_LO(yLoLimit),
                                GC_HI(yHiLimit),        zLoLimit,           zHiLimit,
                                &max_len);                                                                                  
                        
       PLOT;
       
        free_vector(data_array1, 1, max_len);
#endif 


/*______________________________________________________________________
*           INFLUX_OUTFLUX_VOLUME
*_______________________________________________________________________*/

#if switchInclude_Advect_influx_outflux_volume
    /*__________________________________
    * Define plotting variables 
    *___________________________________*/
    plot_type           = 2;
    Number_sub_plots    = 2;
    strcpy(file_basename,"");
    outputfile_type     = 0;
    x_axis_origin       = (xLoLimit);
    y_axis_origin       = (yLoLimit);
    x_axis              = (xHiLimit);
    y_axis              = (yHiLimit);
    outline_ghostcells  = 0;
    strcpy(x_label,"x\0");
    strcpy(y_label,"y\0");
 
/* ______________________________
*  convert the multidimensional arrays 
*   into a float vector
*__________________________________*/ 
    sprintf(graph_label, "Plot1, influx_outflux_volume, Mat. %d \0",m);                 
     data_array1    = convert_darray_3d_to_vector(
                            plot1,
                            xLoLimit,        xHiLimit,       yLoLimit,
                            yHiLimit,        zLoLimit,       zHiLimit,
                            &max_len);                                   


    PLOT;
    free_vector(data_array1, 1, max_len);
    sprintf(graph_label, "plot2 influx_outflux_volume, Mat. %d \0",m);                 

    data_array1    = convert_darray_3d_to_vector(
                            plot2,
                            xLoLimit,        xHiLimit,       yLoLimit,
                            yHiLimit,        zLoLimit,       zHiLimit,
                            &max_len);     
    PLOT;
    
   free_vector(data_array1, 1, max_len);
#endif



/*______________________________________________________________________
*           ADVECT_Q_OUT_FLUX
*_______________________________________________________________________*/

#if switchInclude_Advect_q_out_flux
    /*__________________________________
    * Define plotting variables 
    *___________________________________*/
    plot_type           = 1;
    Number_sub_plots    = 2;
    strcpy(file_basename,"");
    outputfile_type     = 0;
    x_axis_origin       = GC_LO(xLoLimit);
    y_axis_origin       = GC_LO(yLoLimit);
    x_axis              = GC_HI(xHiLimit);
    y_axis              = GC_HI(yHiLimit);
    outline_ghostcells  = 0;
    strcpy(x_label,"x\0");
    strcpy(y_label,"y\0");
    strcpy(graph_label,"Test inside of q_out_flux\0");   
/* ______________________________
*  convert the multidimensional arrays 
*   into a float vector
*__________________________________*/ 
     data_array1    = convert_darray_3d_to_vector(
                            test,
                            xLoLimit,        xHiLimit,       yLoLimit,
                            yHiLimit,        zLoLimit,       zHiLimit,
                            &max_len);                                   


    PLOT;
    free_vector(data_array1, 1, max_len);
    data_array1    = convert_darray_3d_to_vector(
                            test2,
                            xLoLimit,        xHiLimit,       yLoLimit,
                            yHiLimit,        zLoLimit,       zHiLimit,
                            &max_len);     
                               
    strcpy(graph_label,"Test2 inside of q_out_flux\0"); 
    PLOT;
    
   free_vector(data_array1, 1, max_len);
#endif






/*______________________________________________________________________
*       FIND_Q_VERTEX_MAX_MIN
*_______________________________________________________________________*/
#if switchInclude_find_q_vertex_max
        /*__________________________________
        * Define plotting variables 
        *___________________________________*/
        plot_type           = 2;
        Number_sub_plots    = 4;
        strcpy(file_basename,"");
        outputfile_type     = 0;
        x_axis_origin       = GC_LO(xLoLimit);
        y_axis_origin       = GC_LO(yLoLimit);
        x_axis              = GC_HI(xHiLimit);
        y_axis              = GC_HI(yHiLimit);
        strcpy(x_label,"x\0");
        strcpy(y_label,"y\0");
        strcpy(graph_label,"find_q_vertex_max_min\0");

    for (i = 1; i<= 4; i++)
    { 
        /* ______________________________
        *  convert the multidimensional arrays 
        *   into a float vector
        *__________________________________*/                                 
        data_array1    = convert_darray_4d_to_vector(
                                q_VRTX,
                                xLoLimit,        xHiLimit,       yLoLimit,
                                yHiLimit,        zLoLimit,       zHiLimit,
                                 i,               i,
                                &max_len);                                                              
        x_axis = (double)xHiLimit * delX;                        
 
        PLOT;                               

     }
    
    /*__________________________________
    *   plot the max vertex values
    *___________________________________*/
    outputfile_type     = 0;
    Number_sub_plots    = 2;
    strcpy(file_basename,"qmax");
    
    data_array1    = convert_darray_3d_to_vector(
                                q_VRTX_MAX,
                                GC_LO(xLoLimit),        GC_HI(xHiLimit),    GC_LO(yLoLimit),
                                GC_HI(yHiLimit),        zLoLimit,           zHiLimit,
                                &max_len);                                                              
    x_axis = (double)xHiLimit * delX;                        

    PLOT;                               

    data_array1    = convert_darray_3d_to_vector(
                                q_VRTX_MIN,
                                GC_LO(xLoLimit),        GC_HI(xHiLimit),    GC_LO(yLoLimit),
                                GC_HI(yHiLimit),        zLoLimit,           zHiLimit,
                                &max_len);                                                              
    x_axis = (double)xHiLimit * delX;                        

    PLOT;                               
   
    free_vector(data_array1, 1, max_len);  
#endif



/*______________________________________________________________________
*           ADVECT_Q
*_______________________________________________________________________*/
#if switchInclude_Advect_q 
 /*    advect_verify_conservation(    
                            xLoLimit,       yLoLimit,       zLoLimit,
                            xHiLimit,       yHiLimit,       zHiLimit,
                            q_CC,           m); */
/*     printData_3d(       
                            (xLoLimit),    (yLoLimit),       zLoLimit,
                            (xHiLimit),    (yHiLimit),       zHiLimit,
                            "advect_q",         "sum_q_influx",           test); */
    /*__________________________________
    * Define plotting variables 
    *___________________________________*/
    plot_type           = 2;
    Number_sub_plots    = 3;
    strcpy(file_basename,"");
    outputfile_type     = 0;
    outline_ghostcells  = 1;
    x_axis_origin       = GC_LO(xLoLimit);
    y_axis_origin       = GC_LO(yLoLimit);
    x_axis              = GC_HI(xHiLimit);
    y_axis              = GC_HI(yHiLimit);
    strcpy(x_label,"x\0");
    strcpy(y_label,"y\0");
    strcpy(graph_label,"Inside of advect_q sum_q_outflux WITH GHOSTCELLS\0");  
    /* ______________________________
    *  Variable 1
    *__________________________________*/ 
     data_array1    = convert_darray_3d_to_vector(
                            plot1,
                            GC_LO(xLoLimit),    GC_HI(xHiLimit),    (yLoLimit),
                            (yHiLimit),         (zLoLimit),         (zHiLimit),
                            &max_len);                                   
    PLOT;
    free_vector(data_array1, 1, max_len);
    /* ______________________________
    *  Variable 2
    *__________________________________*/ 
    strcpy(graph_label,"Inside of advect_q sum_q_influx WITH GHOSTCELLS\0");
    data_array1    = convert_darray_3d_to_vector(
                            plot2,
                            GC_LO(xLoLimit),        GC_HI(xHiLimit),    (yLoLimit),
                            (yHiLimit),             (zLoLimit),         (zHiLimit),
                            &max_len);        

    /*__________________________________
    *   Use the PLOT macro to do the plotting
    *___________________________________*/
    PLOT;
    free_vector(data_array1, 1, max_len);
    
    /* ______________________________
    *  Variable 3
    *__________________________________*/ 
    strcpy(graph_label,"Inside of advect_q  WITH GHOSTCELLS\0");
    data_array1    = convert_darray_3d_to_vector(
                            plot3,
                            GC_LO(xLoLimit),        GC_HI(xHiLimit),     (yLoLimit),
                            (yHiLimit),             (zLoLimit),          (zHiLimit),
                            &max_len);        

    PLOT;
    free_vector(data_array1, 1, max_len);
/*     fprintf(stderr, "press return to continue");
    getchar(); */
#endif




/*______________________________________________________________________
*           ADVECT_AND_ADVANCE_IN_TIME
*_______________________________________________________________________*/
#if switchInclude_advect_and_advance_in_time
    #if (switchDebug_advect_and_advance_in_time == 1)
        printData_4d(       GC_LO(xLoLimit),    GC_LO(yLoLimit),       zLoLimit,
                            GC_HI(xHiLimit),    GC_HI(yHiLimit),       zHiLimit,
                            m,           m,
                            "advect_and_advance_in_time","rho_CC",    rho_CC);
        printData_4d(       GC_LO(xLoLimit),    GC_LO(yLoLimit),       zLoLimit,
                            GC_HI(xHiLimit),    GC_HI(yHiLimit),       zHiLimit,
                            m,           m,
                            "advect_and_advance_in_time","ymom_CC",    ymom_CC);

        printData_4d(       GC_LO(xLoLimit),    GC_LO(yLoLimit),       zLoLimit,
                            GC_HI(xHiLimit),    GC_HI(yHiLimit),       zHiLimit,
                            m,           m,
                            "advect_and_advance_in_time","advct_ymom_CC",    advct_ymom_CC);

        printData_4d(       GC_LO(xLoLimit),    GC_LO(yLoLimit),       zLoLimit,
                            GC_HI(xHiLimit),    GC_HI(yHiLimit),       zHiLimit,
                            m,            m,
                            "advect_and_advance_in_time","yvel_CC",      vvel_CC);                
    #endif
    #if (switchDebug_advect_and_advance_in_time == 2)  
        /*__________________________________
        * Define plotting variables 
        *___________________________________*/
        m                   = 1;
        plot_type           = 2;
        Number_sub_plots    = 2;
        strcpy(file_basename,"");
        outputfile_type     = 0;
        x_axis_origin       = (xLoLimit);
        y_axis_origin       = (yLoLimit);
        x_axis              = (xHiLimit);
        y_axis              = (yHiLimit);
        outline_ghostcells  = 1;
        strcpy(x_label,"cell\0");
        strcpy(y_label,"cell\0");
        strcpy(graph_label,"advect_and_advance_in_time, advct_rho_CC\0");
        /*__________________________________
        *   Variable 1
        *___________________________________*/                   
        data_array1    = convert_darray_4d_to_vector(
                                advct_rho_CC,
                                (xLoLimit),       (xHiLimit),                       (yLoLimit),
                                (yHiLimit),       (zLoLimit),                       (zHiLimit),
                                m,                      m,                          &max_len);
        PLOT;
        free_vector_nr(    data_array1,       1, max_len);
        /*__________________________________
        *   Variable 2
        *___________________________________*/   
    #if 0      
        strcpy(graph_label,"ymom_CC\0");
        data_array1    = convert_darray_4d_to_vector(
                                ymom_CC,
                                GC_LO(xLoLimit),       GC_HI(xHiLimit),             GC_LO(yLoLimit),
                                GC_HI(yHiLimit),       (zLoLimit),                  (zHiLimit),
                                m,                      m,                          &max_len);
        PLOT;
        free_vector_nr(    data_array1,       1, max_len);
    #endif
        /*__________________________________
        *   Variable 3
        *___________________________________*/         
        strcpy(graph_label,"advct_xmom_CC\0");
        data_array1    = convert_darray_4d_to_vector(
                                advct_xmom_CC,
                                (xLoLimit),             (xHiLimit),                 (yLoLimit),
                                (yHiLimit),             (zLoLimit),                 (zHiLimit),
                                m,                      m,                          &max_len);
        PLOT;
        free_vector_nr(    data_array1,       1, max_len); 
        /*__________________________________
        *   Variable 4
        *___________________________________*/
    #if 0        
        strcpy(graph_label,"advct_int_eng_CC\0");
        data_array1    = convert_darray_4d_to_vector(
                                advct_int_eng_CC,
                                (xLoLimit),             (xHiLimit),                 (yLoLimit),
                                (yHiLimit),             (zLoLimit),                 (zHiLimit),
                                m,                      m,                          &max_len);
        PLOT;
        free_vector_nr(    data_array1,       1, max_len);
    #endif
    #endif
#endif

/*______________________________________________________________________
*           PRESSURE_ITERATION
*_______________________________________________________________________*/
#if switchInclude_pressure_interation

    printData_6d(       xLoLimit,           yLoLimit,       zLoLimit,
                        xHiLimit,           yHiLimit,       zHiLimit,
                        RIGHT,              LEFT,
                        m,                  m,
                       "pressure_interation",     
                       "Uvel_FC",           uvel_FC,        0);

                        

    
    printData_6d(       xLoLimit,           yLoLimit,       zLoLimit,
                        xHiLimit,           yHiLimit,       zHiLimit,
                        TOP,                BOTTOM,
                        m,                  m,
                       "pressure_interation",    
                       "Vvel_FC",           vvel_FC,        0);
                       
    printData_4d(       xLoLimit,           yLoLimit,       zLoLimit,
                        xHiLimit,           yHiLimit,       zHiLimit,
                        m,                  m,
                       "pressure_interation",     
                       "Press_CC",          press_CC);
                       
    printData_4d(       xLoLimit,           yLoLimit,       zLoLimit,
                        xHiLimit,           yHiLimit,       zHiLimit,
                        m,                  m,
                       "pressure_interation",     
                       "delPress_CC",          delPress_CC);
                              
        /*__________________________________
        * Define plotting variables 
        *___________________________________*/
        plot_type           = 1;
        Number_sub_plots    = 2;
        strcpy(file_basename,"");
        outputfile_type     = 0;
        x_axis_origin       = GC_LO(xLoLimit);
        y_axis_origin       = GC_LO(yLoLimit);
        x_axis              = GC_HI(xHiLimit);
        y_axis              = GC_HI(yHiLimit);
        outline_ghostcells  = 1;
        strcpy(x_label,"cell\0");
        strcpy(y_label,"cell\0");

/*__________________________________
*   Variable 1
*___________________________________*/   
        strcpy(graph_label,"pressure_iteration, press_CC\0");                
        data_array1    = convert_darray_4d_to_vector(
                                press_CC,
                                GC_LO(xLoLimit),       GC_HI(xHiLimit),     GC_LO(yLoLimit),
                                GC_HI(yHiLimit),       (zLoLimit),          (zHiLimit),
                                m,                      m,
                                &max_len);
        PLOT;
        free_vector_nr(    data_array1,       1, max_len);
/*__________________________________
*   Variable 2
*___________________________________*/
        strcpy(graph_label,"pressure_iteration, del_press_CC\0");
        data_array1    = convert_darray_4d_to_vector(
                                delPress_CC,
                                GC_LO(xLoLimit),       GC_HI(xHiLimit),     GC_LO(yLoLimit),
                                GC_HI(yHiLimit),       (zLoLimit),          (zHiLimit),
                                m,                      m,
                                &max_len);
        PLOT;
        free_vector_nr(    data_array1,       1, max_len);
        fprintf(stderr,"press return to continue\n");
        getchar();
/*__________________________________
*   Variable 3
*___________________________________*/

        Number_sub_plots    = 2;
        strcpy(graph_label,"pressure_iteration, uvel_FC\0");
        plot_face_centered_data(
                                GC_LO(xLoLimit),        GC_HI(xHiLimit),    GC_LO(yLoLimit),
                                GC_HI(yHiLimit),        (zLoLimit),         (zHiLimit),
                                delX,                   delY,               uvel_FC,
                                x_label,                y_label,            graph_label,
                                outline_ghostcells,     Number_sub_plots,    
                                file_basename,          outputfile_type);
/*__________________________________
*   Variable 4
*___________________________________*/
        strcpy(graph_label,"pressure_iteration, vvel_FC\0");
        plot_face_centered_data(
                                GC_LO(xLoLimit),        GC_HI(xHiLimit),    GC_LO(yLoLimit),
                                GC_HI(yHiLimit),        (zLoLimit),         (zHiLimit),
                                delX,                   delY,               vvel_FC,
                                x_label,                y_label,            graph_label,
                                outline_ghostcells,     Number_sub_plots,    
                                file_basename,          outputfile_type);
        putenv("PGPLOT_ITERATING_NOW=0");
        fprintf(stderr,"press return to continue\n");
        getchar();
        
                              
#endif
/*______________________________________________________________________
*       ITERATIVE PRESSURE SOLVER (VEL_INITIAL_ITERATION)
*______________________________________________________________________*/

#if switchInclude_vel_initial_iteration 
   
    printData_4d(       GC_LO(xLoLimit),       GC_LO(yLoLimit),      zLoLimit,
                        GC_HI(xHiLimit),       GC_HI(yHiLimit),       zHiLimit,
                        m,              m,
                        "Pressure vel_initial_iteration",     
                        "Press_CC",      press_CC);       
    
    printData_6d(       xLo,       yLo,      zLoLimit,
                        xHi,       yHi,       zHiLimit,
                        RIGHT,          LEFT,
                        m,              m,
                        "Pressure vel_initial_iteration",     
                        "Uvel_FC",      uvel_FC,        0);
    
    printData_6d(       xLo,       yLo,       zLoLimit,
                        xHi,       yHi,       zHiLimit,
                        TOP,            BOTTOM,
                        m,              m,
                        "Pressure vel_initial_iteration",     
                        "Vvel_FC",  vvel_FC,            0);
                        
#endif


/*______________________________________________________________________
*           ACCUMULATE_MOMENTUM_SOURCES_SINKS
*_______________________________________________________________________*/
#if switchInclude_accumulate_momentum_source_sinks
                              
        /*__________________________________
        * Define plotting variables 
        *___________________________________*/
        plot_type           = 1;
        Number_sub_plots    = 1;
        strcpy(file_basename,"");
        outputfile_type     = 0;
        x_axis_origin       = GC_LO(xLoLimit);
        y_axis_origin       = GC_LO(yLoLimit);
        x_axis              = GC_HI(xHiLimit);
        y_axis              = GC_HI(yHiLimit);
        outline_ghostcells  = 1;
        strcpy(x_label,"cell\0");
        strcpy(y_label,"cell\0");
 
/*__________________________________
*   Variable 1
*___________________________________*/   
        strcpy(graph_label,"accumulate_momentum_source_sinks, xmom_source_CC\0");                
        data_array1    = convert_darray_4d_to_vector(
                                xmom_source,
                                GC_LO(xLoLimit),       GC_HI(xHiLimit),     GC_LO(yLoLimit),
                                GC_HI(yHiLimit),       (zLoLimit),          (zHiLimit),
                                m,                      m,
                                &max_len);
        PLOT;
        free_vector_nr(    data_array1,       1, max_len);
/*__________________________________
*   Variable 2
*___________________________________*/
/*         strcpy(graph_label,"pressure_iteration, del_press_CC\0");
        data_array1    = convert_darray_3d_to_vector(
                                delPress_CC,
                                GC_LO(xLoLimit),       GC_HI(xHiLimit),     GC_LO(yLoLimit),
                                GC_HI(yHiLimit),       (zLoLimit),          (zHiLimit),
                                &max_len);
        PLOT;
        free_vector_nr(    data_array1,       1, max_len);
        
        getchar(); */
        fprintf(stderr,"press return to continue \n");
                              
#endif





/*______________________________________________________________________
*           EQUATION_OF_STATE
*_______________________________________________________________________*/
#if switchInclude_equation_of_state
                              
        /*__________________________________
        * Define plotting variables 
        *___________________________________*/
        plot_type           = 1;
        Number_sub_plots    = 1;
        strcpy(file_basename,"");
        outputfile_type     = 0;
        x_axis_origin       = GC_LO(xLoLimit);
        y_axis_origin       = GC_LO(yLoLimit);
        x_axis              = GC_HI(xHiLimit);
        y_axis              = GC_HI(yHiLimit);
        outline_ghostcells  = 1;
        strcpy(x_label,"cell\0");
        strcpy(y_label,"cell\0");
 
/*__________________________________
*   Variable 1
*___________________________________*/   
        strcpy(graph_label,"equation_of_state, Press_CC\0");                
        data_array1    = convert_darray_4d_to_vector(
                                press_CC,
                                GC_LO(xLoLimit),       GC_HI(xHiLimit),     GC_LO(yLoLimit),
                                GC_HI(yHiLimit),       (zLoLimit),          (zHiLimit),
                                m,                      m,
                                &max_len);
        PLOT;
        free_vector_nr(    data_array1,       1, max_len);
/*__________________________________
*   Variable 2
*   CURRENTLY NOT USED
*___________________________________*/
/*         strcpy(graph_label,"pressure_iteration, del_press_CC\0");
        data_array1    = convert_darray_3d_to_vector(
                                delPress_CC,
                                GC_LO(xLoLimit),       GC_HI(xHiLimit),     GC_LO(yLoLimit),
                                GC_HI(yHiLimit),       (zLoLimit),          (zHiLimit),
                                &max_len);
        PLOT;
        free_vector_nr(    data_array1,       1, max_len);
        
        getchar(); */
        fprintf(stderr,"press return to continue \n");
                              
#endif
/*______________________________________________________________________
*           Interpolate_vel_CC_to_FC
*_______________________________________________________________________*/
#if switchInclude_compute_face_centered_velocities
    stay_or_go = *getenv("PGPLOT_PLOTTING_ON_OFF");
    if (stay_or_go == '1')
    {                              
        /*__________________________________
        * Define plotting variables 
        *___________________________________*/
        plot_type           = 1;
        Number_sub_plots    = 4;
        strcpy(file_basename,"");
        outputfile_type     = 0;
        x_axis_origin       = GC_LO(xLoLimit);
        y_axis_origin       = GC_LO(yLoLimit);
        x_axis              = GC_HI(xHiLimit);
        y_axis              = GC_HI(yHiLimit);
        outline_ghostcells  = 1;
        strcpy(x_label,"cell\0");
        strcpy(y_label,"cell\0");
/*__________________________________
*   Variable 1
*___________________________________*/   
        strcpy(graph_label,"compute_face_centered_velocities, rho_CC\0");                
        data_array1    = convert_darray_4d_to_vector(
                                rho_CC,
                                GC_LO(xLoLimit),       GC_HI(xHiLimit),     GC_LO(yLoLimit),
                                GC_HI(yHiLimit),       (zLoLimit),          (zHiLimit),
                                m,                      m,
                                &max_len);
        PLOT;
        free_vector_nr(    data_array1,       1, max_len);
        
        strcpy(graph_label,"uvel_CC\0");                
        data_array1    = convert_darray_4d_to_vector(
                                uvel_CC,
                                GC_LO(xLoLimit),       GC_HI(xHiLimit),     GC_LO(yLoLimit),
                                GC_HI(yHiLimit),       (zLoLimit),          (zHiLimit),
                                m,                      m,
                                &max_len);
        PLOT;
        free_vector_nr(    data_array1,       1, max_len);
        
        strcpy(graph_label,"vvel_CC\0");                
        data_array1    = convert_darray_4d_to_vector(
                                vvel_CC,
                                GC_LO(xLoLimit),       GC_HI(xHiLimit),     GC_LO(yLoLimit),
                                GC_HI(yHiLimit),       (zLoLimit),          (zHiLimit),
                                m,                      m,
                                &max_len);
        PLOT;
        free_vector_nr(    data_array1,       1, max_len);
        
        strcpy(graph_label,"wvel_CC\0");                
        data_array1    = convert_darray_4d_to_vector(
                                wvel_CC,
                                GC_LO(xLoLimit),       GC_HI(xHiLimit),     GC_LO(yLoLimit),
                                GC_HI(yHiLimit),       (zLoLimit),          (zHiLimit),
                                m,                      m,
                                &max_len);
        PLOT;
        free_vector_nr(    data_array1,       1, max_len);
        
        if(first_pass == 0)
        {
            fprintf(stderr, "Press return to continue\n");
            getchar();
        }        
/*__________________________________
*   Face centered velocities
*___________________________________*/        
        Number_sub_plots    = 2;
        strcpy(graph_label,"compute_face_centered_velocities, uvel_FC\0");
        plot_face_centered_data(
                                GC_LO(xLoLimit),        GC_HI(xHiLimit),    GC_LO(yLoLimit),
                                GC_HI(yHiLimit),        (zLoLimit),         (zHiLimit),
                                delX,                   delY,               uvel_FC,
                                x_label,                y_label,            graph_label,
                                outline_ghostcells,     Number_sub_plots,    
                                file_basename,          outputfile_type);
        strcpy(graph_label,"compute_face_centered_velocities, vvel_FC\0");
        plot_face_centered_data(
                                GC_LO(xLoLimit),        GC_HI(xHiLimit),    GC_LO(yLoLimit),
                                GC_HI(yHiLimit),        (zLoLimit),         (zHiLimit),
                                delX,                   delY,               vvel_FC,
                                x_label,                y_label,            graph_label,
                                outline_ghostcells,     Number_sub_plots,    
                                file_basename,          outputfile_type); 
        if(first_pass == 0)
        {
            fprintf(stderr, "Press return to continue\n");
            getchar();
            first_pass =1;
        }
        
    }        
#endif

/*______________________________________________________________________
*   Pressure_PCGMG
*_______________________________________________________________________*/
#if switchInclude_pressure_PCG

/*__________________________________
*   Now extract the values
*   from PETSC
*___________________________________*/
    mat = 1;
    ierr = VecGetArray(userctx.b,           &array_1);                                      CHKERRA(ierr);
    ierr = VecGetArray(userctx.stencil.an,  &array_2);                                      CHKERRA(ierr);
    ierr = VecGetArray(userctx.stencil.as,  &array_3);                                      CHKERRA(ierr);
    ierr = VecGetArray(userctx.stencil.ae,  &array_4);                                      CHKERRA(ierr);
    ierr = VecGetArray(userctx.stencil.aw,  &array_5);                                      CHKERRA(ierr);
    ierr = VecGetArray(userctx.stencil.ap,  &array_6);                                      CHKERRA(ierr);
    
    for ( k = zLoLimit; k <= zHiLimit; k++)
    {
        for ( j = yLoLimit; j <= yHiLimit; j++)
        {
            for ( i = xLoLimit; i <= xHiLimit; i++)
            { 
            /*__________________________________
            * map a 3d array to a 1d vector
            *___________________________________*/
            index = (i-xLoLimit) + (j-yLoLimit)*(xHiLimit-xLoLimit+1);                  /* 2D       */
            index = index + (k - zLoLimit)*(xHiLimit-xLoLimit+1)*(yHiLimit-yLoLimit+1); /* 3D       */ 
            plot_1[i][j][k]   = array_1[index];
            plot_2[i][j][k]   = array_2[index];
            plot_3[i][j][k]   = array_3[index];
            plot_4[i][j][k]   = array_4[index];
            plot_5[i][j][k]   = array_5[index];
            plot_6[i][j][k]   = array_6[index];
            }
        }
    }
    ierr = VecRestoreArray(userctx.b,           &array_1);                                  CHKERRA(ierr);
    ierr = VecRestoreArray(userctx.stencil.an,  &array_2);                                  CHKERRA(ierr);
    ierr = VecRestoreArray(userctx.stencil.as,  &array_3);                                  CHKERRA(ierr);
    ierr = VecRestoreArray(userctx.stencil.ae,  &array_4);                                  CHKERRA(ierr);
    ierr = VecRestoreArray(userctx.stencil.aw,  &array_5);                                  CHKERRA(ierr);
    ierr = VecRestoreArray(userctx.stencil.ap,  &array_6);                                  CHKERRA(ierr);

    
    printData_4d(       
                            (xLoLimit),    (yLoLimit),       zLoLimit,
                            (xHiLimit),    (yHiLimit),       zHiLimit,
                            mat,            mat,
                            "PCGMG","delPress_CC",       delPress_CC);
    printData_3d(       
                            (xLoLimit),    (yLoLimit),       zLoLimit,
                            (xHiLimit),    (yHiLimit),       zHiLimit,
                            "PCGMG","Divergence of face centered vel.",       plot_1);
    printData_3d(       
                            (xLoLimit),    (yLoLimit),       zLoLimit,
                            (xHiLimit),    (yHiLimit),       zHiLimit,
                            "PCGMG","an",       plot_2);
    printData_3d(       
                            (xLoLimit),    (yLoLimit),       zLoLimit,
                            (xHiLimit),    (yHiLimit),       zHiLimit,
                            "PCGMG","as",       plot_3);
                            
    printData_3d(       
                            (xLoLimit),    (yLoLimit),       zLoLimit,
                            (xHiLimit),    (yHiLimit),       zHiLimit,
                            "PCGMG","ae",       plot_4);
    printData_3d(       
                            (xLoLimit),    (yLoLimit),       zLoLimit,
                            (xHiLimit),    (yHiLimit),       zHiLimit,
                            "PCGMG","aw",       plot_5);


                              
    /*__________________________________
    * Define plotting variables 
    *___________________________________*/
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
*  Window 1
*___________________________________*/   
        strcpy(graph_label,"PCGMG,NoGhostCells delPress_CC\0");                
        data_array1    = convert_darray_4d_to_vector(
                                delPress_CC,
                                (xLoLimit),       (xHiLimit),         (yLoLimit),
                                (yHiLimit),       (zLoLimit),         (zHiLimit),
                                mat,                      mat,
                                &max_len);
        PLOT;
        free_vector_nr(    data_array1,       1, max_len);
 
        strcpy(graph_label,"Divergence of (*)vel_GC\0");                
        data_array1    = convert_darray_3d_to_vector(
                                plot_1,
                                (xLoLimit),       (xHiLimit),          (yLoLimit),
                                (yHiLimit),       (zLoLimit),          (zHiLimit),
                                &max_len);
        PLOT;
        free_vector_nr(    data_array1,       1, max_len);
        
        strcpy(graph_label,"stencil an\0");         
        data_array1    = convert_darray_3d_to_vector(
                                plot_2,
                                (xLoLimit),       (xHiLimit),          (yLoLimit),
                                (yHiLimit),       (zLoLimit),          (zHiLimit),
                                &max_len);
        PLOT;
        free_vector_nr(    data_array1,       1, max_len);  
        
        strcpy(graph_label,"stencil as\0");         
        data_array1    = convert_darray_3d_to_vector(
                                plot_3,
                                (xLoLimit),       (xHiLimit),          (yLoLimit),
                                (yHiLimit),       (zLoLimit),          (zHiLimit),
                                &max_len);
        PLOT;
        free_vector_nr(    data_array1,       1, max_len); 
        fprintf(stderr,"press return to continue \n");
        getchar(); 
/*__________________________________
*  Window 2
*___________________________________*/
        Number_sub_plots    = 3;   
        strcpy(graph_label,"stencil ae\0");                
        data_array1    = convert_darray_3d_to_vector(
                                plot_4,
                                (xLoLimit),       (xHiLimit),          (yLoLimit),
                                (yHiLimit),       (zLoLimit),          (zHiLimit),
                                &max_len);
        PLOT;
        free_vector_nr(    data_array1,       1, max_len);
 
        strcpy(graph_label,"stencil aw\0");                
        data_array1    = convert_darray_3d_to_vector(
                                plot_5,
                                (xLoLimit),       (xHiLimit),          (yLoLimit),
                                (yHiLimit),       (zLoLimit),          (zHiLimit),
                                &max_len);
        PLOT;
        free_vector_nr(    data_array1,       1, max_len);
        
        strcpy(graph_label,"stencil ap\0");         
        data_array1    = convert_darray_3d_to_vector(
                                plot_6,
                                (xLoLimit),       (xHiLimit),          (yLoLimit),
                                (yHiLimit),       (zLoLimit),          (zHiLimit),
                                &max_len);
        PLOT;
        free_vector_nr(    data_array1,       1, max_len);          
              
        fprintf(stderr,"press return to continue \n");
        getchar();
       
#endif
/*______________________________________________________________________
*           PRESSURE_RESIDUAL
*_______________________________________________________________________*/
#if switchInclude_press_eq_residual
    #if (switchDebug_press_eq_residual == 1)
    
    printData_3d(       
                            (xLoLimit),    (yLoLimit),       zLoLimit,
                            (xHiLimit),    (yHiLimit),       zHiLimit,
                            "press_residual","div_vel_FC",   plot_1);
    printData_3d(       
                            (xLoLimit),    (yLoLimit),       zLoLimit,
                            (xHiLimit),    (yHiLimit),       zHiLimit,
                            "press_residual","delPress_CC", plot_2);
    printData_3d(       
                            (xLoLimit),    (yLoLimit),       zLoLimit,
                            (xHiLimit),    (yHiLimit),       zHiLimit,
                            "press_residual","residual",    plot_3);
    fprintf(stderr,"press return to continue \n");
    getchar();
#endif  
#if (switchDebug_press_eq_residual == 2)                                    
        /*__________________________________
        * Define plotting variables 
        *___________________________________*/
        plot_type           = 2;
        Number_sub_plots    = 2;
        strcpy(file_basename,"");
        outputfile_type     = 0;
        x_axis_origin       = GC_LO(xLoLimit);
        y_axis_origin       = GC_LO(yLoLimit);
        x_axis              = (xHiLimit);
        y_axis              = (yHiLimit);
        outline_ghostcells  = 0;
        strcpy(x_label,"cell\0");
        strcpy(y_label,"cell\0");
 
/*__________________________________
*   Variable 1
*___________________________________*/   
        strcpy(graph_label,"press_eq_residual, div_vel_FC\0");                
        data_array1    = convert_darray_3d_to_vector(
                                plot_1,
                                GC_LO(xLoLimit),  GC_HI(xHiLimit),      (yLoLimit),
                                (yHiLimit),       (zLoLimit),           (zHiLimit),
                                &max_len);
        PLOT;
        free_vector_nr(    data_array1,       1, max_len);
/*__________________________________
*   Variable 2
*___________________________________*/   
        strcpy(graph_label,"delPress_CC\0");                
        data_array1    = convert_darray_3d_to_vector(
                                plot_2,
                                GC_LO(xLoLimit),  GC_HI(xHiLimit),      (yLoLimit),
                                (yHiLimit),       (zLoLimit),           (zHiLimit),
                                &max_len);
        PLOT;
        free_vector_nr(    data_array1,       1, max_len);
/*__________________________________
*   Variable 3
*___________________________________*/   
/*        strcpy(graph_label,"fabs(delPress_dt + div_vel_FC)\0");                
        data_array1    = convert_darray_3d_to_vector(
                                plot_3,
                                (xLoLimit),       (xHiLimit),           (yLoLimit),
                                (yHiLimit),       (zLoLimit),           (zHiLimit),
                                &max_len);
        PLOT;
        free_vector_nr(    data_array1,       1, max_len); */
        
    #endif                              
#endif



/*______________________________________________________________________
*   SHEAR_STRESS.C
*_______________________________________________________________________*/
#if switchInclude_shear_stress_Xdir

        /*__________________________________
        * Define plotting variables 
        *___________________________________*/
        Number_sub_plots    = 0;
        strcpy(file_basename,"");
        outputfile_type     = 0;
        outline_ghostcells  = 0;
        strcpy(x_label,"x\0");
        strcpy(y_label,"y\0"); 
/*__________________________________
*   Face centered variables
*___________________________________*/        
        strcpy(graph_label,"shear_stress_Xdir, tau_XX and tau_YX\0");
        plot_face_centered_data(
                                GC_LO(xLoLimit),        GC_HI(xHiLimit),    GC_LO(yLoLimit),
                                GC_HI(yHiLimit),        (zLoLimit),         (zHiLimit),
                                delX,                   delY,               tau_X_FC,
                                x_label,                y_label,            graph_label,
                                outline_ghostcells,     Number_sub_plots,    
                                file_basename,          outputfile_type);
        /*__________________________________
        * QUITE FULLWARN COMMENTS       
        *___________________________________*/
        plot_type = plot_type;      x_axis_origin = x_axis_origin;       y_axis_origin = y_axis_origin;
        x_axis = x_axis;            y_axis = y_axis;                    max_len = max_len;
        data_array1 = data_array1;
        if(first_pass == 0)
        {
            fprintf(stderr, "Press return to continue\n");
            getchar();
        }                        
#endif
/*__________________________________
*   SHEAR_STRESS_YDIR
*___________________________________*/
#if switchInclude_shear_stress_Ydir

        /*__________________________________
        * Define plotting variables 
        *___________________________________*/
        Number_sub_plots    = 0;
        strcpy(file_basename,"");
        outputfile_type     = 0;
        outline_ghostcells  = 0;
        strcpy(x_label,"x\0");
        strcpy(y_label,"y\0"); 
/*__________________________________
*   Face centered variables
*___________________________________*/        
        strcpy(graph_label,"shear_stress_Ydir, tau_XY and tau_XY\0");
        plot_face_centered_data(
                                GC_LO(xLoLimit),        GC_HI(xHiLimit),    GC_LO(yLoLimit),
                                GC_HI(yHiLimit),        (zLoLimit),         (zHiLimit),
                                delX,                   delY,               tau_Y_FC,
                                x_label,                y_label,            graph_label,
                                outline_ghostcells,     Number_sub_plots,    
                                file_basename,          outputfile_type);
        /*__________________________________
        * QUITE FULLWARN COMMENTS       
        *___________________________________*/
        plot_type = plot_type;      x_axis_origin = x_axis_origin;       y_axis_origin = y_axis_origin;
        x_axis = x_axis;            y_axis = y_axis;                    max_len = max_len;
        data_array1 = data_array1;

        if(first_pass == 0)
        {
            fprintf(stderr, "Press return to continue\n");
            getchar();
        } 
                               
#endif
