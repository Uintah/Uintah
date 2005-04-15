




/*______________________________________________________________________
*   C  U  S  T  O  M  I  Z  A  T  I  O  N  S 
*_______________________________________________________________________*/

#if switchInclude_main
       /*__________________________________
        * Define plotting variables 
        *___________________________________*/

        strcpy(file_basename,"");
        outputfile_type     = 0;
        x_axis_origin       = GC_LO(xLoLimit);
        y_axis_origin       = GC_LO(yLoLimit);
        x_axis              = GC_HI(xHiLimit);
        y_axis              = GC_HI(yHiLimit);
        outline_ghostcells  = 0;
        strcpy(x_label,"cell\0");
        strcpy(y_label,"cell\0");   
/*`==========TESTING==========*/ 
/*__________________________________
* Testing
*___________________________________*/        
        /* ______________________________
        *  difference in the velocity field
        *__________________________________*/
        plot_type           = 1;
        Number_sub_plots    = 1;
        strcpy(graph_label,"Main Program vvel_CC_old - vvel_CC\0");                    
        data_array1    = convert_darray_3d_to_vector(
                                vel_difference,
                                GC_LO(xLoLimit),       GC_HI(xHiLimit),  GC_LO(yLoLimit),
                                GC_HI(yHiLimit),       (zLoLimit),       (zHiLimit),
                                &max_len);
        PLOT;

        free_vector_nr(    data_array1,       1, max_len);  
 /*==========TESTING==========`*/      
        
#endif


/*__________________________________
* Include the main differences
*___________________________________*/
#include "../../Header_files/debugcode.i"
