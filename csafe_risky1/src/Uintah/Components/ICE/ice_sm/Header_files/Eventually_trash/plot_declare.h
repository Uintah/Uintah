/* void plot_2d_scatter_MM(    double ****x_data_MM,   double ****y_data_MM, 
                            float x_min,            float x_max,         float y_min,
                            float y_max,            int max_len);  */    
                                                 
                            
void plot_dbl_flt(          int index,              double ****dbl_1,
                            double ****dbl_2,       float *flt_1, 
                            float *flt_2);                            


/*______________________________________________________________________
*   plot_common.c
*_______________________________________________________________________*/
void plot_open_window_screen(
        int max_sub_win,        int *n_sub_win,     int *plot_ID);

void plot_open_window_file(
        int max_sub_win,        int *n_sub_win,     char *basename, 
        int filetype);


void plot_generate_axis(    
        char *x_label,          char *y_label,      char *graph_label,
        float *x_min,           float *x_max,       float *y_min,      
        float *y_max,           int *error);
        
void plot_legend(
        float data_max,         float data_min,     float x_max,        
        float y_max);

int plot_scaling(  
        float *data_array,      int max_len,        float *y_min,       float *y_max);

void plot_color_spectrum();

void plot_scaling_CC(      
        int xLoLimit,           int yLoLimit,       int zLoLimit,
        int xHiLimit,           int yHiLimit,       int zHiLimit, 
        double ***data_array,   float *y_min,       float *y_max);

void plot_scaling_CC_MM(   
        int xLoLimit,           int yLoLimit,       int zLoLimit,
        int xHiLimit,           int yHiLimit,       int zHiLimit,
        double ****data_array,  int m,
        float *y_min,           float *y_max);

/*______________________________________________________________________
*   plot_control.c
*_______________________________________________________________________*/
void    plot(    
        float *data,            float *data2,       int max_len,    
        double delX,            double xLo,         double xHi,
        double delY,            double yLo,         double yHi, 
        char *x_label,          char *y_label,     char *graph_label,  
        int plot_type,          int max_sub_win,     
        char *file_basename,     int filetype,       int *plot_ID);
        
/*______________________________________________________________________
*   plot_contour.c
*_______________________________________________________________________*/
    void plot_contour(   
        int xLoLimit,           int xHiLimit,       int yLoLimit,   
        int yHiLimit,           float data_min,     float data_max,   
        float *data );
    
/*______________________________________________________________________
*   plot_vector.c
*_______________________________________________________________________*/
void plot_vector_2D(
        int xLoLimit,           int yLoLimit,       
        int xHiLimit,           int yHiLimit,
        int max_len,            float *data_array1, float *data_array2);
