#ifndef __FUNCTIONDECLARE_H
#define __FUNCTIONDECLARE_H

#include <stdio.h>
#include <time.h>
#ifdef IMPLICIT
#include "pcgmg.h"
#endif


#ifdef __cplusplus
extern "C" {
#endif

/*______________________________________________________________________
*   INLINE.H
*_______________________________________________________________________*/
/* void find_transport_property_FC(
                    int i,                  int j,                      int k,
                    int m,                  double ****data_CC,         double *data_FC);  */         
/*______________________________________________________________________
*      I  N  P  U  T  .  C 
*_______________________________________________________________________*/
double  readdouble(  FILE *fp,              char var_name[],            int printSwitch);
float   readfloat(   FILE *fp,              char var_name[],            int printSwitch);
int     readint(     FILE *fp,              char var_name[],            int printSwitch);
void    readstring(  FILE *fp,char output[],char var_name[],            int printSwitch);    
                                            
void readInputFile( int *xLoLimit,          int *yLoLimit,              int *zLoLimit,
                    int *xHiLimit,          int *yHiLimit,              int *zHiLimit, 
                    double *delX,           double *delY,               double *delZ,
                    double ****uvel_CC,     double ****vvel_CC,         double ****wvel_CC, 
                    double ****Temp_CC,     double ****press_cc,        double ****rho_CC,
                    double ****scalar1_CC,  double ****scalar2_CC,      double ****scalar3_CC,
                    double ****viscosity_CC,double ****thermalCond_CC,  double ****cv_CC, 
                    double *R,              double *gamma,
                    double *t_final,        double *t_output_vars,      double *delt_limits, 
                    char output_file_basename[],                        char output_file_desc[],       
                    double *grav,           double ****speedSound,
                    int **BC_inputs,        double ***BC_Values,        double *CFL,
                    int *nMaterials);                                            
          
void testInputFile( int xLoLimit,           int yLoLimit,               int zLoLimit,
                    int xHiLimit,           int yHiLimit,               int zHiLimit,
                    double delX,            double delY,                double delZ, 
                    double ****Temp_CC,     double ****press_cc,        double ****rho_CC,
                    double ****viscosity_CC,double ****thermalCond_CC,  double ****cv_CC,
                    double ****speedSound,      
                    double t_final,         double *t_output_vars,      double *delt_limits,
                    int **BC_inputs,        int printSwitch,            double CFL,
                    int nMaterials);
                    
                    
                    
/*______________________________________________________________________
*    E  Q  U  A  T  I  O  N  _  O  F  _  S  T  A  T  E  .  C 
*_______________________________________________________________________*/
void equation_of_state(
                    int xLoLimit,           int yLoLimit,               int zLoLimit,
                    int xHiLimit,           int yHiLimit,               int zHiLimit,
                    double *R,              double ****press_CC,        double ****rho_CC,
                    double ****Temp_CC,     double ****cv_CC,           int nMaterials   );

/*______________________________________________________________________
*    S  P  E  E  D  _  O  F  _  S  O  U  N  D  .  C 
*_______________________________________________________________________*/
void speed_of_sound(
                    int xLoLimit,           int yLoLimit,               int zLoLimit,
                    int xHiLimit,           int yHiLimit,               int zHiLimit,
                    double *gamma,          double *R,                  double ****Temp_CC,
                    double ****speedSound,  int nMaterials   );
/*______________________________________________________________________
*    G  R  I  D  .  C 
*_______________________________________________________________________*/                    
void generateGrid(  int xLoLimit,           int yLoLimit,               int zLoLimit,
                    int xHiLimit,           int yHiLimit,               int zHiLimit,
                    double delX,            double delY,                double delZ, 
                    double ***x_CC,         double ***y_CC,             double ***z_CC,   
                    double ***Vol_CC,
                    double *****x_FC,       double *****y_FC,          double *****z_FC );
/*______________________________________________________________________
*   I N  I  T  I  A  L  I  Z  E  _  V  A  R  I  A  B  L  E  S  .  C 
*_______________________________________________________________________*/  
void    initialize_darray_4d( 
                    int xLoLimit,           int yLoLimit,               int zLoLimit,
                    int xHiLimit,           int yHiLimit,               int zHiLimit,
                    double ****data_array,  int m,                      double constant,          
                    int grad_dir,           int flag_GC);
                    
void    initialize_darray_3d( 
                    int xLoLimit,           int yLoLimit,               int zLoLimit,
                    int xHiLimit,           int yHiLimit,               int zHiLimit,
                    double ***data_array,   double constant,            int grad_dir, 
                    int flag_GC); 
                      
void initializeVariables(int xLoLimit,      int yLoLimit,               int zLoLimit,
                    int xHiLimit,           int yHiLimit,               int zHiLimit,
                    double ****uvel_CC,     double ****vvel_CC,         double ****wvel_CC,
                    double *****uvel_FC,    double *****vvel_FC,        double *****wvel_FC,
                    double ****xmom_L_CC,   double ****ymom_L_CC,       double ****zmom_L_CC,
                    double ****rho_CC,      double ****press_cc,        double ****Temp_CC,    
                    double ****rho_L_CC,    double ***Vol_CC,           double ****Temp_L_CC,  
                    double ****Vol_L_CC,    double ****press_L_CC,      int nMaterials);
/*______________________________________________________________________
*  T  E  C  P  L  O  T     F  U  N  C  T  I  O  N  S 
*_______________________________________________________________________*/
int Is_it_time_to_write_output(
                    double  t,              double  *t_output_vars  );
                      
void tecplot_CC(    int xLoLimit,           int yLoLimit,               int zLoLimit,
                    int xHiLimit,           int yHiLimit,               int zHiLimit,
                    double ***x_CC,         double ***y_CC,             double ***z_CC,  
                    double ****uvel_CC,     double ****vvel_CC,         double ****wvel_CC,
                    double ****press_cc,    double ****Temp_CC,         double ****rho_CC,
                    double ****scalar1_CC,  double ****scalar2_CC,      double ****scalar3_CC,
                    int fileNum,            char fileDesc[],            char title[],
                    int nMaterials );
void dumpArrayTecplotCC(        
                    int xLoLimit,           int yLoLimit,               int zLoLimit,
                    int xHiLimit,           int yHiLimit,               int zHiLimit,
                    double ***data_array );
                            
void dumpArrayTecplotCC_MM(    
                    int xLoLimit,           int yLoLimit,               int zLoLimit,
                    int xHiLimit,           int yHiLimit,               int zHiLimit,
                    double ****data_array,  int nMaterials );    
                                                    
void tecplot_FC(    int xLoLimit,           int yLoLimit,               int zLoLimit,
                    int xHiLimit,           int yHiLimit,               int zHiLimit,
                    double *****x_FC,       double *****y_FC,           double *****z_FC,  
                    double ******uvel_FC,   double ******vvel_FC,       double ******wvel_FC,
                    int fileNum,            char fileDesc[],            char title[],
                    int nMaterials );                            

void dumpArrayTecplotFC(
                    int xLoLimit,           int yLoLimit,               int zLoLimit,
                    int xHiLimit,            int yHiLimit,               int zHiLimit,
                    double *****data_array);
                            
void dumpArrayTecplotFC_MM( 
                    int xLoLimit,           int yLoLimit,               int zLoLimit,
                    int xHiLimit,           int yHiLimit,               int zHiLimit,
                    double ******data_array,int nMaterials);
/*______________________________________________________________________
*    L  A  G  R  A  N  G  I  A  N  .  C 
*_______________________________________________________________________*/
void lagrangian_vol(  
                    int xLoLimit,           int yLoLimit,               int zLoLimit,         
                    int xHiLimit,           int yHiLimit,               int zHiLimit,         
                    double delX,            double delY,                double delZ,             
                    double delt,            double ****Vol_L_CC,        double ***Vol_CC,       
                    double ******uvel_FC,   double ******vvel_FC,       double ******wvel_FC,     
                    int nMaterials      );
void lagrangian_values(           
                    int xLoLimit,           int yLoLimit,               int zLoLimit,         
                    int xHiLimit,           int yHiLimit,               int zHiLimit,         
                    double ****Vol_L_C,     double ***Vol_CC,           
                    double ****rho_CC,      double ****rho_L_CC,
                    double ****xmom_CC,     double ****ymom_CC,         double ****zmom_CC,    
                    double ****uvel_CC,     double ****vvel_CC,         double ****wvel_CC,
                    double ****xmom_L_CC,   double ****ymom_L_CC,       double ****zmom_L_CC,     
                    double ****mass_L_CC,   double ****mass_source,
                    double ****xmom_source, double ****ymom_source,     double ****zmom_source,
                    double ****int_eng_CC,  double ****int_eng_L_CC,    double ****int_eng_source,
                    int N_Materials     );
/*______________________________________________________________________
*   PFACE.C
* This is probably old
*_______________________________________________________________________*/
void    press_face( int xLoLimit,           int yLoLimit,               int zLoLimit,
                    int xHiLimit,           int yHiLimit,               int zHiLimit,
                    double delX,            double delY,                double delZ,
                    int ***BC_types,        int ***BC_float_or_fixed,   double ***BC_Values,
                    double ****press_CC,    double ******press_FC,      double ****rho_CC,
                    int nMaterials );
                    
/*______________________________________________________________________
*    M  O  M  E  N  T  U  M  _  R  E  L  A  T  E  D  .  C 
*_______________________________________________________________________*/
void calc_flux_or_primitive_vars(   int flag,           
                    int xLoLimit,           int yLoLimit,               int zLoLimit,
                    int xHiLimit,           int yHiLimit,               int zHiLimit,
                    double ****rho_CC,      double  ***Vol_CC,          double ****uvel_CC,
                    double ****vvel_CC,     double ****wvel_CC,         double ****xmom,
                    double ****ymom,        double ****zmom,            
                    double ****cv_CC,       double ****int_eng_CC,      double ****Temp_CC,
                    int nMaterials );

/*______________________________________________________________________
*    M  O  M  E  N  T  U  M  .  C 
*_______________________________________________________________________*/
void accumulate_momentum_source_sinks(
                    int xLoLimit,           int yLoLimit,               int zLoLimit,
                    int xHiLimit,           int yHiLimit,               int zHiLimit,
                    double delt,
                    double delX,            double delY,                double delZ,
                    double *grav,
                    double ****mass_CC,     double ****rho_CC,
                    double ******press_FC,  double ****Temp_CC,         double ****cv_CC,
                    double ****uvel_CC,     double ****vvel_CC,         double ****wvel_CC,
                    double ******tau_X_FC,  double ******tau_Y_FC,      double ******tau_Z_FC,
                    double ****viscosity_CC,
                    double ****xmom_source, double ****ymom_source,     double ****zmom_source,
                    int nMaterials   );

/*______________________________________________________________________
*    E  N  E  R  G  Y  .  C 
*_______________________________________________________________________*/
void accumulate_energy_source_sinks(
                    int xLoLimit,           int yLoLimit,               int zLoLimit,
                    int xHiLimit,           int yHiLimit,               int zHiLimit,
                    double delt,            
                    double delX,            double delY,                double delZ,    
                    double *grav,           double ****mass_CC,         double ****rho_CC,          
                    double ****press_CC,    double ****delPress_CC,     double ****Temp_CC,         
                    double ****cv_CC,       double ****speedSound,      
                    double ****uvel_CC,     double ****vvel_CC,         double ****wvel_CC,         double ****int_eng_source,  
                    double ****div_velFC_CC,
                    int nMaterials   );
/*______________________________________________________________________
*  I  N  T  E  R  P  O  L  A  T  E  _  V  E  L  _  C  C  _  T  O  _  F  C .C
*_______________________________________________________________________*/
void vel_Face(  int xLoLimit,               int yLoLimit,               int zLoLimit,
                int xHiLimit,               int yHiLimit,               int zHiLimit,
                double delX,                double delY,                double delZ,
                double delt,
                double ****press_L_CC,      double ****rho_CC,          double *grav,
                double ****uvel_CC,         double ****vvel_CC,         double ****wvel_CC,
                double ******uvel_FC,       double ******vvel_FC,       double ******wvel_FC,                              
                int nMaterials);
                
void compute_face_centered_velocities( 
                int xLoLimit,               int yLoLimit,               int zLoLimit,
                int xHiLimit,               int yHiLimit,               int zHiLimit,
                double delX,                double delY,                double delZ,
                double delt,
                int    ***BC_types,         int    ***BC_float_or_fixed,double ***BC_Values,
                double ****rho_CC,          double *grav,               double ****press_L_CC,
                double ****uvel_CC,         double ****vvel_CC,         double ****wvel_CC,
                double ******uvel_FC,       double ******vvel_FC,       double ******wvel_FC,
                int nMaterials);
/*______________________________________________________________________
*  T  I  M  E  A  D  V  A  N  C  E  D .C
*_______________________________________________________________________*/
void advect_and_advance_in_time(   
                int xLoLimit,               int yLoLimit,               int zLoLimit,      
                int xHiLimit,               int yHiLimit,               int zHiLimit,
                double delX,                double delY,                double delZ,                
                double ***Vol_CC,           double ****rho_CC,   
                double ****xmom_CC,         double ****ymom_CC,         double ****zmom_CC,       
                double ****Vol_L_CC,        double ****rho_L_CC,        double ****mass_L_CC,
                double ****xmom_L_CC,       double ****ymom_L_CC,       double ****zmom_L_CC,
                double ****int_eng_CC,      double ****int_eng_L_CC, 
                double ******uvel_FC,       double ******vvel_FC,       double ******wvel_FC,       
                double delt,                
                int nMaterials);
                      
    
/*______________________________________________________________________
*  C  O  M  M  O  N  F  U  N  C  T  I  O  N  S .C
*_______________________________________________________________________*/
 void find_delta_time_based_on_FC_vel(
                int xLoLimit,               int yLoLimit,               int zLoLimit,
                int xHiLimit,               int yHiLimit,               int zHiLimit,
                double *delt,               double *delt_limits,               
                double delX,                double delY,                double delZ,
                double ******uvel_FC,       double ******vvel_FC,       double ******wvel_FC, 
                double ****speedSound,      double CFL,                 int nMaterials);
                
 void find_delta_time_based_on_CC_vel(
                int xLoLimit,               int yLoLimit,               int zLoLimit,
                int xHiLimit,               int yHiLimit,               int zHiLimit,
                double *delt,               double *delt_limits,               
                double delX,                double delY,                double delZ,
                double ****uvel_CC,         double ****vvel_CC,         double ****wvel_CC, 
                double ****speedSound,      double CFL,                 int nMaterials);
void find_delta_time_based_on_change_in_vol(
                int xLoLimit,               int yLoLimit,               int zLoLimit,
                int xHiLimit,               int yHiLimit,               int zHiLimit,
                double *delt,               double *delt_limits,
                double delX,                double delY,                double delZ,
                double ****uvel_CC,         double ****vvel_CC,         double ****wvel_CC,
                double ******uvel_FC,       double ******vvel_FC,       double ******wvel_FC,
                double ****speedSound,      double CFL,                 int nMaterials  );


 void find_loop_index_limits_at_domain_edges(                
                int xLoLimit,               int yLoLimit,               int zLoLimit,
                int xHiLimit,               int yHiLimit,               int zHiLimit,
                int *xLo,                   int *yLo,                   int *zLo,
                int *xHi,                   int *yHi,                   int *zHi,
                int wall    );

void zero_arrays_3d(  
                int xLoLimit,               int yLoLimit,               int zLoLimit,
                int xHiLimit,               int yHiLimit,               int zHiLimit,
                int n_data_arrays,
                double ***array1,           ...);
                
void divergence_of_face_centered_velocity(  
                int xLoLimit,               int yLoLimit,               int zLoLimit,
                int xHiLimit,               int yHiLimit,               int zHiLimit,
                double delX,                double delY,                double delZ,
                double ******uvel_FC,       double ******vvel_FC,       double ******wvel_FC,
                double ****div_vel_FC,      int nMaterials); 
                
void zero_arrays_4d(  
                int xLoLimit,               int yLoLimit,               int zLoLimit,
                int xHiLimit,               int yHiLimit,               int zHiLimit,
                int n4dl,                   int n4dh,                   int n_data_arrays,
                double ****array1,          ...);
                
void zero_arrays_5d(  
                int xLoLimit,               int yLoLimit,               int zLoLimit,
                int xHiLimit,               int yHiLimit,               int zHiLimit,
                int n4dlo,                  int n4dhi,                  
                int n5dlo,                  int n5dhi,                  int n_data_arrays,
                double *****array1,          ...);

void zero_arrays_6d(  
                int xLoLimit,               int yLoLimit,               int zLoLimit,
                int xHiLimit,               int yHiLimit,               int zHiLimit,
                int n4dlo,                  int n4dhi,                  
                int n5dlo,                  int n5dhi,                  int n_data_arrays,
                double ******array1,          ...);              

                                   
void grad_q(
                int xLoLimit,               int yLoLimit,               int zLoLimit,
                int xHiLimit,               int yHiLimit,               int zHiLimit,
                double delX,                double delY,                double delZ,
                double ****q_CC,
                double ***grad_q_X,         double ***grad_q_Y,         double ***grad_q_Z,
                int m);


 void grad_FC_Xdir( 
                int i,                      int j,                      int k, 
                int m,                      double ****data,            double delX, 
                double *grad);
 void grad_FC_Ydir( 
                int i,                      int j,                      int k, 
                int m,                      double ****data,            double delY, 
                double *grad);
 void grad_FC_Zdir( 
                int i,                      int j,                      int k,
                int m,                      double ****data,            double delZ, 
                double *grad);
 
 void interpolate_to_FC_MF(
                double ****,                double ****,                double *results, 
                int i,                      int j,                      int k, 
                int m);
 
 void Message(  int abort,                  char filename[],            char subroutine[], 
                char message[]);
 void stopwatch(char message[],             time_t start);
 
 void printData_5d(  
                int xLoLimit,               int yLoLimit,               int zLoLimit,
                int xHiLimit,               int yHiLimit,               int zHiLimit,
                int n4dlo,                  int n4dhi,
                int n5dlo,                  int n5dhi,
                char subroutine[],          char message[],             double *****data_array, 
                int ptr_flag,               int ghostcells);
                
 void printData_6d(  
                int xLoLimit,               int yLoLimit,               int zLoLimit,
                int xHiLimit,               int yHiLimit,               int zHiLimit,
                int n4dlo,                  int n4dhi,
                int n5dlo,                  int n5dhi,
                char subroutine[],          char message[],             double ******data_array, 
                int ghostcells);
                        
 void printData_FC(     
                int xLoLimit,               int yLoLimit,               int zLoLimit,
                int xHiLimit,               int yHiLimit,               int zHiLimit,
                char subroutine[],          char message[],             double ****data_array);
 
 void printData_4d(     
                int xLoLimit,               int yLoLimit,               int zLoLimit,
                int xHiLimit,               int yHiLimit,               int zHiLimit,
                int n4dlo,                  int n4dhi,
                char subroutine[],          char message[],             double ****data_array);
                 
 void printData_3d(     
                int xLoLimit,               int yLoLimit,               int zLoLimit,
                int xHiLimit,               int yHiLimit,               int zHiLimit,
                char subroutine[],          char message[],             double ***data_array);

 void printData_1d(  
                int xLoLimit,               int xHiLimit,
                char subroutine[],          char message[],             double  *data_array   );

void print_5d_where_computations_have_taken_place(                         
                int xLoLimit,               int yLoLimit,               int zLoLimit,           
                int xHiLimit,               int yHiLimit,               int zHiLimit,           
                int n4dlo,                  int n4dhi,                  int n5dlo, 
                int n5dhi,
                char subroutine[],          char message[],             double *****data_array,
                int ghostcells       ); 
                
void print_4d_where_computations_have_taken_place(                         
                int xLoLimit,               int yLoLimit,               int zLoLimit,           
                int xHiLimit,               int yHiLimit,               int zHiLimit,           
                int n4dlo,                  int n4dhi,                  
                char subroutine[],          char message[],             double ****data_array,
                int ghostcells       );

void print_3d_where_computations_have_taken_place(                         
                int xLoLimit,               int yLoLimit,               int zLoLimit,           
                int xHiLimit,               int yHiLimit,               int zHiLimit,                             
                char subroutine[],          char message[],             double ***data_array,
                int ghostcells       ); 


void explicit_delPress
             (  
                int xLoLimit,               int yLoLimit,               int zLoLimit,
                int xHiLimit,               int yHiLimit,               int zHiLimit,
                double delX,                double delY,                double delZ,
                double ****div_velFC_CC,
                double ****delPress_CC,     double ****press_CC,
                double ****rho_CC,          double delt,                double ****speedSound,
                int nMaterials );
/*______________________________________________________________________
*    P  R  E  S  S  U  R  E  _  P  C  G  .  C 
*_______________________________________________________________________*/
#ifdef IMPLICIT
void   compute_delta_Press_Using_PCGMG(
                int xLoLimit,               int yLoLimit,               int zLoLimit,
                int xHiLimit,               int yHiLimit,               int zHiLimit,
                double delX,                double delY,                double delZ,
                double delt,                double ****rho_CC,          double ****speedSound,      
                double ******uvel_FC,       double ******vvel_FC,       double ******wvel_CC,
                double ****delPress_CC,     double ****press_CC,        int ***BC_types,
                int nMaterials);
                
int initializeLinearSolver(
                int xLoLimit,               int yLoLimit,               int zLoLimit,
                int xHiLimit,               int yHiLimit,               int zHiLimit,
                double delX,                double delY,                double delZ,
                double delt,                double ****rho_CC,          double ****speedSound,
                int ***BC_types,            int nMaterials,             UserCtx *userctx); 
#endif
/*______________________________________________________________________
*    C  O  M  P  U  T  E  S  T  E  N  C  I  L  W  E  I  G  H  T  S  .  C 
*_______________________________________________________________________*/ 
#ifdef IMPLICIT                   
void calc_delPress_Stencil_Weights_Neuman(
                int xLoLimit,               int yLoLimit,               int zLoLimit,             
                int xHiLimit,               int yHiLimit,               int zHiLimit,             
                double delX,                double delY,                double delZ,
                double delt,                int ***BC_types,            
                double ****rho_CC,          double ****speedSound,      int nMaterials,
                stencilMatrix* stencil,     Mat*    A);            

void calc_delPress_Stencil_Weights_Dirichlet(
                int xLoLimit,               int yLoLimit,               int zLoLimit,             
                int xHiLimit,               int yHiLimit,               int zHiLimit,             
                double delX,                double delY,                double delZ,
                double delt,                int ***BC_types,            
                double ****rho_CC,          double ****speedSound,      int nMaterials,
                stencilMatrix* stencil,     Mat*    A); 
#endif
/*______________________________________________________________________
*    C  O  M  P  U  T  E  S  O  U  R  C  E  .  C 
*_______________________________________________________________________*/        
#ifdef IMPLICIT        
void calc_delPress_RHS(
                int xLoLimit,               int yLoLimit,               int zLoLimit,        
                int xHiLimit,               int yHiLimit,               int zHiLimit,        
                double delX,                double delY,                double delZ,
                double ******uvel_FC,       double ******vvel_FC,       double ******wvel_FC,
                UserCtx *userctx,           Vec *solution,       
                int nMaterials);
#endif
/*______________________________________________________________________
*    P  R  E  S  S  U  R  E  _  .  C 
*_______________________________________________________________________*/
 void   press_eq_residual(
                int xLoLimit,               int yLoLimit,               int zLoLimit,
                int xHiLimit,               int yHiLimit,               int zHiLimit,
                double delX,                double delY,                double delZ,
                double delt,                double ****rho_CC,          double ****speedSound,
                double ******uvel_FC,       double ******vvel_FC,       double ******wvel_FC,
                double ****delPress_CC,     double ****press_CC,        double *residual,
                int nMaterials);
/*______________________________________________________________________
*    S  H  E  A  R  _  S  T  R  E  S  S  .  C 
*_______________________________________________________________________*/

void shear_stress_Xdir(
                int xLoLimit,               int yLoLimit,               int zLoLimit,
                int xHiLimit,               int yHiLimit,               int zHiLimit,
                double delX,                double delY,                double delZ,
                double ****uvel_CC,         double ****vvel_CC,         double ****wvel_CC,
                double ****viscosity_CC,    double ******tau_X_FC,      int nMaterials   );

void shear_stress_Ydir(
                int xLoLimit,               int yLoLimit,               int zLoLimit,
                int xHiLimit,               int yHiLimit,               int zHiLimit,
                double delX,                double delY,                double delZ,
                double ****uvel_CC,         double ****vvel_CC,         double ****wvel_CC,
                double ****viscosity_CC,    double ******tau_Y_FC,      int nMaterials   );

/*______________________________________________________________________
*    E  Q  U  A  T  E  _  P  T  R  _  A  D  D  R  S  S  .  C 
*_______________________________________________________________________*/
void equate_ptr_addresses_adjacent_cell_faces(
                double *****x_FC,           double *****y_FC,           double *****z_FC,
                double ******uvel_FC,       double ******vvel_FC,       double ******wvel_FC,
                double ******press_FC,      
                double ******tau_x_FC,      double ******tau_y_FC,      double ******tau_z_FC,     
                int n_materials);


/*______________________________________________________________________
*    B  O  U  N  D  A  R  Y  _  C  O  N  D  .  C 
*_______________________________________________________________________*/
 void definition_of_different_physical_boundary_conditions( 
                int **BC_inputs,            int ***BC_types,            int ***BC_float_or_fixed,      
                double ***BC_Values,        int nMaterials        );


 void update_CC_FC_physical_boundary_conditions( 
                int xLoLimit,               int yLoLimit,               int zLoLimit,
                int xHiLimit,               int yHiLimit,               int zHiLimit,
                double delX,                double delY,                double delZ,
                int ***BC_types,            int ***BC_float_or_fixed,   double ***BC_Values,
                int nMaterials,             int n_data_arrays,          
                double ****data_CC,         int var,                    double ******data_FC,
                ... );
                
 void update_CC_physical_boundary_conditions( 
                int xLoLimit,               int yLoLimit,               int zLoLimit,
                int xHiLimit,               int yHiLimit,               int zHiLimit,
                double delX,                double delY,                double delZ,
                int ***BC_types,            int ***BC_float_or_fixed,   double ***BC_Values,
                int nMaterials,             int n_data_arrays,          
                double ****data_CC,         int var, ...);

void set_Dirichlet_BC(
                int xLoLimit,               int yLoLimit,               int zLoLimit,
                int xHiLimit,               int yHiLimit,               int zHiLimit,
                double ****data_CC,         int var,                    
                int ***BC_types,            double ***BC_values, 
                int m);
                
void set_Neumann_BC(
                int xLoLimit,               int yLoLimit,               int zLoLimit,
                int xHiLimit,               int yHiLimit,               int zHiLimit,
                double delX,                double delY,                double delZ,
                double ****data_CC,         int var,              
                int ***BC_types,            double ***BC_values, 
                int m);
                
void neumann_BC_diffenence_formula( 
                int xLo,                    int yLo,                    int zLo,
                int xHi,                    int yHi,                    int zHi,
                int wall,
                double delX,                double delY,                double delZ,
                double ****data_CC,         double ***BC_Values,        int var,                                           
                int m        );
                
void set_corner_cells_BC( 
             
                int xLoLimit,               int yLoLimit,               int zLoLimit,
                int xHiLimit,               int yHiLimit,               int zHiLimit,
                double  ****data_CC,        
                int m);
                
void set_Periodic_BC(
                int xLoLimit,               int yLoLimit,               int zLoLimit,
                int xHiLimit,               int yHiLimit,               int zHiLimit,
                double ****data_CC,         int var,                    
                int ***BC_types,            int m);
                                
 void set_Wall_BC_FC( 
                int xLoLimit,               int yLoLimit,               int zLoLimit,
                int xHiLimit,               int yHiLimit,               int zHiLimit,
                double ******uvel_FC,       double ******vvel_FC,       double ******wvel_FC,
                int ***BC_types,             int m);


                
 void setPressureBoundaryConditions(
                int xLoLimit,               int yLoLimit,               int zLoLimit,
                int xHiLimit,               int yHiLimit,               int zHiLimit,
                double ****press_CC,        int m); 
/*______________________________________________________________________
*    B  O  U  N  D  A  R  Y  _  C  O  N  D  _  F  C  .  C 
*_______________________________________________________________________*/
void set_Neumann_BC_FC(
                int xLoLimit,               int yLoLimit,               int zLoLimit,
                int xHiLimit,               int yHiLimit,               int zHiLimit,
                double ****data_CC,         double ******data_FC,       int var,       
                int ***BC_types,            int ***BC_float_or_fixed,   int m);
                
 void set_Dirichlet_BC_FC(           
                int xLoLimit,               int yLoLimit,               int zLoLimit,             
                int xHiLimit,               int yHiLimit,               int zHiLimit,             
                double ******data_FC,       int var,                      
                int ***BC_types,            double ***BC_Values,        int ***BC_float_or_FLOAT,                          
                int nMaterials        );

/*______________________________________________________________________
*    A  D  V  E  C  T  _  Q  _  F  L  U  X  .  C 
*_______________________________________________________________________*/
 
void q_out_flux( 
                int xLoLimit,               int yLoLimit,               int zLoLimit,
                int xHiLimit,               int yHiLimit,               int zHiLimit,
                double delX,                double delY,                double delZ,
                double ***gradient_limiter, 
                double ****outflux_vol,     double ****outflux_vol_CF,     
                double ****r_out_x,         double ****r_out_y,         double ****r_out_z,
                double ****r_out_x_CF,      double ****r_out_y_CF,      double ****r_out_z_CF,
                double ****q_outflux,       double ****q_outflux_CF,
                double ****q_CC,           int m );
                
void q_in_flux(
                int xLoLimit,               int yLoLimit,               int zLoLimit,
                int xHiLimit,               int yHiLimit,               int zHiLimit,
                double ****q_influx,        double ****q_influx_CF,
                double ****q_outflux,       double ****q_outflux_CF,
                int m );
/*______________________________________________________________________
*  A  D  V  E  C  T  _  Q  _  V  E  R  T  E  X  .  C 
*_______________________________________________________________________*/
void find_q_vertex_max_min(    
                int xLoLimit,               int yLoLimit,               int zLoLimit,
                int xHiLimit,               int yHiLimit,               int zHiLimit,
                double delX,                double delY,                double delZ,
                double ****q_CC,            double ***q_VRTX_MAX,       double ***q_VRTX_MIN,
                int m);
                
void find_q_vertex(    
                int xLoLimit,               int yLoLimit,               int zLoLimit,
                int xHiLimit,               int yHiLimit,               int zHiLimit,
                double delX,                double delY,                double delZ,
                double ****q_CC,            double *****q_VRTX,         int m);


/*______________________________________________________________________
*    A  D  V  E  C  T  _  Q  .  C 
*_______________________________________________________________________*/
void advect_q(
                int xLoLimit,               int yLoLimit,               int zLoLimit,
                int xHiLimit,               int yHiLimit,               int zHiLimit,
                double delX,                double delY,                double delZ,
                double ****q_CC,
                double ****r_out_x,         double ****r_out_y,         double ****r_out_z,
                double ****r_out_x_CF,      double ****r_out_y_CF,      double ****r_out_z_CF, 
                double ****outflux_vol,     double ****outflux_vol_CF,
                double ****influx_vol,      double ****influx_vol_CF,                            
                double ****advect_q_CC,     int m );
                
void influx_outflux_volume(    
                int xLoLimit,               int yLoLimit,               int zLoLimit,
                int xHiLimit,               int yHiLimit,               int zHiLimit,
                double delX,                double delY,                double delZ,
                double delt, 
                double ******uvel_FC,       double ******vvel_FC,       double ******wvel_FC,
                double ****influx_vol,      double ****influx_vol_CF,
                double ****outflux_vol,     double ****outflux_vol_CF,  int m);
/*______________________________________________________________________
*    A  D  V  E  C  T  _  P  R  E  P  R  O  C  E  S  S  .  C 
*_______________________________________________________________________*/
void advect_preprocess(
                int xLoLimit,               int yLoLimit,               int zLoLimit,
                int xHiLimit,               int yHiLimit,               int zHiLimit,
                double delX,                double delY,                double delZ,
                double delt,
                double ******uvel_FC,       double ******vvel_FC,       double ******wvel_FC,
                double ****r_out_x,         double ****r_out_y,         double ****r_out_z,
                double ****r_out_x_CF,      double ****r_out_y_CF,      double ****r_out_z_CF,
                double ****outflux_vol,     double ****outflux_vol_CF,
                double ****influx_vol,      double ****influx_vol_CF,
                int m);
/*______________________________________________________________________
*    A  D  E  C  T  _  C  E  N  T  R  O  I  D  S  .  C 
*_______________________________________________________________________*/
void outflow_vol_centroid(    
                int xLoLimit,               int yLoLimit,               int zLoLimit,
                int xHiLimit,               int yHiLimit,               int zHiLimit,
                double delX,                double delY,                double delZ,
                double delt,
                double ******uvel_FC,       double ******vvel_FC,       double ******wvel_FC,
                double ****r_out_x,         double ****r_out_y,         double ****r_out_z,
                double ****r_out_x_CF,      double ****r_out_y_CF,      double ****r_out_z_CF,
                int m   );
/*______________________________________________________________________
*    A  D  V  E  C  T  _  G  R  A  D  _  L  I  M  I  T  E  R  .  C 
*_______________________________________________________________________*/

void gradient_limiter(    
                int xLoLimit,               int yLoLimit,               int zLoLimit,         
                int xHiLimit,               int yHiLimit,               int zHiLimit,         
                double delX,                double delY,                double delZ,             
                double ****q_CC,            double ***grad_limiter,     int m );                       


void find_q_CC_max_min(    
                int xLoLimit,               int yLoLimit,               int zLoLimit,
                int xHiLimit,               int yHiLimit,               int zHiLimit,
                double ****q_CC,            double ***q_CC_max,         double ***q_CC_min,
                int m);
/*______________________________________________________________________
*    P  L  O  T  _  C  O  M  M  O  N  .  C 
*_______________________________________________________________________*/
void plot_open_window_screen(
                int max_sub_win,            int *n_sub_win);

void plot_open_window_file(
                int max_sub_win,            int *n_sub_win,             char *basename, 
                int filetype);


void plot_generate_axis(    
                char *x_label,              char *y_label,              char *graph_label,
                float *x_min,               float *x_max,               float *y_min,      
                float *y_max,               int *error);
        
void plot_legend(
                float data_max,             float data_min,             float x_max,        
                float y_max);

int plot_scaling(  
                const float *data_array,    int max_len,                float *y_min,       float *y_max);

void plot_color_spectrum(void);

/*  old stuff

    void plot_scaling_CC(      
        int xLoLimit,           int yLoLimit,       int zLoLimit,
        int xHiLimit,           int yHiLimit,       int zHiLimit, 
        double ***data_array,   float *y_min,       float *y_max);*/

void plot_scaling_FC(   
                int xLoLimit,               int yLoLimit,               int zLoLimit,
                int xHiLimit,               int yHiLimit,               int zHiLimit,
                double ******data_array,    int m,
                float *y_min,               float *y_max); 

/*______________________________________________________________________
*    P  L  O  T  _  C  O  N  T  R  O  L  .  C 
*_______________________________________________________________________*/
void    plot(    
                const float *data,          const float *data2,         int max_len,    
                double delX,                int xLoLimit,               int xHiLimit,
                int yLoLimit,               int yHiLimit,
                char *x_label,              char *y_label,              char *graph_label,
                int plot_type,              int outline_ghostcells,     int max_sub_win,
                char *file_basename,        int filetype);
 
/*______________________________________________________________________
*    P  L  O  T  _  C  O  N  T  O  U  R  .  C 
*_______________________________________________________________________*/
    void plot_contour(   
               int xLoLimit,                int xHiLimit,               int yLoLimit,
               int yHiLimit,                float data_min,             float data_max,
               const float *data );
    void plot_contour_checkerboard(   
               int xLoLimit,                int xHiLimit,               int yLoLimit,
               int yHiLimit,                float data_min,             float data_max,
               const float *data );
    
/*______________________________________________________________________
*    P  L  O  T  _  V  E  C  T  O  R  .  C 
*_______________________________________________________________________*/
void plot_vector_2D(       
               int xHiLimit,               int yHiLimit,
               int max_len,                const float *data_array1,
               const float *data_array2);
/*______________________________________________________________________
*    P  L  O  T  _  2  D  _  L  I  N  E  .  C 
*_______________________________________________________________________*/

 void plot_2d_scatter(
               const float *x_data,         const float *y_data,
               int max_len,                 int color,                   int symbol,
               float symbol_size    );
        
 void plot_dbl_flt(                        
                int     max_len,
                double  ****dbl_1,          double  ****dbl_2,
                float   *flt_1,             float   *flt_2 );

/*______________________________________________________________________
*   P L  O  T  _  F  A  C  E  _  C  E  N  T  E  R  .  C 
*_______________________________________________________________________*/
 void plot_face_centered_data( 
                int xLoLimit,               int xHiLimit,
                int yLoLimit,               int yHiLimit,
                int zLoLimit,               int zHiLimit,
                double delX,                double delY,
                double ******data,
                char x_label[],             char y_label[],             char graph_label[],
                int  outline_ghostcells,    int  max_sub_win,
                char file_basename[],       int  filetype,              int m);

/*______________________________________________________________________
*    P  L  O  T  _  C  U  R  S  O  R  _  P  O  S  .  C 
*_______________________________________________________________________*/
void plot_cursor_position(
                int max_sub_win,            int *n_sub_win);








/*______________________________________________________________________
*   PRESS_VEL_FACE.C OLD as of 12/17/99
*_______________________________________________________________________*/
 void vel_initial_iteration(  
                int xLoLimit,               int yLoLimit,               int zLoLimit,
                int xHiLimit,               int yHiLimit,               int zHiLimit,               
                int **which_BC_set,
                double ******uvel_FC,       double ******vvel_FC,       double ******wvel_FC,
                double ****press_CC,
                double ****rho_CC,          double delt,                double *grav,
                double delX,                double delY,                double delZ,
                int nMaterials);
                
void vel_Face_n_iteration(
                int xLoLimit,               int yLoLimit,               int zLoLimit,
                int xHiLimit,               int yHiLimit,               int zHiLimit,
                double delX,                double delY,                double delZ,
                double ******uvel_FC,       double ******vvel_FC,       double ******wvel_FC,
                double ******uvel_half_FC,  double ******vvel_half_FC,  double ******wvel_half_FC,
                double ****delPress_CC,     double ****press_CC,
                double ****rho_CC,          double delt,               double *grav,
                double ****speedSound,      int nMaterials);
                
/*______________________________________________________________________
*   PRESSURE_ITERATION.C (OLD AS OF 12/17/99)
*_______________________________________________________________________*/
 void pressure_iteration(    
                int xLoLimit,               int yLoLimit,               int zLoLimit,
                int xHiLimit,               int yHiLimit,               int zHiLimit,
                double delX,                double delY,                double delZ,
                double ******uvel_FC,       double ******vvel_FC,       double ******wvel_FC,
                double ****uvel_CC,         double ****vvel_CC,         double ****wvel_CC,                
                double ****press_CC,        double ****delPress_CC,        
                double ****rho_CC,          double delt,
                double *grav,
                int ***BC_types,             double ***BC_Values,       int ***BC_float_or_fixed,
                double ****speedSound,     int nMaterials); 
                
/*______________________________________________________________________
*   CONVECTIVE_VISCOUS_TERMS.C (OLD AS OF 12/17/99)
*_______________________________________________________________________*/
void    convective_viscous_terms(
                int xLoLimit,               int yLoLimit,               int zLoLimit,
                int xHiLimit,               int yHiLimit,               int zHiLimit,
                double delX,                double delY,                double delZ,
                double ******uvel_FC,       double ******vvel_FC,       double ******wvel_FC,
                double delt,           
                double ****F,               double ****G);

                             
#ifdef __cplusplus
}
#endif
#endif      /*FUNCTIONDECLARE_H*/
