
/* 
======================================================================*/
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include "nrutil+.h"
#include "functionDeclare.h"
#include "parameters.h"
#include "switches.h"
#include "macros.h"
#include "inline.h"                    
/* ---------------------------------------------------------------------
 Function:  shear_stress_Xdir--SOURCE: Step 4,Compute the x-component of the shear stress
 Filename:  shear_stress.c
 Purpose:   This function computes the x-component of the shear stress
            tau_xx, ta_yx, tau_zx for the left, right, top, bottom
            front and back faces.
            
 Computational Domain:
            The shear stress is computed on every face and in every cell 
            inside of the domain.
            
 Ghostcell data dependency: 
            This routine uses a single layer of ghostcells to compute the 
            viscosity and and edge velocities  
   
 Version       Programmer         Date       Description                      
     -------   ----------         ----       -----------                 
        1.0     Todd Harman       1/18/00 
                                   
 Implementation Note:
            find_transport_property_FC
            find_edge_vel__edge_is_parallel_w_(*)_axis 
            are inline functions.

Note currently I comput the shear stress for each face of the cell
I only need to do three of the faces, not all 6, with the pointers
equated
 ---------------------------------------------------------------------  */
void shear_stress_Xdir(
    int         xLoLimit,               /* x-array Lower Interior Nodes     */
    int         yLoLimit,               /* y-array Lower Interior Nodes     */
    int         zLoLimit,               /* z-array Lower Interior Nodes     */
    int         xHiLimit,               /* x-array Upper Interior Nodes     */
    int         yHiLimit,               /* y-array Upper Interior Nodes     */
    int         zHiLimit,               /* z-array Upper Interior Nodes     */
    double      delX,                   /* cell spacing                     (INPUT) */
    double      delY,                   /*          ---//---                (INPUT) */
    double      delZ,                   /*          ---//---                (INPUT) */
    double      ****uvel_CC,            /* cell-centered velocities         (INPUT) */
    double      ****vvel_CC,            /*          ---//---                (INPUT) */
    double      ****wvel_CC,            /*          ---//---                (INPUT) */
    double      ****viscosity_CC,       /* cell centered viscosity          (INPUT) */
    double      ******tau_X_FC,         /* face-centered shearstress        (OUTPUT)*/
    int         nMaterials   )
                                                         
{
    int     i, j, k, m;                 /* cell face locators               */
    double  *viscosity_FC,               /* effective viscosity at the face  */
            term1,                      /* temporary terms                  */
            term2,
            grad_1,
            grad_2;
            
    double  vvel_ED_top_right_z,        /* edge velocities                  */        
            vvel_ED_top_left_z,
            vvel_ED_bottom_right_z,
            vvel_ED_bottom_left_z,
            wvel_ED_front_right_y,
            wvel_ED_front_left_y,
            wvel_ED_back_right_y,
            wvel_ED_back_left_y;
                                  
    double  *div_vel_FC;                /* face-centered divergenc of the   */ 
                                        /* velocity                         */                
/*__________________________________
*   Plotting variables
*___________________________________*/
#if switchDebug_shear_stress_Xdir
    #include "plot_declare_vars.h"   
#endif
/*__________________________________
* double check inputs, 
*___________________________________*/
    assert ( xLoLimit >= 0 && xHiLimit <= X_MAX_LIM);
    assert ( yLoLimit >= 0 && yHiLimit <= Y_MAX_LIM);
    assert ( zLoLimit >= 0 && zHiLimit <= Z_MAX_LIM);
/*__________________________________
*   Allocate memory, initialize variables
*   
*___________________________________*/
    viscosity_FC    = dvector_nr(1, N_CELL_FACES);
    div_vel_FC      = dvector_nr(1, N_CELL_FACES);
/*__________________________________
*   Now accumulate the different contributions
*___________________________________*/
    for ( m = 1; m <= nMaterials; m++)
    {
        for ( i = xLoLimit; i <= xHiLimit; i++)
        {
            for ( j = yLoLimit; j <= yHiLimit; j++)
            {
                for ( k = zLoLimit; k <= zHiLimit; k++)
                {

                    /*__________________________________
                    * Calculate the viscosity at the face-centers
                    * and the velocity at edge on the cell
                    * These are inline functions defined 
                    * in inline.h
                    *___________________________________*/
                    find_transport_property_FC(i, j, k, m, viscosity_CC, viscosity_FC);

                    vvel_ED_top_right_z     = find_edge_vel__edge_is_parallel_w_Z_axis(vvel_CC, i,  j,  k,  m);
                    vvel_ED_top_left_z      = find_edge_vel__edge_is_parallel_w_Z_axis(vvel_CC, i-1,j,  k,  m);
                    vvel_ED_bottom_right_z  = find_edge_vel__edge_is_parallel_w_Z_axis(vvel_CC, i,  j-1,k,  m);
                    vvel_ED_bottom_left_z   = find_edge_vel__edge_is_parallel_w_Z_axis(vvel_CC, i-1,j-1,k,  m);

                    wvel_ED_front_right_y   = find_edge_vel__edge_is_parallel_w_Y_axis(wvel_CC, i,  j,  k,  m);
                    wvel_ED_front_left_y    = find_edge_vel__edge_is_parallel_w_Y_axis(wvel_CC, i-1,j,  k,  m);
                    wvel_ED_back_right_y    = find_edge_vel__edge_is_parallel_w_Y_axis(wvel_CC, i,  j,  k-1,m);
                    wvel_ED_back_left_y     = find_edge_vel__edge_is_parallel_w_Y_axis(wvel_CC, i-1,j,  k-1,m);
                    /*__________________________________
                    *   Compute the term due to compressibility
                    *___________________________________*/
                    divergence_of_velocity_for_tau_terms_FC(
                                i,          j,          k,
                                delX,       delY,       delZ,                        
                                uvel_CC,    vvel_CC,    wvel_CC,
                                div_vel_FC, m);             
                    /*__________________________________
                    *  Left Face
                    *   tau_XX
                    *___________________________________*/
                    term1           = 2.0       * viscosity_FC[LEFT] * (uvel_CC[m][i][j][k] - uvel_CC[m][i-1][j][k])/delX;
                    term2           = (2.0/3.0) * viscosity_FC[LEFT] * div_vel_FC[LEFT];

                    *tau_X_FC[i][j][k][LEFT][m] =  term1 - term2;

                    /*__________________________________
                    *  right Face
                    *   tau_XX
                    *___________________________________*/       
                    term1           = 2.0       * viscosity_FC[RIGHT] * (uvel_CC[m][i+1][j][k] - uvel_CC[m][i][j][k])/delX;
                    term2           = (2.0/3.0) * viscosity_FC[RIGHT] * div_vel_FC[RIGHT];

                    *tau_X_FC[i][j][k][RIGHT][m] =  term1 - term2;               

                    /*__________________________________
                    *   Top Face
                    *   tau_YX
                    *___________________________________*/  
                    grad_1          = (uvel_CC[m][i][j+1][k]    - uvel_CC[m][i][j][k])  /delY;
                    grad_2          = (vvel_ED_top_right_z      - vvel_ED_top_left_z)   /delX;

                    *tau_X_FC[i][j][k][TOP][m]   = viscosity_FC[TOP] * (grad_1 + grad_2);
                    /*__________________________________
                    *   Bottom Face
                    *   tau_YX
                    *___________________________________*/ 
                    grad_1          = (uvel_CC[m][i][j][k]      - uvel_CC[m][i][j-1][k])/delY;
                    grad_2          = (vvel_ED_bottom_right_z   - vvel_ED_bottom_left_z)/delX;

                    *tau_X_FC[i][j][k][BOTTOM][m]   = viscosity_FC[BOTTOM] * (grad_1 + grad_2);

    #if (N_DIMENSIONS == 3)                
                    /*__________________________________
                    *   Front Face
                    *   tau_ZX
                    *___________________________________*/
                    grad_1          = (uvel_CC[m][i][j][k+1]    - uvel_CC[m][i][j][k])  /delZ;
                    grad_2          = (wvel_ED_front_right_y    - wvel_ED_front_left_y) /delX;

                    *tau_X_FC[i][j][k][FRONT][m]   = viscosity_FC[FRONT] * (grad_1 + grad_2);
                    /*__________________________________
                    *   Back Face
                    *   tau_ZX
                    *___________________________________*/
                    grad_1          = (uvel_CC[m][i][j][k]      - uvel_CC[m][i][j][k-1])/delZ;
                    grad_2          = (wvel_ED_back_right_y     - wvel_ED_back_left_y)  /delX;

                    *tau_X_FC[i][j][k][BACK][m]   = viscosity_FC[BACK] * (grad_1 + grad_2);
    #endif
                }
            }
        }
    }

/*______________________________________________________________________
*   DEBUGGING SECTION   
*_______________________________________________________________________*/
#if switchDebug_shear_stress_Xdir
     #define switchInclude_shear_stress_Xdir 1
     #include "debugcode.i"
     #undef switchInclude_shear_stress_Xdir
#endif

/*__________________________________
*   Free the local memory
*___________________________________*/
    free_dvector_nr( div_vel_FC,  1, N_CELL_FACES);
    free_dvector_nr(viscosity_FC, 1, N_CELL_FACES);
/*__________________________________
*   Quite all fullwarn compiler remarks
*___________________________________*/
    delZ                    = delZ;
    wvel_ED_front_right_y   = wvel_ED_front_right_y;  
    wvel_ED_front_left_y    = wvel_ED_front_left_y;   
    wvel_ED_back_right_y    = wvel_ED_back_right_y;
    wvel_ED_back_left_y     = wvel_ED_back_left_y;                  
}
/*STOP_DOC*/


/* 
======================================================================*/
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include "nrutil+.h"
#include "functionDeclare.h"
#include "parameters.h"
#include "switches.h"
#include "macros.h"
#include "inline.h"                    
/* ---------------------------------------------------------------------
 Function:  shear_stress_Ydir--SOURCE: Step 4,Compute the x-component of the shear stress
 Filename:  shear_stress.c
 Purpose:   This function computes the y-component of the shear stress
            tau_xy, ta_yy, tau_zy for the left, right, top, bottom
            front and back faces.
            
 Computational Domain:
            The shear stress is computed on every face and in every cell 
            inside of the domain.
            
 Ghostcell data dependency: 
            This routine uses a single layer of ghostcells to compute the 
            viscosity and and edge velocities  
   
 Version       Programmer         Date       Description                      
     -------   ----------         ----       -----------                 
        1.0     Todd Harman       1/18/00 
                                   
 Implementation Note:
            find_transport_property_FC
            find_edge_vel__edge_is_parallel_w_(*)_axis 
            are inline functions.

Note currently I comput the shear stress for each face of the cell
I only need to do three of the faces, not all 6, with the pointers
equated
 ---------------------------------------------------------------------  */
void shear_stress_Ydir(
    int         xLoLimit,               /* x-array Lower Interior Nodes     */
    int         yLoLimit,               /* y-array Lower Interior Nodes     */
    int         zLoLimit,               /* z-array Lower Interior Nodes     */
    int         xHiLimit,               /* x-array Upper Interior Nodes     */
    int         yHiLimit,               /* y-array Upper Interior Nodes     */
    int         zHiLimit,               /* z-array Upper Interior Nodes     */
    double      delX,                   /* cell spacing                     (INPUT) */
    double      delY,                   /*          ---//---                (INPUT) */
    double      delZ,                   /*          ---//---                (INPUT) */
    double      ****uvel_CC,            /* cell-centered velocities         (INPUT) */
    double      ****vvel_CC,            /*          ---//---                (INPUT) */
    double      ****wvel_CC,            /*          ---//---                (INPUT) */
    double      ****viscosity_CC,       /* cell centered viscosity          (INPUT) */
    double      ******tau_Y_FC,         /* face-centered shearstress        (OUTPUT)*/
    int         nMaterials   )
                                                         
{
    int     i, j, k, m;                 /* cell face locators               */
    double  *viscosity_FC,               /* effective viscosity at the face  */
            term1,                      /* temporary terms                  */
            term2,
            grad_1,
            grad_2;
            
    double uvel_ED_right_top_z,         /* edge velocities                  */
           uvel_ED_left_top_z,
           uvel_ED_right_bottom_z,
           uvel_ED_left_bottom_z,
           wvel_ED_front_top_x,
           wvel_ED_front_bottom_x,
           wvel_ED_back_top_x,
           wvel_ED_back_bottom_x;
                                  
    double  *div_vel_FC;                /* face-centered divergenc of the   */ 
                                        /* velocity                         */                
/*__________________________________
*   Plotting variables
*___________________________________*/
#if switchDebug_shear_stress_Ydir
    #include "plot_declare_vars.h"   
#endif
/*__________________________________
* double check inputs, 
*___________________________________*/
    assert ( xLoLimit >= 0 && xHiLimit <= X_MAX_LIM);
    assert ( yLoLimit >= 0 && yHiLimit <= Y_MAX_LIM);
    assert ( zLoLimit >= 0 && zHiLimit <= Z_MAX_LIM);
/*__________________________________
*   Allocate memory, initialize variables
*   
*___________________________________*/
    viscosity_FC    = dvector_nr(1, N_CELL_FACES);
    div_vel_FC      = dvector_nr(1, N_CELL_FACES);
/*__________________________________
*   Now accumulate the different contributions
*___________________________________*/
    for ( m = 1; m <= nMaterials; m++)
    {
        for ( i = xLoLimit; i <= xHiLimit; i++)
        {
            for ( j = yLoLimit; j <= yHiLimit; j++)
            {
                for ( k = zLoLimit; k <= zHiLimit; k++)
                {

                    /*__________________________________
                    * Calculate the viscosity at the face-centers
                    * and the velocity at edge on the cell
                    * These are inline functions defined 
                    * in inline.h
                    *___________________________________*/
                    find_transport_property_FC(i, j, k, m, viscosity_CC, viscosity_FC);

                    uvel_ED_right_top_z     = find_edge_vel__edge_is_parallel_w_Z_axis(vvel_CC, i,  j,  k,  m);
                    uvel_ED_left_top_z      = find_edge_vel__edge_is_parallel_w_Z_axis(vvel_CC, i-1,j,  k,  m);
                    uvel_ED_right_bottom_z  = find_edge_vel__edge_is_parallel_w_Z_axis(vvel_CC, i,  j-1,k,  m);
                    uvel_ED_left_bottom_z   = find_edge_vel__edge_is_parallel_w_Z_axis(vvel_CC, i-1,j-1,k,  m);

                    wvel_ED_front_top_x     = find_edge_vel__edge_is_parallel_w_X_axis(wvel_CC, i,  j,  k,  m);
                    wvel_ED_front_bottom_x  = find_edge_vel__edge_is_parallel_w_X_axis(wvel_CC, i  ,j-1,k,  m);
                    wvel_ED_back_top_x      = find_edge_vel__edge_is_parallel_w_X_axis(wvel_CC, i,  j,  k-1,m);
                    wvel_ED_back_bottom_x   = find_edge_vel__edge_is_parallel_w_X_axis(wvel_CC, i  ,j-1,k-1,m);
                    /*__________________________________
                    *   Compute the term due to compressibility
                    *___________________________________*/
                    divergence_of_velocity_for_tau_terms_FC(
                                i,          j,          k,
                                delX,       delY,       delZ,                        
                                uvel_CC,    vvel_CC,    wvel_CC,
                                div_vel_FC, m);             
                    /*__________________________________
                    *  Left Face
                    *   tau_XY
                    *___________________________________*/
                    grad_1          = (uvel_ED_left_top_z       - uvel_ED_left_bottom_z)   /delY;
                    grad_2          = (vvel_CC[m][i][j][k]      - vvel_CC[m][i-1][j][k])    /delX;
                    *tau_Y_FC[i][j][k][LEFT][m] =  viscosity_FC[LEFT] * (grad_1 + grad_2);

                    /*__________________________________
                    *  right Face
                    *   tau_XY
                    *___________________________________*/       
                    grad_1          = (uvel_ED_right_top_z      - uvel_ED_right_bottom_z)   /delY;
                    grad_2          = (vvel_CC[m][i+1][j][k]    - vvel_CC[m][i][j][k])      /delX;
                    *tau_Y_FC[i][j][k][RIGHT][m] =  viscosity_FC[RIGHT] * (grad_1 + grad_2);              

                    /*__________________________________
                    *   Top Face
                    *   tau_YY
                    *___________________________________*/
                    term1           = 2.0       * viscosity_FC[TOP] * (vvel_CC[m][i][j+1][k] - vvel_CC[m][i][j][k])/delY;
                    term2           = (2.0/3.0) * viscosity_FC[TOP] * div_vel_FC[TOP];

                    *tau_Y_FC[i][j][k][TOP][m] =  term1 - term2;
                    /*__________________________________
                    *   Bottom Face
                    *   tau_YY
                    *___________________________________*/ 
                    term1           = 2.0       * viscosity_FC[BOTTOM] * (vvel_CC[m][i][j][k] - vvel_CC[m][i][j-1][k])/delY;
                    term2           = (2.0/3.0) * viscosity_FC[BOTTOM] * div_vel_FC[BOTTOM];

                    *tau_Y_FC[i][j][k][BOTTOM][m] =  term1 - term2;

    #if (N_DIMENSIONS == 3)                
                    /*__________________________________
                    *   Front Face
                    *   tau_ZY
                    *___________________________________*/
                    grad_1          = (vvel_CC[m][i][j][k+1]    - vvel_CC[m][i][j][k])      /delZ;
                    grad_2          = (wvel_ED_front_top_x      - wvel_ED_front_bottom_x)   /delY2;

                    *tau_Y_FC[i][j][k][FRONT][m]   = viscosity_FC[FRONT] * (grad_1 + grad_2);
                    /*__________________________________
                    *   Back Face
                    *   tau_ZY
                    *___________________________________*/
                    grad_1          = (vvel_CC[m][i][j][k]      - vvel_CC[m][i][j][k-1])    /delZ;
                    grad_2          = (wvel_ED_back_top_x       - wvel_ED_back_bottom_x)    /delY;

                    *tau_Y_FC[i][j][k][BACK][m]   = viscosity_FC[BACK] * (grad_1 + grad_2);
    #endif
                }
            }
        }
    }

/*______________________________________________________________________
*   DEBUGGING SECTION   
*_______________________________________________________________________*/
#if switchDebug_shear_stress_Ydir
     #define switchInclude_shear_stress_Ydir 1
     #include "debugcode.i"
     #undef switchInclude_shear_stress_Ydir
#endif

/*__________________________________
*   Free the local memory
*___________________________________*/
    free_dvector_nr( div_vel_FC,  1, N_CELL_FACES);
    free_dvector_nr(viscosity_FC, 1, N_CELL_FACES);
/*__________________________________
*   Quite all fullwarn compiler remarks
*___________________________________*/
    delZ                    = delZ;
    wvel_ED_front_top_x     = wvel_ED_front_top_x;  
    wvel_ED_front_bottom_x  = wvel_ED_front_bottom_x;   
    wvel_ED_back_top_x      = wvel_ED_back_top_x;
    wvel_ED_back_bottom_x   = wvel_ED_back_bottom_x;                  
}
/*STOP_DOC*/

/* 
======================================================================*/
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include "nrutil+.h"
#include "functionDeclare.h"
#include "parameters.h"
#include "switches.h"
#include "macros.h"
#include "inline.h"                    
/* ---------------------------------------------------------------------
 Function:  shear_stress_Zdir--SOURCE: Step 4,Compute the x-component of the shear stress
 Filename:  shear_stress.c
 Purpose:   This function computes the y-component of the shear stress
            tau_xz, ta_yz, tau_zz for the left, right, top, bottom
            front and back faces.
            
 Computational Domain:
            The shear stress is computed on every face and in every cell 
            inside of the domain.
            
 Ghostcell data dependency: 
            This routine uses a single layer of ghostcells to compute the 
            viscosity and and edge velocities  
   
 Version       Programmer         Date       Description                      
     -------   ----------         ----       -----------                 
        1.0     Todd Harman       1/18/00 
                                   
 Implementation Note:
            find_transport_property_FC
            find_edge_vel__edge_is_parallel_w_(*)_axis 
            are inline functions.

Note currently I comput the shear stress for each face of the cell
I only need to do three of the faces, not all 6, with the pointers
equated
 ---------------------------------------------------------------------  */
void shear_stress_Zdir(
    int         xLoLimit,               /* x-array Lower Interior Nodes     */
    int         yLoLimit,               /* y-array Lower Interior Nodes     */
    int         zLoLimit,               /* z-array Lower Interior Nodes     */
    int         xHiLimit,               /* x-array Upper Interior Nodes     */
    int         yHiLimit,               /* y-array Upper Interior Nodes     */
    int         zHiLimit,               /* z-array Upper Interior Nodes     */
    double      delX,                   /* cell spacing                     (INPUT) */
    double      delY,                   /*          ---//---                (INPUT) */
    double      delZ,                   /*          ---//---                (INPUT) */
    double      ****uvel_CC,            /* cell-centered velocities         (INPUT) */
    double      ****vvel_CC,            /*          ---//---                (INPUT) */
    double      ****wvel_CC,            /*          ---//---                (INPUT) */
    double      ****viscosity_CC,       /* cell centered viscosity          (INPUT) */
    double      ******tau_Z_FC,         /* face-centered shearstress        (OUTPUT)*/
    int         nMaterials   )
                                                         
{
    int     i, j, k, m;                 /* cell face locators               */
    double  *viscosity_FC,               /* effective viscosity at the face  */
            term1,                      /* temporary terms                  */
            term2,
            grad_1,
            grad_2;
            
    double  uvel_ED_right_front_y,       /* edge velocities                  */
            uvel_ED_left_front_y,
            uvel_ED_right_back_y,
            uvel_ED_left_back_y,
            vvel_ED_top_front_x,     
            vvel_ED_bottom_front_x,  
            vvel_ED_top_back_x,      
            vvel_ED_bottom_back_x;   
                                  
    double  *div_vel_FC;                /* face-centered divergenc of the   */ 
                                        /* velocity                         */                
/*__________________________________
*   Plotting variables
*___________________________________*/
#if switchDebug_shear_stress_Zdir
    #include "plot_declare_vars.h"   
#endif
/*__________________________________
* double check inputs, 
*___________________________________*/
    assert ( xLoLimit >= 0 && xHiLimit <= X_MAX_LIM);
    assert ( yLoLimit >= 0 && yHiLimit <= Y_MAX_LIM);
    assert ( zLoLimit >= 0 && zHiLimit <= Z_MAX_LIM);
/*__________________________________
*   Allocate memory, initialize variables
*   
*___________________________________*/
    viscosity_FC    = dvector_nr(1, N_CELL_FACES);
    div_vel_FC      = dvector_nr(1, N_CELL_FACES);
    m               = nMaterials;

/*__________________________________
*   Now accumulate the different contributions
*___________________________________*/
    for ( m = 1; m <= nMaterials; m++)
    {
        for ( i = xLoLimit; i <= xHiLimit; i++)
        {
            for ( j = yLoLimit; j <= yHiLimit; j++)
            {
                for ( k = zLoLimit; k <= zHiLimit; k++)
                {

                    /*__________________________________
                    * Calculate the viscosity at the face-centers
                    * and the velocity at edge on the cell
                    * These are inline functions defined 
                    * in inline.h
                    *___________________________________*/
                    find_transport_property_FC(i, j, k, m, viscosity_CC, viscosity_FC);

                    uvel_ED_right_front_y   = find_edge_vel__edge_is_parallel_w_Y_axis(vvel_CC, i,  j,  k,  m);
                    uvel_ED_left_front_y    = find_edge_vel__edge_is_parallel_w_Y_axis(vvel_CC, i-1,j,  k,  m);
                    uvel_ED_right_back_y    = find_edge_vel__edge_is_parallel_w_Y_axis(vvel_CC, i,  j-1,k,  m);
                    uvel_ED_left_back_y     = find_edge_vel__edge_is_parallel_w_Y_axis(vvel_CC, i-1,j-1,k,  m);

                    vvel_ED_top_front_x     = find_edge_vel__edge_is_parallel_w_X_axis(wvel_CC, i,  j,  k,  m);
                    vvel_ED_bottom_front_x  = find_edge_vel__edge_is_parallel_w_X_axis(wvel_CC, i  ,j-1,k,  m);
                    vvel_ED_top_back_x      = find_edge_vel__edge_is_parallel_w_X_axis(wvel_CC, i,  j,  k-1,m);
                    vvel_ED_bottom_back_x   = find_edge_vel__edge_is_parallel_w_X_axis(wvel_CC, i  ,j-1,k-1,m);
                    /*__________________________________
                    *   Compute the term due to compressibility
                    *___________________________________*/
                    divergence_of_velocity_for_tau_terms_FC(
                                i,          j,          k,
                                delX,       delY,       delZ,                        
                                uvel_CC,    vvel_CC,    wvel_CC,
                                div_vel_FC, m);             
                    /*__________________________________
                    *  Left Face
                    *   tau_XZ
                    *___________________________________*/
                    grad_1          = (uvel_ED_left_front_y     - uvel_ED_left_back_y)      /delZ;
                    grad_2          = (wvel_CC[m][i][j][k]      - wvel_CC[m][i-1][j][k])    /delX;
                    *tau_Z_FC[i][j][k][LEFT][m] =  viscosity_FC[LEFT] * (grad_1 + grad_2);

                    /*__________________________________
                    *  right Face
                    *   tau_XZ
                    *___________________________________*/       
                    grad_1          = (uvel_ED_right_front_y    - uvel_ED_right_back_y)     /delZ;
                    grad_2          = (wvel_CC[m][i+1][j][k]    - wvel_CC[m][i][j][k])      /delX;
                    *tau_Z_FC[i][j][k][RIGHT][m] =  viscosity_FC[RIGHT] * (grad_1 + grad_2);              

                    /*__________________________________
                    *   Top Face
                    *   tau_YZ
                    *___________________________________*/
                    grad_1          = (vvel_ED_top_front_x      - vvel_ED_top_back_x)       /delZ;
                    grad_2          = (wvel_CC[m][i][j+1][k]    - wvel_CC[m][i][j][k])      /delY;
                    *tau_Z_FC[i][j][k][TOP][m] =  viscosity_FC[TOP] * (grad_1 + grad_2);     

                    *tau_Z_FC[i][j][k][TOP][m] =  term1 - term2;
                    /*__________________________________
                    *   Bottom Face
                    *   tau_YZ
                    *___________________________________*/ 
                    grad_1          = (vvel_ED_bottom_front_x   - vvel_ED_bottom_back_x)     /delZ;
                    grad_2          = (wvel_CC[m][i][j][k]      - wvel_CC[m][i][j][k-1])     /delY;
                    *tau_Z_FC[i][j][k][BOTTOM][m] =  viscosity_FC[BOTTOM] * (grad_1 + grad_2);     

                    *tau_Z_FC[i][j][k][BOTTOM][m] =  term1 - term2;

    #if (N_DIMENSIONS == 3)                
                    /*__________________________________
                    *   Front Face
                    *   tau_ZZ
                    *___________________________________*/
                    term1           = 2.0       * viscosity_FC[FRONT] * (wvel_CC[m][i][j][k+1] - wvel_CC[m][i][j][k])/delZ;
                    term2           = (2.0/3.0) * viscosity_FC[FRONT] * div_vel_FC[FRONT];

                    *tau_Y_FC[i][j][k][FRONT][m] =  term1 - term2;
                    /*__________________________________
                    *   Back Face
                    *   tau_ZZ
                    *___________________________________*/
                    term1           = 2.0       * viscosity_FC[BACK] * (wvel_CC[m][i][j][k] - wvel_CC[m][i][j][k-1])/delZ;
                    term2           = (2.0/3.0) * viscosity_FC[BACK] * div_vel_FC[BACK];

                    *tau_Y_FC[i][j][k][BACK][m] =  term1 - term2;
    #endif
                }
            }
        }
    }
/*______________________________________________________________________
*   DEBUGGING SECTION   
*_______________________________________________________________________*/
#if switchDebug_shear_stress_Zdir
     #define switchInclude_shear_stress_Zdir 1
     #include "debugcode.i"
     #undef switchInclude_shear_stress_Zdir
#endif

/*__________________________________
*   Free the local memory
*___________________________________*/
    free_dvector_nr( div_vel_FC,  1, N_CELL_FACES);
    free_dvector_nr(viscosity_FC, 1, N_CELL_FACES);
/*__________________________________
*   Quite all fullwarn compiler remarks
*___________________________________*/
    delZ    = delZ;
    wvel_CC = wvel_CC;           
}
/*STOP_DOC*/
