/* 
 ======================================================================*/
#include <sys/types.h>
#include <math.h>
#include <assert.h>
#include "functionDeclare.h"
#include "switches.h"
#include "nrutil+.h"
#include "parameters.h"
#include "macros.h"
/*
 Function:  vel_face
 Filename:  press_vel_face.c
 Purpose:
   This function determines the face centered velocitiies around the
   boundaries of the domain
 Reference:
 
 Steps: 
    1)  determine the looping limits
    2)  calculate the viscous terms
    3)  Calculate face centered velocities for the interior nodes
   
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       08/22/99    

 References:
    (1)  
    Casulli, V. and Greenspan, D, Pressure Method for the Numerical Solution
    of Transient, Compressible Fluid Flows, International Journal for Numerical
    Methods in Fluids, Vol. 4, 1001-1012, (1984)
 ---------------------------------------------------------------------  */
void vel_initial_iteration(                 
    int     xLoLimit,                   /* x-array Lower Interior Nodes     */
    int     yLoLimit,                   /* y-array Lower Interior Nodes     */
    int     zLoLimit,                   /* z-array Lower Interior Nodes     */
    int     xHiLimit,                   /* x-array Upper Interior Nodes     */
    int     yHiLimit,                   /* y-array Upper Interior Nodes     */
    int     zHiLimit,                   /* z-array Upper Interior Nodes     */
    int     **BC_types,                 /* defines which boundary conditions*/
                                        /* have been set on each wall       */
    double  ******uvel_FC,              /* u-face-centered velocity         */
                                        /* uvel_FC(x,y,z,face)              */
    double  ******vvel_FC,              /*  v-face-centered velocity        */
                                        /* vvel_FC(x,y,z, face)             */
    double  ******wvel_FC,              /* w face-centered velocity         */
                                        /* wvel_FC(x,y,z,face)              */
    double  ****press_CC,               /* Cell-center pressure             */
    double  ****rho_CC,                 /* Cell-centered density            */
    double  delt,                       /* delta t                          */
    double  ****grav,                   /* Gravity(x,y,z,direction)         */ 
    double  delX,                       /* distance/cell, xdir              */
    double  delY,                       /* distance/cell, ydir              */
    double  delZ,                       /* distance/cell, zdir              */              
    int     nMaterials          )                                                         
{
    int     i, j, k, m,                 /* cell face locators               */
            xLo,    xHi,
            yLo,    yHi,
            zLo,    zHi;          
                             
    double  term1, term2,
            rho_FC_RIGHT,               /* temp variables to ease eq writing*/
            rho_FC_TOP,                 /* and using cvd                    */ 
            ****F,                      /* viscous, convective and body force*/
                                        /* term for the x-momentum eq       */ 
            ****G;                      /* viscous, convective and body force*/
                                        /* term for the y-momentum eq       */         
/*__________________________________
* double check inputs, allocate memory
* and initialize the F and G arrays to 0.0
*___________________________________*/
    assert ( xLoLimit >= 0 && xHiLimit <= X_MAX_LIM);
    assert ( yLoLimit >= 0 && yHiLimit <= Y_MAX_LIM);
    assert ( zLoLimit >= 0 && zHiLimit <= Z_MAX_LIM);
    
    F    = darray_4d(0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM, 1, N_CELL_FACES);
    G    = darray_4d(0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM, 1, N_CELL_FACES);
    for ( k = zLoLimit; k <= zHiLimit; k++)
    {
        for ( j = yLoLimit; j <= yHiLimit; j++)
        {
            for ( i = xLoLimit; i <= xHiLimit; i++)
            {
                F[i][j][k][RIGHT]   = 0.0;
                G[i][j][k][TOP]     = 0.0;
            }
        }
    }
/* _______________________________________________________________________

                                 ________ 
                                /  1    /|
                               /_______/ |
                              |       | ______(3)
                       (4)____| I,J,K |  |     
                              |       | /      
                              |_______|/
                                  |               (6) = back face
                                 (2)              (5) = front face
                                 
    y-dir faces
_______________________________________________________________________ */
    m           = 1;      /* material  HARDWIRED FOR NOW  */
/*______________________________________________________________________
* Set the upper and lower looping indices 
*   If a specified velocity boundary conditions has been set then move
*   the looping limits in toward the center by one cell
*_______________________________________________________________________*/
    xLo = xLoLimit - 1;
    xHi = xHiLimit + 1;
    yLo = yLoLimit - 1;
    yHi = yHiLimit + 1;
    zLo = zLoLimit;
    zHi = zHiLimit;

    if( BC_types[LEFT][VEL_BC]    == DIRICHLET )     xLo = xLoLimit;        
    if( BC_types[RIGHT][VEL_BC]   == DIRICHLET )     xHi = xHiLimit;
        
    if( BC_types[TOP][VEL_BC]     == DIRICHLET )     yHi = yHiLimit;        
    if( BC_types[BOTTOM][VEL_BC]  == DIRICHLET )     yLo = yLoLimit;
 
    if( BC_types[FRONT][VEL_BC]   == DIRICHLET )     zLo = zLoLimit; 
    if( BC_types[BACK][VEL_BC]    == DIRICHLET )     zHi = zHiLimit; 
  
/*__________________________________
*   calculate the viscous terms and the
*   advection terms
*___________________________________*/    
    convective_viscous_terms(
                xLoLimit,       yLoLimit,       zLoLimit,
                xHiLimit,       yHiLimit,       zHiLimit,
                delX,           delY,           delZ,
                uvel_FC,        vvel_FC,        wvel_FC,
                delt,           
                F,              G);
                               
/*__________________________________
*   Step 2) calculate the interior nodes
*   U Velocity.  We only want to change interior nodes
*   eq(22) of (1)
*___________________________________*/

    for ( k = zLo; k <= zHi; k++)
    {
        for ( j = yLo; j <= yHi; j++)
        {
            for ( i = xLo; i <= xHi; i++)
            {   
                rho_FC_RIGHT    = ( rho_CC[m][i][j][k] + rho_CC[m][i+1][j][k] )/2.0;
                term1           = delt * (press_CC[m][i+1][j][k] - press_CC[m][i][j][k] )/
                                    (rho_FC_RIGHT * delX);
                term2           = F[i][j][k][RIGHT];
                *uvel_FC[i][j][k][RIGHT][m]  = *uvel_FC[i][j][k][RIGHT][m] - delt * term2 - term1; 
            }
        }
    }
/*__________________________________
* step 2 Calculate interior nodes
*   V Velocity
*___________________________________*/

    for ( k = zLo; k <= zHi; k++)
    {
        for ( j = yLo; j <= yHi; j++)
        {
            for ( i = xLo; i <= xHi; i++)
            {  
                rho_FC_TOP      = ( rho_CC[m][i][j][k] + rho_CC[m][i][j+1][k] )/2.0;
                term1           = delt * (press_CC[m][i][j+1][k] - press_CC[m][i][j][k] )/
                                    ( rho_FC_TOP * delY);
                term2           = G[i][j][k][TOP];
                *vvel_FC[i][j][k][TOP][m]    = *vvel_FC[i][j][k][TOP][m] - delt * term2 - term1;
            }
        }
    }
    
/*______________________________________________________________________
*   DEBUGGING AND STOP WATCH INFORMATION
*_______________________________________________________________________*/
#if switchDebug_vel_initial_iteration
    fprintf(stderr,"****************************************************************************\n");
    fprintf(stderr,"                        VEL_INITIAL_ITERATION\n");
    fprintf(stderr,"****************************************************************************\n");
    
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
    fprintf(stderr,"****************************************************************************\n");    
    fprintf(stderr,"press return to continue\n");
    getchar();
#endif
    
/*______________________________________________________________________
* Free up the memory
*_______________________________________________________________________*/    
    free_darray_4d(F, 0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM, 1, N_CELL_FACES);    
    free_darray_4d(G, 0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM, 1, N_CELL_FACES);    

}




/* 
 ======================================================================*/
#include <math.h>
#include <assert.h>
#include "functionDeclare.h"
#include "switches.h"
#include "parameters.h"
#include "macros.h"
#include "nrutil+.h"
/*
 Function:  vel_face
 Filename:  press_vel_face.c
 Purpose:
   This function calculates the face centered velocity for n+1 iterations
 
 References:
    Casulli, V. and Greenspan, D, Pressure Method for the Numerical Solution
    of Transient, Compressible Fluid Flows, International Journal for Numerical
    Methods in Fluids, Vol. 4, 1001-1012, (1984)
    
(2) Bulgarelli, U., Casulli, V. and Greenspan, D., "Pressure Methods for the 
    Numerical Solution of Free Surface Fluid Flows, Pineridge Press
    (1984)
      
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       08/24/99    

  
 Prerequisites:
 The functions vel_Face_0_iteration and del
 
 ---------------------------------------------------------------------  */
void vel_Face_n_iteration
             ( 
                
    int     xLoLimit,                   /* x-array Lower Interior Nodes     */
    int     yLoLimit,                   /* y-array Lower Interior Nodes     */
    int     zLoLimit,                   /* z-array Lower Interior Nodes     */
    int     xHiLimit,                   /* x-array Upper Interior Nodes     */
    int     yHiLimit,                   /* y-array Upper Interior Nodes     */
    int     zHiLimit,                   /* z-array Upper Interior Nodes     */
    int     **BC_types,                 /* defines which boundary conditions*/
                                        /* have been set on each wall       */
    double  delX,                       /* distance/cell, xdir              */
    double  delY,                       /* distance/cell, ydir              */
    double  delZ,                       /* distance/cell, zdir              */ 
    double  ******uvel_FC,              /* u-face-centered velocity         */
                                        /* uvel_FC(x,y,z,face,material)     */
    double  ******vvel_FC,              /*  v-face-centered velocity        */
                                        /* vvel_FC(x,y,z, face,material)    */
    double  ******wvel_FC,              /* w face-centered velocity         */
                                        /* wvel_FC(x,y,z,face,material)     */
/*__________________________________
* Face-centered variables half way
* through one iteration
*___________________________________*/
    double  *****uvel_half_FC,          /* u-face-centered velocity         */
                                        /* uvel_half_FC(x,y,z,face,material)*/
    double  *****vvel_half_FC,          /*  v-face-centered velocity        */
                                        /* vvel_half_FC(x,y,z, face,material)*/
    double  *****wvel_half_FC,          /* w face-centered velocity         */
                                        /* wvel_half_FC(x,y,z,face,material)*/ 
    double  ***delPress_CC,             /* Change in the cell-centered press*/
    double  ****press_CC,               /* Cell-center pressure             */
    double  ****rho_CC,                 /* Cell-centered density            */
    double  delt,                       /* delta t                          */
    double  ****grav,                   /* Gravity(x,y,z,direction)         */             
    int     nMaterials          )   
    
    
{
    int     i, j, k, m,                 /* indices                          */            
            xLo,        xHi,
            yLo,        yHi,
            zLo,        zHi,
            press_xLo,  press_xHi,
            press_yLo,  press_yHi,
            press_zLo,  press_zHi,
            
/*`Testing*/             
           *ytesting,
           *xtesting;           /*`*/         /* testing array                    */
    double  
            rho_FC_RIGHT,               /* Temporary variables for calculating*/
            rho_FC_LEFT,                /* face-centered velocities         */
            rho_FC_TOP,                 /* these are nice for debuggine     */
            rho_FC_BOTTOM,
            rho_FC_FRONT,
            rho_FC_BACK,
            u_FC_RIGHT,
            u_FC_LEFT,
            v_FC_TOP,
            v_FC_BOTTOM,
            w_FC_FRONT,
            w_FC_BACK,
            grad_X, grad_Y, grad_Z,
            term1,
            *****test_array,             /* Debugging variables              */
            ****test_array2,                   
            ***test_array3; 
/*__________________________________
*   TESTING 
*   Allocate memory to see where computations are made
*___________________________________*/
    test_array  = darray_5d(0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM, 1, N_CELL_FACES, 1, N_MATERIAL);
    test_array2 = darray_4d(1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
    test_array3 = darray_3d(0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
/*__________________________________
* double check inputs
*___________________________________*/
    assert ( xLoLimit >= 0 && xHiLimit <= X_MAX_LIM);
    assert ( yLoLimit >= 0 && yHiLimit <= Y_MAX_LIM);
    assert ( zLoLimit >= 0 && zHiLimit <= Z_MAX_LIM);
    m = 1;      /* material  HARDWIRED FOR NOW  */
/*`Testing*/     
/*__________________________________
*   Allocate Memory
*___________________________________*/
    ytesting = ivector_nr(0, X_MAX_LIM);
    xtesting = ivector_nr(0,Y_MAX_LIM);
    xtesting[2] = 2;
    xtesting[3] = xHiLimit+1;
    xtesting[4] = xHiLimit+1;
    xtesting[5] = xHiLimit+1;
    
    ytesting[1] = 2;
    ytesting[2] = 2;
    ytesting[3] = yHiLimit+1;
    ytesting[4] = yHiLimit+1;
    ytesting[5] = yHiLimit+1; /*`*/
/* _______________________________________________________________________

                                 ________ 
                                /  1    /|
                               /_______/ |
                              |       | ______(3)
                       (4)____| I,J,K |  |     
                              |       | /      
                              |_______|/
                                  |               (6) = back face
                                 (2)              (5) = front face
                                 
    y-dir faces
_______________________________________________________________________ */
/*______________________________________________________________________
* Set the upper and lower looping indices 
*   If a specified velocity boundary conditions has been set then move
*   the looping limits in toward the center by one cell, similarly for the 
*   pressure.  Note that the computations start in the first layer of 
*   ghost cells surrounding the domain
*_______________________________________________________________________*/

/*`Testing*/
/*__________________________________
*   Testing
*   Should we be using the ghost cells
*   along the floor and ceiling of the domain
*___________________________________*/
    xLo = xLoLimit -1;  press_xLo = xLoLimit -1;
    xHi = xHiLimit +1;  press_xHi = xHiLimit -1;
    yLo = yLoLimit -1;  press_yLo = yLoLimit -1;
    yHi = yHiLimit +1;  press_yHi = yHiLimit +1;
    
    yLo = yLoLimit;  press_yLo = yLoLimit;
    yHi = yHiLimit;  press_yHi = yHiLimit;   
    xLo = xLoLimit -1;  press_xLo = xLoLimit -1;
    xHi = xHiLimit +1;  press_xHi = xHiLimit + 1;    
    
    zLo = zLoLimit;     press_zLo = zLoLimit;
    zHi = zHiLimit;     press_zHi = zHiLimit; /*`*/
    
    if( BC_types[LEFT][VEL_BC]    == DIRICHLET )  xLo = xLoLimit;     
    if( BC_types[RIGHT][VEL_BC]   == DIRICHLET )  xHi = xHiLimit;
           
    if( BC_types[TOP][VEL_BC]     == DIRICHLET )  yHi = yHiLimit;      
    if( BC_types[BOTTOM][VEL_BC]  == DIRICHLET )  yLo = yLoLimit;
    
    if( BC_types[FRONT][VEL_BC]   == DIRICHLET )  zLo = zLoLimit;
    if( BC_types[BACK][VEL_BC]    == DIRICHLET )  zHi = zHiLimit;
    
/*__________________________________
*   TESTING
*   If the Dirichlet BC for the pressure
*   is set then don't updat that pressure??
*___________________________________*/    
    if( BC_types[LEFT][PRESS_BC]  == DIRICHLET )  press_xLo = xLoLimit;
    
/*`Testing*/     
    if( BC_types[LEFT][PRESS_BC]  == DIRICHLET )  press_xLo = xLoLimit-1;    
     /*`*/
    if( BC_types[RIGHT][PRESS_BC] == DIRICHLET )  press_xHi = xHiLimit;
    
    if( BC_types[BOTTOM][PRESS_BC]== DIRICHLET )  press_yLo = yLoLimit;
    if( BC_types[TOP][PRESS_BC]   == DIRICHLET )  press_yHi = yHiLimit;
    
    if( BC_types[FRONT][PRESS_BC] == DIRICHLET )  press_zLo = zLoLimit;
    if( BC_types[BACK][PRESS_BC]  == DIRICHLET )  press_zHi = zHiLimit; 

/*__________________________________
* NOTE you must loop from left to right
*  front to back, and bottom to top
*  see pg 212 of reference (2)
*
*   NEED TO ADD 3D HERE
*___________________________________*/
/*`Testing*/ 
    i = xLo;
    for ( j = ytesting[i]; j <= yHi; j++)
    {
        for ( k = zLo; k <= zHi; k++)
        {
            for ( i = xLo; i <= xtesting[j]; i++)
            { /*`*/
                /*______________________________________________________________________
                *   Step 1) Calculate the delta pressure for both interior or boundary nodes
                *   Update the pressure that is not a boundary condition
                *_______________________________________________________________________*/
                 delPress_CC[i][j][k]   = 0.0;
                 
                     grad_X                  = 0.0;
                     grad_Y                  = 0.0;
                     grad_Z                  = 0.0;
                       
                     grad_X                  = ( *uvel_FC[i][j][k][RIGHT][m]- *uvel_half_FC[i][j][k][LEFT]  )/delX;
#if (N_DIMENSIONS == 2)                 
                     grad_Y                  = ( *vvel_FC[i][j][k][TOP][m]  - *vvel_half_FC[i][j][k][BOTTOM])/delY;
#endif
#if (N_DIMENSIONS == 3)                 
                     grad_Z                  = ( *wvel_FC[i][j][k][FRONT][m]- *wvel_half_FC[i][j][k][BACK]  )/delZ;
#endif
                     term1                   = -2.0*delt * ( 1.0/pow(delX,2) + 1.0/pow(delY,2)  );
                     
                if (    i >= press_xLo && i <= press_xHi 
                     && j >= press_yLo && j <= press_yHi)
                {
                     delPress_CC[i][j][k]    = OVERRELAXATION * (grad_X + grad_Y + grad_Z)/term1; 
                     press_CC[m][i][j][k]    = press_CC[m][i][j][k] + delPress_CC[i][j][k];
                     test_array2[i][j][k][m] = 1.0;
                     test_array3[i][j][k]    = 1.0;
                }
                /*______________________________________________________________________
                *   Step 2) Calculate the intermediate iterative velocities (*_half_FC)
                *   and the full iterative velocities(*_FC).  Note only perform these
                *   calculations where the velocity boundary condition has not been specified
                *   That's why there are conditional statements before each calculation
                *_______________________________________________________________________*/
                /*__________________________________
                *   Left and Right face velocities Eq 3.14
                *___________________________________*/
               
/*`Testing*/                
                /*__________________________________
                *   TESTING
                *___________________________________*/
                
                *uvel_half_FC[i][j][k][RIGHT]   = 0.0;
                *vvel_half_FC[i][j][k][TOP]     = 0.0; /*`*/
                if (  i >= xLo && i <= xHi && j >= yLo && j <= yHi )
                {
                    rho_FC_RIGHT    = ( rho_CC[m][i][j][k] + rho_CC[m][i+1][j][k] )/2.0;
                    u_FC_RIGHT      = *uvel_FC[i][j][k][RIGHT][m];
                    u_FC_RIGHT      = u_FC_RIGHT + (delt/rho_FC_RIGHT) * (delPress_CC[i][j][k]/delX);
                    *uvel_half_FC[i][j][k][RIGHT]   = u_FC_RIGHT;           
                          
                    rho_FC_LEFT     = ( rho_CC[m][i-1][j][k] + rho_CC[m][i][j][k] )/2.0;
                    u_FC_LEFT       = *uvel_half_FC[i][j][k][LEFT];
                    u_FC_LEFT       = u_FC_LEFT - (delt/rho_FC_LEFT) * (delPress_CC[i][j][k]/delX);
                    *uvel_FC[i][j][k][LEFT][m]      = u_FC_LEFT;
                    
                /*__________________________________
                *   Top and bottom face velocites Eq 3.14
                *___________________________________*/
                    rho_FC_TOP       = ( rho_CC[m][i][j][k] + rho_CC[m][i][j+1][k] )/2.0;
                    v_FC_TOP         = *vvel_FC[i][j][k][TOP][m];             
                    v_FC_TOP         = v_FC_TOP + (delt/rho_FC_TOP) * (delPress_CC[i][j][k]/delY);
                    *vvel_half_FC[i][j][k][TOP]     = v_FC_TOP;

           
                    rho_FC_BOTTOM    = ( rho_CC[m][i][j][k] + rho_CC[m][i][j-1][k] )/2.0;
                    v_FC_BOTTOM      = *vvel_half_FC[i][j][k][BOTTOM];             
                    v_FC_BOTTOM      = v_FC_BOTTOM - (delt/rho_FC_BOTTOM) * (delPress_CC[i][j][k]/delY); 
                    *vvel_FC[i][j][k][BOTTOM][m]    = v_FC_BOTTOM;
                    
                    
                    test_array[i][j][k][m][RIGHT]   = 1.0;
                    test_array[i][j][k][m][LEFT]    = 1.0;
                    test_array[i][j][k][m][TOP]     = 1.0;
                    test_array[i][j][k][m][BOTTOM]  = 1.0;
                }
                /*__________________________________
                *   Front and back velocites
                *   UNCOMMENT WHEN RUNNING 3D
                *___________________________________*/

            }
        }
    }
/*`Testing*/  
 /*__________________________________
 *  TESTING
 *___________________________________*/
    free_ivector_nr(ytesting, 0, X_MAX_LIM);
  /*`*/
 
    
/*______________________________________________________________________
*   DEBUGGING
*_______________________________________________________________________*/
#if switchDebug_vel_Face_n_iteration
    fprintf(stderr,"****************************************************************************\n");
    fprintf(stderr,"                        VEL_FACE_N_ITERATION\n");
    fprintf(stderr,"****************************************************************************\n");
   
                
                
     print_5d_where_computations_have_taken_place(      
                         xLoLimit,           yLoLimit,       zLoLimit,
                         xHiLimit,           yHiLimit,       zHiLimit,
                         TOP,                LEFT,
                         m,                  m,
                        "vel_Face_n_iteration Update Face velocities",     
                        "Computation Locations",             test_array,   2);
                        

     print_4d_where_computations_have_taken_place(      
                         xLoLimit,           yLoLimit,       zLoLimit,
                         xHiLimit,           yHiLimit,       zHiLimit,
                         m,                  m,
                        "vel_Face_n_iteration Update Pressure",     
                        "Computation Locations",             test_array2,   2);
                        
      print_3d_where_computations_have_taken_place(      
                        xLoLimit,           yLoLimit,       zLoLimit,
                        xHiLimit,           yHiLimit,       zHiLimit,
                        "vel_Face_n_iteration Update delPress",     
                        "Computation Locations",             test_array3,   2);                      
                               
    printData_6d(       xLo,                yLo,       zLo,
                        xHi,                yHi,       zHi,
                        RIGHT,              LEFT,
                        m,                  m,
                       "vel_Face_n_iteration",     
                       "Uvel_FC",           uvel_FC,        0);

                        

    
    printData_6d(       xLo,                yLo,       zLo,
                        xHi,                yHi,       zHi,
                        TOP,                BOTTOM,
                        m,                  m,
                       "vel_Face_n_iteration",    
                       "Vvel_FC",           vvel_FC,        0);
    /*__________________________________
    *   Advanced level of debugging
    *___________________________________*/    
    if( switchDebug_vel_Face_n_iteration == 2 )
    {


       printData_5d(     xLoLimit,           yLoLimit,       zLoLimit,
                         xHiLimit,           yHiLimit,       zHiLimit,
                         RIGHT,              LEFT,
                         m,                  m,
                        "vel_Face_n_iteration",     
                        "Uvel_half_FC",      uvel_half_FC,   1,      2);
                        
      printData_5d(      xLoLimit,           yLoLimit,       zLoLimit,
                         xHiLimit,           yHiLimit,       zHiLimit,
                         TOP,                BOTTOM,
                         m,                  m,
                        "vel_Face_n_iteration",     
                        "Vvel_half_FC",      vvel_half_FC,   1,      2);
    }
                       
    printData_4d(       xLo,                yLo,            zLo,
                        xHi,                yHi,            zHi,
                        m,                  m,
                       "vel_Face_n_iteration",     
                       "Press_CC",          press_CC);
                       
    printData_3d(       xLo,                yLo,            zLo,
                        xHi,                yHi,            zHi,
                       "vel_Face_n_iteration",     
                       "delPress_CC",       delPress_CC);
   fprintf(stderr,"****************************************************************************\n");
    
    fprintf(stderr,"press return to continue\n");
    getchar();
#endif


/*__________________________________
*   TESTIN
*   Freeup memory
*___________________________________*/
        free_darray_5d( test_array, 0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM, 1, N_CELL_FACES, 1, N_MATERIAL);
        free_darray_4d( test_array2,1, N_MATERIAL,  0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
        free_darray_3d( test_array3,0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
}
