/* 
 ======================================================================*/
#include <time.h>
#include <math.h>
#include <assert.h>
#include "functionDeclare.h"
#include "switches.h"
#include "nrutil+.h"
#include "parameters.h"
/*
 Function:  vel_face
 Filename:  vel_face.c
 Purpose:
   This function calculates the face centered velocity 
   
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       02/22/99    

 IN args/commons         Units      Description
 ---------------         -----      ----------- 
  needed
  
 Prerequisites:
 
 ---------------------------------------------------------------------  */
void vel_Face(  
        int     xLoLimit,               /* x-array lower limit              */
        int     yLoLimit,               /* y-array lower limit              */
        int     zLoLimit,               /* z-array lower limit              */
        int     xHiLimit,               /* x-array upper limit              */
        int     yHiLimit,               /* y-array upper limit              */
        int     zHiLimit,               /* z-array upper limit              */
        double  delX,                   /* distance/cell, xdir              */
        double  delY,                   /* distance/cell, ydir              */
        double  delZ,                   /* distance/cell, zdir              */
        double  delt,                   /* delta t                          */
        double  ****press_L_CC,         /* Cell-center lagrangian pressure  */
        double  ****rho_CC,             /* Cell-centered density            */        
        double  ****grav,               /* Gravity(x,y,z,direction)         */
        double  ****uvel_CC,            /* u-cell-centered velocity         */
                                        /* uvel_CC(x,y,z, material)         */
        double  ****vvel_CC,            /*  v-cell-centered velocity        */
                                        /* vvel_CC(x,y,z, material)         */
        double  ****wvel_CC,            /* w cell-centered velocity         */
                                        /* wvel_CC(x,y,z,material)          */
        double  ******uvel_FC,          /* u-face-centered velocity         */
                                        /* uvel_FC(x,y,z,face,material)     */
        double  ******vvel_FC,          /*  v-face-centered velocity        */
                                        /* vvel_FC(x,y,z, face,material)    */
        double  ******wvel_FC,          /* w face-centered velocity         */
                                        /* wvel_FC(x,y,z,face,material)     */
        int     nMaterials      )

{
    int i, j, k, f,m,                   /* cell face locators               */            
        cell;                           /* variables that change in formula */
    
    double  term1, term2, term3,        /* temp symbols to represent terms  */
            A,                          /* temporary variables              */   
            *grad_P;                    /* gradient of the pressure at each */
                                        /* cell-face                        */
#if sw_vel_face    
    time_t start,secs;                  /* timing variables                */
    start = time(NULL);
#endif 
/*__________________________________
*  Allocate memory for grad array
*___________________________________*/
    grad_P = dvector(1,N_CELL_FACES);
/*__________________________________
* double check inputs
*___________________________________*/
    assert ( xLoLimit > 0 && xHiLimit < X_MAX_LIM);
    assert ( yLoLimit > 0 && yHiLimit < Y_MAX_LIM);
    assert ( zLoLimit > 0 && zHiLimit < Z_MAX_LIM);

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
    m = nMaterials;      /* material  HARDWIRED FOR NOW  */
/*__________________________________
* top bottom faces
*___________________________________*/
   
    for ( i = xLoLimit; i <= xHiLimit; i++)
    {
        for ( j = yLoLimit; j <= yHiLimit; j++)
        {
            for ( k = zLoLimit; k <= zHiLimit; k++)
            {   
                cell   = j+1;            
  
                for(f = TOP; f <= BOTTOM; f++)
                {
                    A = (rho_CC[i][cell][k][m]    + rho_CC[i][j][k][m]);
                    assert(A >= SMALL_NUM);     /* bullet proofing      */
                    /*__________________________________
                    * interpolation to the face
                    *___________________________________*/            
                    term1 = (rho_CC[i][cell][k][m]    * vvel_CC[i][cell][k][m] 
                         +   rho_CC[i][j][k][m]       * vvel_CC[i][j][k][m])/ A;
                       
                    /*__________________________________
                    * pressure correction
                    *___________________________________*/
#if press_correction_vel_FC
                    grad_FC_Ydir(i, j, k, m, press_L_CC, delY, grad_P);   
                    term2 =  2.0 * delt * grad_P[f]/ A;
#else 
                    term2 = 0.0;
#endif 
                    /*__________________________________
                    * gravity term
                    *___________________________________*/
                    term3 =  delt * grav[i][j][k][2]; 
                                   
                    *uvel_FC[i][j][k][f][m] = 0.0;
                    *vvel_FC[i][j][k][f][m] = term1 - term2 + term3;
                    *wvel_FC[i][j][k][f][m] = 0.0;
                    /*__________________________________
                    * change cell index and signs on 
                    * pressure terms
                    *___________________________________*/
                    cell = j-1;
                }

            }
        }
    }
    
 
/*__________________________________
* left right faces
*___________________________________*/         
    for ( i = xLoLimit; i <= xHiLimit; i++)
    {
        for ( j = yLoLimit; j <= yHiLimit; j++)
        {
            for ( k = zLoLimit; k <= zHiLimit; k++)
            {   
                cell   = i+1;       
 
                for(f = RIGHT; f <= LEFT; f++)
                {
                    A = (rho_CC[cell][j][k][m]    + rho_CC[i][j][k][m]);
                    assert(A >= SMALL_NUM);     /* bullet proofing      */
                    /*__________________________________
                    * interpolation to the face
                    *___________________________________*/            
                    term1 = (rho_CC[cell][j][k][m]    * uvel_CC[cell][j][k][m] 
                         +  rho_CC[i][j][k][m]       * uvel_CC[i][j][k][m])/ A;
                            
                    /*__________________________________
                    * pressure correction
                    *___________________________________*/
#if press_correction_vel_FC
                    grad_FC_Xdir(i, j, k, m, press_L_CC, delX, grad_P);                        
                    term2 =  2.0 * delt * grad_P[f]/A;
#else 
                    term2 = 0.0;
#endif 
                    /*__________________________________
                    * gravity term
                    *___________________________________*/
                    term3 =  delt * grav[i][j][k][1]; 
                                   
                    *uvel_FC[i][j][k][f][m] = term1 - term2 + term3;
                    *vvel_FC[i][j][k][f][m] = 0.0;
                    *wvel_FC[i][j][k][f][m] = 0.0;
                    /*__________________________________
                    * change cell index and signs on 
                    * pressure terms
                    *___________________________________*/
                    cell = i-1;

                }

            }
        }
    }
/*__________________________________
* front and back faces
*___________________________________*/         
    for ( i = xLoLimit; i <= xHiLimit; i++)
    {
        for ( j = yLoLimit; j <= yHiLimit; j++)
        {
            for ( k = zLoLimit; k <= zHiLimit; k++)
            {   
                cell   = k+1;            
 
                for(f = FRONT; f <= BACK; f++)
                {
                    A = (rho_CC[i][j][cell][m]    + rho_CC[i][j][k][m]);
                    assert(A >= SMALL_NUM);     /* bullet proofing      */
                    /*__________________________________
                    * interpolation to the face
                    *___________________________________*/            
                    term1 = (rho_CC[i][j][cell][m]    * wvel_CC[i][j][cell][m] 
                         +   rho_CC[i][j][k][m]       * wvel_CC[i][j][k][m])/ A;

                    /*__________________________________
                    * pressure correction
                    *___________________________________*/
#if press_correction_vel_FC
                    grad_FC_Zdir(i, j, k, m, press_L_CC, delZ, grad_P);                        
                    term2 =  2.0 * delt * grad_P[f]/ A;
#else 
                    term2 = 0.0;
#endif 
                    /*__________________________________
                    * gravity term
                    *___________________________________*/
                    term3 =  delt * grav[i][j][k][3]; 
                                   
                    *uvel_FC[i][j][k][f][m] = 0.0;
                    *vvel_FC[i][j][k][f][m] = 0.0;
                    *wvel_FC[i][j][k][f][m] = term1 - term2 + term3;
                    /*__________________________________
                    * change cell index and signs on 
                    * pressure terms
                    *___________________________________*/
                    cell = k-1;
 
                }

            }
        }
    }
/*______________________________________________________________________
*   DEBUGGING AND STOP WATCH INFORMATION
*   CHANGES_NEEDED  Update the printData_FC_MV to 6d *vel_FC printData
*_______________________________________________________________________*/
/*__________________________________
* No printout debugging and timing info
*___________________________________*/ 
#if switchDebug_vel_face
/*     printData_FC_MF(   xLoLimit,       yLoLimit,       zLoLimit,
                       xHiLimit,       yHiLimit,       zHiLimit,
                       "vel_face",     "u face velocity", uvel_FC, m);
                       
    printData_FC_MF(   xLoLimit,       yLoLimit,       zLoLimit,
                       xHiLimit,       yHiLimit,       zHiLimit,
                       "vel_face",     "v face velocity", vvel_FC, m);
                       
    printData_FC_MF(   xLoLimit,       yLoLimit,       zLoLimit,
                       xHiLimit,       yHiLimit,       zHiLimit,
                       "vel_face",     "w face velocity", wvel_FC, m); */
#endif 
       
#if sw_vel_face
    stopwatch("vel_face",start);
#endif

/*__________________________________
* Free up memory
*___________________________________*/
    free_dvector(grad_P, 1, N_CELL_FACES);
}


/* 
 ======================================================================*/
#include <time.h>
#include <math.h>
#include <assert.h>
#include "functionDeclare.h"
#include "switches.h"
#include "parameters.h"
#include "macros.h"
/*
 Function:  vel_Face_before_iterative_pressure_solver
 Filename:  vel_face.c
 Purpose:
   This function calculates the face centered velocity BEFORE the
   iterative pressure solver is called.  This function takes cell-centered
   data and interpolates it out to the faces
   
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       02/22/99    

 IN args/commons         Units      Description
 ---------------         -----      ----------- 
  needed
  
 ---------------------------------------------------------------------  */
void vel_Face_before_iterative_pressure_solver(  
        int     xLoLimit,               /* x-array lower limit              */
        int     yLoLimit,               /* y-array lower limit              */
        int     zLoLimit,               /* z-array lower limit              */
        int     xHiLimit,               /* x-array upper limit              */
        int     yHiLimit,               /* y-array upper limit              */
        int     zHiLimit,               /* z-array upper limit              */
        double  ****rho_CC,             /* Cell-centered density            */
        double  ****uvel_CC,            /* u-cell-centered velocity         */
                                        /* uvel_CC(x,y,z, material)         */
        double  ****vvel_CC,            /*  v-cell-centered velocity        */
                                        /* vvel_CC(x,y,z, material)         */
        double  ****wvel_CC,            /* w cell-centered velocity         */
                                        /* wvel_CC(x,y,z,material)          */
 
        double  ******uvel_FC,          /* u-face-centered velocity         */
                                        /* uvel_FC(x,y,z,face,material)     */
        double  ******vvel_FC,          /*  v-face-centered velocity        */
                                        /* vvel_FC(x,y,z, face,material)    */
        double  ******wvel_FC,          /* w face-centered velocity         */
                                        /* wvel_FC(x,y,z,face,material)     */
        int     nMaterials          )

{
    int i, j, k, f,m,                   /* cell face locators               */            
        cell,                           /* variables that change in formula */
        xLo, yLo, zLo,
        xHi, yHi, zHi;
    
    double  term1,                      /* temp symbols to represent terms  */
            A;                          /* temporary variables              */
            
/*__________________________________
* PLOTTING VARIABLES
*___________________________________*/
#if switchDebug_vel_Face_before_iterative_pressure_solver
    double delX, delY;
    #include "plot_declare_vars.h" 
#endif 

#if switchsw_vel_Face_before_iterative_pressure_solver       
    time_t start,secs;                  /* timing variables                */
    start = time(NULL); 
#endif


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
    m = nMaterials;      /* material  HARDWIRED FOR NOW  */
    
/*__________________________________
*   Figure the upper and lower indices
*   so that you don't reach outside 
*   of the ghostcells
*___________________________________*/
    xLo = GC_LO(xLoLimit) + 1;
    yLo = GC_LO(yLoLimit) + 1;
    zLo = GC_LO(zLoLimit) + 1;
    xHi = GC_HI(xHiLimit) - 1;   
    yHi = GC_HI(yHiLimit) - 1;
    zHi = GC_HI(zHiLimit) - 1;

/*__________________________________
*   Testing to see if I'm going out of bounds
*___________________________________*/
/*     xLo = xLoLimit;
    yLo = yLoLimit;

    xHi = xHiLimit;
    yHi = yHiLimit; */
    
    zLo = zLoLimit;
    zHi = zHiLimit;
/*__________________________________
* double check inputs
*___________________________________*/
    assert ( xLo > 0 && xHi < X_MAX_LIM);
    assert ( yLo > 0 && yHi < Y_MAX_LIM);
    assert ( zLo > 0 && zHi < Z_MAX_LIM);
/*__________________________________
* top bottom faces
*___________________________________*/
   
    for ( k = zLo; k <= zHi; k++)
    {
        for ( j = yLo; j <= yHi; j++)
        {
            for ( i = xLo; i <= xHi; i++)
            {   
                cell   = j+1;            
  
                for(f = TOP; f <= BOTTOM; f++)
                {
                    A = (rho_CC[i][cell][k][m]    + rho_CC[i][j][k][m]);
                    
                    A = 2.0;
                    
                    assert(A >= SMALL_NUM);     /* bullet proofing      */
                    /*__________________________________
                    * interpolation to the face
                    *___________________________________*/
           
                     term1  = (rho_CC[i][cell][k][m]    * vvel_CC[i][cell][k][m] 
                            +   rho_CC[i][j][k][m]      * vvel_CC[i][j][k][m])/ A;
                                   
                    *uvel_FC[i][j][k][f][m] = 0.0;
                    *vvel_FC[i][j][k][f][m] = term1;                  
                    *wvel_FC[i][j][k][f][m] = 0.0; 
                    /*__________________________________
                    * change cell index and signs on 
                    * pressure terms
                    *___________________________________*/
                    cell = j-1;
                }

            }
        }
    }
    
 
/*__________________________________
* left right faces
*___________________________________*/         
    for ( k = zLo; k <= zHi; k++)
    {
        for ( j = yLo; j <= yHi; j++)
        {
            for ( i = xLo; i <= xHi; i++)
            {   
                cell   = i+1;       
 
                for(f = RIGHT; f <= LEFT; f++)
                {
                    A = (rho_CC[cell][j][k][m]    + rho_CC[i][j][k][m]);
                                      
                    assert(A >= SMALL_NUM);     /* bullet proofing      */
                    /*__________________________________
                    * interpolation to the face
                    *___________________________________*/ ;           
                    term1   = (((rho_CC[cell][j][k][m]    * uvel_CC[cell][j][k][m] )
                            +  (rho_CC[i][j][k][m]       * uvel_CC[i][j][k][m]))/ A);

                    *uvel_FC[i][j][k][f][m] = term1;
                    *vvel_FC[i][j][k][f][m] = 0.0;
                    *wvel_FC[i][j][k][f][m] = 0.0; 
                    /*__________________________________
                    * change cell index and signs on 
                    * pressure terms
                    *___________________________________*/
                    cell = i-1;

                }

            }
        }
    }
/*__________________________________
* front and back faces
*___________________________________*/         
    for ( k = zLo; k <= zHi; k++)
    {
        for ( j = yLo; j <= yHi; j++)
        {
            for ( i = xLo; i <= xHi; i++)
            {   
                cell   = k+1;            
 
                for(f = FRONT; f <= BACK; f++)
                {
                    A = (rho_CC[i][j][cell][m]    + rho_CC[i][j][k][m]);
                    assert(A >= SMALL_NUM);     /* bullet proofing          */        
                    /*__________________________________
                    * interpolation to the face
                    *___________________________________*/
            
                    term1 = (rho_CC[i][j][cell][m]    * wvel_CC[i][j][cell][m] 
                         +   rho_CC[i][j][k][m]       * wvel_CC[i][j][k][m])/ A;
                                   
                    *uvel_FC[i][j][k][f][m] = 0.0;
                    *vvel_FC[i][j][k][f][m] = 0.0;
                    *wvel_FC[i][j][k][f][m] = term1; 
                    /*__________________________________
                    * change cell index and signs on 
                    * pressure terms
                    *___________________________________*/
                    cell = k-1;
 
                }

            }
        }
    }
/*______________________________________________________________________
*   DEBUGGING AND STOP WATCH INFORMATION
*_______________________________________________________________________*/
#if switchDebug_vel_Face_before_iterative_pressure_solver

    #define switchInclude_vel_Face_before_iterative_pressure_solver 1
        #include "debugcode.i"
    #undef switchInclude_vel_Face_before_iterative_pressure_solver
    
    
/*     printData_6d(       xLo,            yLo,      zLo,
                        xHi,            yHi,      zHi,
                        RIGHT,          LEFT,
                        m,              m,
                        "vel_Face_before_iterative_pressure_solver",     
                        "Uvel_FC",      uvel_FC,        0);
    
    printData_6d(       xLo,            yLo,      zLo,
                        xHi,            yHi,      zHi,
                        TOP,            BOTTOM,
                        m,              m,
                        "vel_Face_before_iterative_pressure_solver",     
                        "Vvel_FC",  vvel_FC,            0);  */
#endif      
#if switchsw_vel_Face_before_iterative_pressure_solver
    stopwatch("vel_face",start);
#endif

}
