 /* 
 ======================================================================*/
#include <math.h>
#include <assert.h>
#include "switches.h"
#include "parameters.h"
#include "functionDeclare.h"
#include "macros.h"
#include "nrutil+.h"
/*
 Function:  set_Neumann_BC_FC--BOUNDARY CONDITIONS: Sets Neumann BC for all face-centered variables.
 Filename:  boundary_cond_FC.c
 Purpose:
            This function sets the face centered values along the inner and
            outer perimeter of the ghostcells for each dependent variable.
            Only do this if the dependent variable is "FLOAT" throughout the
            computation. 
  Steps:
    1) Find the looping indices for the problem
    2) Test to see if you should be in this function
    3) Loop over all of the walls (minus the corner ghostcells)
            - Determine what the upper an lower indices are for that wall
            - Set the face-centered data_FC = data_CC
    4) Set the face-centered BC in each of the corner ghostcells

 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       12/8/99    

Note: You must call "set_Neumann_BC" BEFORE this function.  Specifically,
this function needs an updated data_CC.
 ---------------------------------------------------------------------  */
 
 void set_Neumann_BC_FC( 
              
    int     xLoLimit,                   /* x-array lower limit              */
    int     yLoLimit,                   /* y-array lower limit              */
    int     zLoLimit,                   /* z-array lower limit              */
    int     xHiLimit,                   /* x-array upper limit              */
    int     yHiLimit,                   /* y-array upper limit              */
    int     zHiLimit,                   /* z-array upper limit              */
    double  ****data_CC,                /* cell-centered data               (INPUT) */    
    double  ******data_FC,              /* u-face-centered velocity         (IN/OUT)*/
    int     var,                        /* TEMP, PRESS, UVEL                (INPUT) */                    

    int     ***BC_types,                /* defines which boundary conditions(INPUT) */
                                        /* have been set on each wall               */
    int     ***BC_float_or_FLOAT,       /* BC_float_or_FLOAT[wall][var][m]  (INPUT)*/
                                        /* Variable on boundary is either   */
                                        /* FLOAT or it floats during the    */
                                        /* computation                      */
    int     nMaterials        )

 {
    int     i,j,k,f,m,                  /* indices                           */
            wall,      
            xLo,        xHi,         
            yLo,        yHi, 
            zLo,        zHi,
            wallLo,     wallHi,
            faceLo,     faceHi, 
            should_I_leave;
            m = nMaterials;             /* Hardwired                        */
/*__________________________________
* double check inputs
*___________________________________*/
    assert ( xLoLimit >= 0 && xHiLimit <= X_MAX_LIM);
    assert ( yLoLimit >= 0 && yHiLimit <= Y_MAX_LIM);
    assert ( zLoLimit >= 0 && zHiLimit <= Z_MAX_LIM);

/*__________________________________
*   Step 1)
*   Determine the looping indices
*   for multidimensional problems
*___________________________________*/
#if (N_DIMENSIONS == 1) 
        wallLo = LEFT;  wallHi = RIGHT;
        faceLo = LEFT;  faceHi = RIGHT;
#endif
#if (N_DIMENSIONS == 2) 
        wallLo = TOP;   wallHi = LEFT;
        faceLo = TOP;   faceHi = LEFT;
#endif
#if (N_DIMENSIONS == 3) 
        wallLo = TOP;   wallHi = BACK;
        faceLo = TOP;   faceHi = BACK;
#endif
/*__________________________________
*   Step 2)
*   Test to see if you should be in this function
*___________________________________*/
    should_I_leave = YES;
    for(m = 1; m <= nMaterials; m++)
    {    
        for( wall = wallLo; wall <= wallHi; wall ++)
        {
            if(( BC_types[wall][var][m]            == NEUMANN ||
                 BC_types[wall][var][m]            == PERIODIC ) &&
                 BC_float_or_FLOAT[wall][var][m]   == FLOAT) should_I_leave = NO;
        }
    }
    if (should_I_leave == YES) return;
/*______________________________________________________________________
*   Step 3)
*   Now loop over all materials and walls and set the appropriate BC
*   whether its velocity or pressure
*   You need to find the appropriate looping limits for each wall
*_______________________________________________________________________*/
    for(m = 1; m <= nMaterials; m++)
    {            
        for( wall = wallLo; wall <= wallHi; wall ++)
        {

             find_loop_index_limits_at_domain_edges(                
                        xLoLimit,                  yLoLimit,                   zLoLimit,
                        xHiLimit,                  yHiLimit,                   zHiLimit,
                        &xLo,                      &yLo,                       &zLo,
                        &xHi,                      &yHi,                       &zHi,
                        wall    );
           /*__________________________________
           * Finally set the boundary condition
           *___________________________________*/
            for ( k = zLo; k <= zHi; k++)
            {
                for ( j = yLo; j <= yHi; j++)
                { 
                    for ( i = xLo; i <= xHi; i++)
                    {        
                        for(f = faceLo ; f <= faceHi; f ++)
                        {            
                            *data_FC[i][j][k][f][m] = data_CC[m][i][j][k];
                        }
                    }
                }
            } 
        }
    }
/*______________________________________________________________________
    Step 4
*   NOW TAKE CARE OF THE CORNER GHOSTCELLS

        
     --A---------------------------------------A--
    D| 2 |   |   |   |   |   |   |   |   |   | 3 |B  -- top_GC       
     ---------------------------------------------                   
     |   | + | + | + | + | + | + | + | + | + |   |   -- yHiLimit     
     ---------------------------------------------                   
     |   | + | + | + | + | + | + | + | + | + |   |                   
     ---------------------------------------------                   
     |   | + | + | + | + | + | + | + | + | + |   |                   
     ---------------------------------------------                    
     |   | + | + | + | + | + | + | + | + | + |   |   -- yLoLimit     
     ---------------------------------------------                   
    D| 1 |   |   |   |   |   |   |   |   |   | 4 |B  -- bottom_GC    
     --C---------------------------------------C--                   
       |   | xLoLimit             xHiLimit |   |                     
       |                                       |                     
       left_GC                              right_GC  
      
*   NEED TO ADD 3D, specifically the other corner cells on the front face
*_______________________________________________________________________*/
    for(m = 1; m <= nMaterials; m++)
    {   
        for ( k = zLo; k <= zHi; k++)
        {   
            i = GC_LO(xLoLimit); 
            j = GC_LO(yLoLimit);
            *data_FC[i][j][k][BOTTOM][m] = data_CC[m][i][j][k];
            *data_FC[i][j][k][LEFT][m]   = data_CC[m][i][j][k];
            
            i = GC_LO(xLoLimit); 
            j = GC_HI(yHiLimit);
            *data_FC[i][j][k][TOP][m]    = data_CC[m][i][j][k];
            *data_FC[i][j][k][LEFT][m]   = data_CC[m][i][j][k];
            
            i = GC_HI(xHiLimit); 
            j = GC_HI(yHiLimit);
            *data_FC[i][j][k][TOP][m]    = data_CC[m][i][j][k];
            *data_FC[i][j][k][RIGHT][m]  = data_CC[m][i][j][k];
            
            i = GC_HI(xHiLimit); 
            j = GC_LO(yLoLimit);
            *data_FC[i][j][k][BOTTOM][m] = data_CC[m][i][j][k];
            *data_FC[i][j][k][RIGHT][m]  = data_CC[m][i][j][k];
        }
    } 
/*______________________________________________________________________
*   DEBUGGING INFORMATION
*_______________________________________________________________________*/ 
#if switchDebug_set_Neumann_BC_FC
    fprintf(stderr,"****************************************************************************\n");
    fprintf(stderr,"                        SET_Neumann_BC_FC\n");
    fprintf(stderr,"****************************************************************************\n");         
    for (m = 1; m <= nMaterials; m++)
    {         
        fprintf(stderr,"\t Material %i \n",m);         
        printData_6d(       GC_LO(xLoLimit),    GC_LO(yLoLimit),       (zLoLimit),
                            GC_HI(xHiLimit),    GC_HI(yHiLimit),       (zHiLimit),
                            TOP,                LEFT,
                            m,                  m,
                            "set_Neumann_BC_FC",     
                           "data_FC with ghost cells",                  data_FC,        0);
    }
    fprintf(stderr,"****************************************************************************\n");         
    
    fprintf(stderr,"press return to continue\n");
    getchar();            
#endif

 }
/*STOP_DOC*/


 /* 
 ======================================================================*/
#include <math.h>
#include <assert.h>
#include "switches.h"
#include "parameters.h"
#include "functionDeclare.h"
#include "macros.h"
#include "nrutil+.h"
/*
 Function:  set_Dirichlet_BC_FC--BOUNDARY CONDITIONS: Set Dirichlet BC for all fac-centered variables.
 Filename:  boundary_cond_FC.c
 Purpose:
            This function sets the face centered values along the inner and
            outer perimeter of the ghostcells for each dependent variable.
            Only do this if the dependent variable is "fixed" throughout the
            computation. 
  Steps:
    1) Find the looping indices for the problem
    2) Test to see if you should be in this function
    3) Loop over all of the walls (minus the corner ghostcells)
            - Determine what the upper an lower indices are for that wall
            - Set the face-centered data_FC = BC_Values[wall][*][m]
    4) Set the face-centered BC in each of the corner ghostcells

 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       08/22/99    

 ---------------------------------------------------------------------  */
 
 void set_Dirichlet_BC_FC( 
              
    int     xLoLimit,                   /* x-array lower limit              */
    int     yLoLimit,                   /* y-array lower limit              */
    int     zLoLimit,                   /* z-array lower limit              */
    int     xHiLimit,                   /* x-array upper limit              */
    int     yHiLimit,                   /* y-array upper limit              */
    int     zHiLimit,                   /* z-array upper limit              */
                /*---------Face Centered Values-----[*]_FC(x,y,z,face, material)----*/
    double  ******data_FC,              /* face-centered data               (IN/OUT)*/
    int     var,                        /* used to designate whether        (INPUT) */
                                        /* the input array is UVEL,TEMP.... */                   
    int     ***BC_types,                /* defines which boundary conditions(INPUT) */
                                        /* have been set on each wall               */
    double  ***BC_Values,               /* BC values BC_values[wall][variable(INPUT)*/
    int     ***BC_float_or_fixed,       /* BC_float_or_fixed[wall][var][m]   (INPUT)*/
                                        /* Variable on boundary is either   */
                                        /* fixed or it floats during the    */
                                        /* compuation                       */
    int     nMaterials        )

 {
    int     i,j,k,f,m,                  /* indices                           */
            wall,       
            xLo,        xHi,         
            yLo,        yHi, 
            zLo,        zHi,
            wallLo,     wallHi,
            faceLo,     faceHi,     
            should_I_leave;
            m = nMaterials;             /* Hardwired                        */
/*__________________________________
* double check inputs
*___________________________________*/
    assert ( xLoLimit >= 0 && xHiLimit <= X_MAX_LIM);
    assert ( yLoLimit >= 0 && yHiLimit <= Y_MAX_LIM);
    assert ( zLoLimit >= 0 && zHiLimit <= Z_MAX_LIM);
/*__________________________________
*   Step 1)
*   Determine the looping indices
*   for multidimensional problems
*___________________________________*/
#if (N_DIMENSIONS == 1) 
        wallLo = LEFT;  wallHi = RIGHT;
        faceLo = LEFT;  faceHi = RIGHT;
#endif
#if (N_DIMENSIONS == 2) 
        wallLo = TOP;   wallHi = LEFT;
        faceLo = TOP;   faceHi = LEFT;
#endif
#if (N_DIMENSIONS == 3) 
        wallLo = TOP;   wallHi = BACK;
        faceLo = TOP;   faceHi = BACK;
#endif
/*__________________________________
*   Step 2)
*   Test to see if you should be in this function
*___________________________________*/
    should_I_leave = YES;
    for(m = 1; m <= nMaterials; m++)
    {    
        for( wall = wallLo; wall <= wallHi; wall ++)
        {
            if(BC_types[wall][var][m]               == DIRICHLET &&
               BC_float_or_fixed[wall][var][m]      == FIXED) should_I_leave = NO;
        }
    }
    if (should_I_leave == YES) return;
/*______________________________________________________________________
*   Step 3)
*   Now loop over all materials and walls and set the appropriate BC
*   whether its velocity or pressure
*   You need to find what the appropriate looping limits are for each 
*   ghostcells layers for each wall.
*_______________________________________________________________________*/
    for(m = 1; m <= nMaterials; m++)
    {            
        for( wall = wallLo; wall <= wallHi; wall ++)
        {                                     
             find_loop_index_limits_at_domain_edges(
                        xLoLimit,                  yLoLimit,                   zLoLimit,
                        xHiLimit,                  yHiLimit,                   zHiLimit,
                        &xLo,                      &yLo,                       &zLo,
                        &xHi,                      &yHi,                       &zHi,
                        wall    );
           /*__________________________________
           * Finally set the boundary condition
           *___________________________________*/

            for ( k = zLo; k <= zHi; k++)
            {
                for ( j = yLo; j <= yHi; j++)
                {
                    for ( i = xLo; i <= xHi; i++)
                    {
                        for(f = faceLo ; f <= faceHi; f ++)
                        {
                            *data_FC[i][j][k][f][m] = BC_Values[wall][var][m];

                        }
                    }
                }
            }
        }
    }
/*______________________________________________________________________
    Step 4
*   NOW TAKE CARE OF THE CORNER GHOSTCELLS

        
     --A---------------------------------------A--
    D| 2 |   |   |   |   |   |   |   |   |   | 3 |B  -- top_GC       
     ---------------------------------------------                   
     |   | + | + | + | + | + | + | + | + | + |   |   -- yHiLimit     
     ---------------------------------------------                   
     |   | + | + | + | + | + | + | + | + | + |   |                   
     ---------------------------------------------                   
     |   | + | + | + | + | + | + | + | + | + |   |                   
     ---------------------------------------------                    
     |   | + | + | + | + | + | + | + | + | + |   |   -- yLoLimit     
     ---------------------------------------------                   
    D| 1 |   |   |   |   |   |   |   |   |   | 4 |B  -- bottom_GC    
     --C---------------------------------------C--                   
       |   | xLoLimit             xHiLimit |   |                     
       |                                       |                     
       left_GC                              right_GC  
       
    Face A =    BC_values[TOP][*][m]
    Face B =    BC_values[RIGHT][*][m]
    Face C =    BC_values[BOTTOM][*][m]
    Face D =    BC_values[LEFT][*][m]   
      
*   NEED TO ADD 3D
*_______________________________________________________________________*/
    for(m = 1; m <= nMaterials; m++)
    {   
        for ( k = zLo; k <= zHi; k++)
        {      
            *data_FC[GC_LO(xLoLimit)][GC_LO(yLoLimit)][k][BOTTOM][m] = BC_Values[BOTTOM][var][m];
            *data_FC[GC_LO(xLoLimit)][GC_LO(yLoLimit)][k][LEFT][m]   = BC_Values[LEFT][var][m];

            *data_FC[GC_LO(xLoLimit)][GC_HI(yHiLimit)][k][TOP][m]    = BC_Values[TOP][var][m];
            *data_FC[GC_LO(xLoLimit)][GC_HI(yHiLimit)][k][LEFT][m]   = BC_Values[LEFT][var][m]; 

            *data_FC[GC_HI(xHiLimit)][GC_HI(yHiLimit)][k][TOP][m]    = BC_Values[TOP][var][m];
            *data_FC[GC_HI(xHiLimit)][GC_HI(yHiLimit)][k][RIGHT][m]  = BC_Values[RIGHT][var][m];      

            *data_FC[GC_HI(xHiLimit)][GC_LO(yLoLimit)][k][BOTTOM][m] = BC_Values[BOTTOM][var][m];
            *data_FC[GC_HI(xHiLimit)][GC_LO(yLoLimit)][k][RIGHT][m]  = BC_Values[RIGHT][var][m];
        }
    }         
/*______________________________________________________________________
*   DEBUGGING INFORMATION
*_______________________________________________________________________*/ 
#if switchDebug_set_Dirichlet_BC_FC
    fprintf(stderr,"****************************************************************************\n");
    fprintf(stderr,"                        SET_DIRICHLET_BC_FC\n");
    fprintf(stderr,"****************************************************************************\n");         
    for (m = 1; m <= nMaterials; m++)
    {         
        fprintf(stderr,"\t Material %i \n",m);          
         
        printData_6d(       GC_LO(xLoLimit),    GC_LO(yLoLimit),       (zLoLimit),
                            GC_HI(xHiLimit),    GC_HI(yHiLimit),       (zHiLimit),
                            TOP,                LEFT,
                            m,                  m,
                            "set_Dirichlet_BC_FC",     
                           "data_FC with ghost cells",                  data_FC,        0);
    }
    fprintf(stderr,"****************************************************************************\n");         
    
    fprintf(stderr,"press return to continue\n");
    getchar();            
#endif

 }
/*STOP_DOC*/ 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 /*______________________________________________________________________
 *  Below is old code that I'm not ready to throwout
 *_______________________________________________________________________*/
 
 
 
 
 
 
 
 #if 0      /*DONT COMPILE FOR NOW 12/4/99*/
 /* 
 ======================================================================*/
#include <math.h>
#include <assert.h>
#include "switches.h"
#include "parameters.h"
#include "functionDeclare.h"
#include "macros.h"
/*
set_Wall_BC_FC
boundary_cond.c
 Purpose:
            This function sets the velocity boundary conditions along the walls
            of the computational domain. For each wall the following is set
    
   Right Wall      uvel_FC[xHiLimit,j,k,RIGHT,m]   = 0
                   vvel_FC(xHiLimit+1,j,k,TOP,m]   = -vvel_FC(xHiLimit,j,k,TOP,m]
                   vvel_FC(xHiLimit+1,j,k,BOTTOM,m]= -vvel_FC(xHiLimit,j,k,BOTTOM,m]

   Left Wall       uvel_FC[xLoLimit,j,k,RIGHT,m]   = 0
                   vvel_FC(xLoLimit-1,j,k,TOP,m]   = -vvel_FC(xLoLimit,j,k,TOP,m]
                   vvel_FC(xLoLimit-1,j,k,BOTTOM,m]= -vvel_FC(xLoLimit,j,k,BOTTOM,m]

   BOTTOM Wall     vvel_FC[yLoLimit,j,k,BOTTOM,m]  = 0
                   uvel_FC(i,yLoLimit-1,k,LEFT,m]  = -uvel_FC(i,yLoLimit,k,LEFT,m]
                   uvel_FC(i,yLoLimit-1,k,RIGHT,m] = -uvel_FC(i,yLoLimit,k,RIGHT,m]

   TOP Wall        vvel_FC[yHiLimit,j,k,BOTTOM,m]  = 0
                   uvel_FC(i,yHiLimit+1,k,LEFT,m]  = -uvel_FC(i,yHiLimit,k,LEFT,m]
                   uvel_FC(i,yHiLimit+1,k,RIGHT,m] = -uvel_FC(i,yHiLimit,k,RIGHT,m]

References: 
    This is from 
    Bulgarelli, U., Casulli, V. and Greenspan, D., "Pressure Methods for the 
    Numerical Solution of Free Surface Fluid Flows, Pineridge Press
    (1984) pg 100
    
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       08/22/99    

 ---------------------------------------------------------------------  */
 
 void set_Wall_BC_FC( 
              
        int     xLoLimit,               /* x-array lower limit              */
        int     yLoLimit,               /* y-array lower limit              */
        int     zLoLimit,               /* z-array lower limit              */
        int     xHiLimit,               /* x-array upper limit              */
        int     yHiLimit,               /* y-array upper limit              */
        int     zHiLimit,               /* z-array upper limit              */
        double  ******uvel_FC,          /* u-face-centered velocity         (IN/OUT)*/
                                        /* uvel_FC(x,y,z,face)                      */
        double  ******vvel_FC,          /*  v-face-centered velocity        (IN/OUT)*/
                                        /* vvel_FC(x,y,z, face)                     */
        double  ******wvel_FC,          /* w face-centered velocity         (IN/OUT)*/
                                        /* wvel_FC(x,y,z,face)                      */
 
        int     ***BC_types,            /* defines which boundary conditions(INPUT) */
                                        /* have been set on each wall       */
        int     nMaterials )

 {
    int     i,j,k,f,m,                  /* indices                           */
            wall,       var,
            xLo,        xHi, 
            yLo,        yHi, 
            zLo,        zHi,
            wallLo,     wallHi,
            faceLo,     faceHi,
            should_I_leave;
/*__________________________________
* double check inputs
*___________________________________*/
    assert ( xLoLimit >= 0 && xHiLimit <= X_MAX_LIM);
    assert ( yLoLimit >= 0 && yHiLimit <= Y_MAX_LIM);
    assert ( zLoLimit >= 0 && zHiLimit <= Z_MAX_LIM);
/*__________________________________
*   Determine the looping indices
*   for multidimensional problems
*___________________________________*/
#if (N_DIMENSIONS == 1) 
        wallLo = LEFT;  wallHi = RIGHT;
        faceLo = LEFT;  faceHi = RIGHT;
#endif
#if (N_DIMENSIONS == 2) 
        wallLo = TOP;   wallHi = LEFT;
        faceLo = TOP;   faceHi = LEFT;
#endif
#if (N_DIMENSIONS == 3) 
        wallLo = TOP;   wallHi = BACK;
        faceLo = TOP;   faceHi = BACK;
#endif
/*__________________________________
*   Test to see if you should be in this function
*___________________________________*/
    should_I_leave = YES;
    for(m = 1; m <= nMaterials; m++)
    {    
        for( wall = wallLo; wall <= wallHi; wall ++)
        {
            for (var = VEL_BC; var <= PRESS_BC; var ++)
            {
                if(BC_types[wall][var][m] == WALL) should_I_leave = NO;
            }
        }
    }
    if (should_I_leave == YES) return;
/*______________________________________________________________________
*   Now loop over all of the walls and set the appropriate BC
*   You need to find what the appropriate looping limits are for each 
*   side of the computational domain
*_______________________________________________________________________*/
            
    for( wall = wallLo; wall <= wallHi; wall ++)
    {

       if( BC_types[wall][VEL_BC][m] == WALL)
       {          
            /*_________________________________
            *   Define the corner of each wall
            *   BE CAREFULL HERE NOT TO OVERLAP!
            *   NEED TO ADD 3D.  Note this is different
            *   from the upper and lower limits 
            *   calculated in "find_loop_index_limits_at_domain_edges("
            *___________________________________*/
           if( wall == LEFT )
           {
                f   = LEFT;
                xLo = xLoLimit;                xHi = xLoLimit;
                yLo = yLoLimit;                yHi = yHiLimit; 
                zLo = zLoLimit;                zHi = zHiLimit;
            }
           if( wall == RIGHT ) 
           {    
                f   = RIGHT;
                xLo = xHiLimit;                xHi = xHiLimit;
                yLo = yLoLimit;                yHi = yHiLimit; 
                zLo = zLoLimit;                zHi = zHiLimit;
            }
           if( wall == TOP ) 
           {
                f   = TOP;
                xLo = xLoLimit;                xHi = xHiLimit;
                yLo = yHiLimit;                yHi = yHiLimit; 
                zLo = zLoLimit;                zHi = zHiLimit;
            }
           if( wall == BOTTOM ) 
           {
                f   = BOTTOM;
                xLo = xLoLimit;                xHi = xHiLimit;
                yLo = yLoLimit;                yHi = yLoLimit; 
                zLo = zLoLimit;                zHi = zHiLimit;
           }
           /*__________________________________
           *   For each variable set the boundary condition
           *   if appropriate
           *___________________________________*/
           for(m = 1; m <= nMaterials; m++)
           {
              for ( k = zLo; k <= zHi; k++)
              {
                  for ( j = yLo; j <= yHi; j++)
                  {
                      for ( i = xLo; i <= xHi; i++)
                      {
                           /*__________________________________
                           *   RIGHT WALL
                           *___________________________________*/
                           if (f == RIGHT)
                           { 
                               *uvel_FC[xHi][j][k][f][m]       = 0.0;
                               *vvel_FC[xHi+1][j][k][TOP][m]   = -*vvel_FC[xHi][j][k][TOP][m];
                               *vvel_FC[xHi+1][j][k][BOTTOM][m]= -*vvel_FC[xHi][j][k][BOTTOM][m];
                           }
                           /*__________________________________
                           *   LEFT WALL
                           *___________________________________*/
                           if (f == LEFT)
                           { 
                               *uvel_FC[xLo][j][k][f][m]       = 0.0;
                               *vvel_FC[xLo-1][j][k][TOP][m]   = -*vvel_FC[xLo][j][k][TOP][m];
                               *vvel_FC[xLo-1][j][k][BOTTOM][m]= -*vvel_FC[xLo][j][k][BOTTOM][m];
                           }
                           /*__________________________________
                           *   BOTTOM WALL
                           *___________________________________*/
                            if (f == BOTTOM)
                           { 
                               *vvel_FC[i][yLo][k][f][m]       = 0.0;
                               *uvel_FC[i][yLo-1][k][LEFT][m]  = -*uvel_FC[i][yLo][k][LEFT][m];
                               *uvel_FC[i][yLo-1][k][RIGHT][m] = -*uvel_FC[i][yLo][k][RIGHT][m];
                           }
                           /*__________________________________
                           *   TOP WALL
                           *___________________________________*/
                            if (f == TOP)
                           { 
                               *vvel_FC[i][yHi][k][f][m]       = 0.0;
                               *uvel_FC[i][yHi+1][k][LEFT][m]  = -*uvel_FC[i][yHi][k][LEFT][m];
                               *uvel_FC[i][yHi+1][k][RIGHT][m] = -*uvel_FC[i][yHi][k][RIGHT][m];

                               /*__________________________________
                               *   HARDWIRED FOR DRIVEN CAVITY PROBLEM
                               * where utop = 0.5
                               *___________________________________*/
                              /*  *uvel_FC[i][yHi+1][k][RIGHT][m]= 1.0 - *uvel_FC[i][yHi][k][RIGHT][m];
                               *uvel_FC[i][yHi+1][k][LEFT][m]= 1.0 - *uvel_FC[i][yHi][k][LEFT][m];  */
                           }


                      }
                  }
              }
           }
       }
    }


/*______________________________________________________________________
*   DEBUGGING INFORMATION
*_______________________________________________________________________*/ 
#if switchDebug_set_wall_BC_FC
    fprintf(stderr,"****************************************************************************\n");
    fprintf(stderr,"                        SET_WALL_BC_FC\n");
    fprintf(stderr,"****************************************************************************\n");         
         
    printData_6d(       GC_LO(xLoLimit),     GC_LO(yLoLimit),       GC_LO(zLoLimit),
                        GC_HI(xHiLimit),     (yHiLimit),       (zHiLimit),
                        RIGHT,              LEFT,
                        m,                  m,
                       "set_Wall_BC_FC",     
                       "Uvel_FC with ghost cells",           uvel_FC,        0);

    printData_6d(       GC_LO(xLoLimit),     GC_LO(yLoLimit),       GC_LO(zLoLimit),
                        GC_HI(xHiLimit),     (yHiLimit),            (zHiLimit),
                        TOP,                BOTTOM,
                        m,                  m,
                       "set_Wall_BC_FC",     
                       "vvel_FC with ghost cells",           vvel_FC,        0);

                           
    fprintf(stderr,"****************************************************************************\n");         
    
    fprintf(stderr,"press return to continue\n");
    getchar();            
#endif
/*__________________________________
*   Quite fullwarn remarks is a way that
*   is compiler independent
*___________________________________*/
    faceHi = faceHi;                                faceLo = faceLo;
    QUITE_FULLWARN(*wvel_FC[0][0][0][1][1]);
 }
#endif
