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
    double  delX,                       /* cell size in x direction         (INPUT) */
    double  delY,                       /* cell size in y direction         (INPUT) */
    double  delZ,                       /* cell size in z direction         (INPUT) */
    double  ****data_CC,                /* cell-centered data               (INPUT) */    
    double  ******data_FC,              /* u-face-centered velocity         (IN/OUT)*/
    int     var,                        /* TEMP, PRESS, UVEL                (INPUT) */                    

    int     ***BC_types,                /* defines which boundary conditions(INPUT) */
                                        /* have been set on each wall               */
    double  ***BC_Values,               /* BC values BC_values[wall][var][m](INPUT)*/
                                        
    int     ***BC_float_or_FLOAT,       /* BC_float_or_FLOAT[wall][var][m]  (INPUT)*/
                                        /* Variable on boundary is either   */
                                        /* FLOAT or it floats during the    */
                                        /* computation                      */
    int     nMaterials        )

 {
    int     i,j,k,f,m,                  /* indices                           */
            wall,       cell,    
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
                        
                        
#if 0                        
/*`==========TESTING==========*/ 
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
 /*==========TESTING==========`*/    
 #endif                    

            /*__________________________________
            *  TOP of domain
            *___________________________________*/                        
            if (wall == TOP) 
            {                   
                for ( k = zLo; k <= zHi; k++)
                {
                    for ( j = yLo; j <= yHi; j++)
                    { 
                        cell = j-1; 
                        for ( i = xLo; i <= xHi; i++)
                        {
                            *data_FC[i][j][k][TOP][m] = *data_FC[i][cell][k][TOP][m]
                            + BC_Values[wall][var][m] * delY;
                            
                            /* *data_FC[i][j][k][TOP][m] = data_CC[m][i][j][k]; */
                            /* *data_FC[i][j][k][BOTTOM][m] = data_CC[m][i][j][k]; */
                        }
                    }
                }
            }
            
            /*__________________________________
            *  BOTTOM of domain
            *___________________________________*/                        
            if (wall == BOTTOM) 
            {                   
                    
                for ( k = zLo; k <= zHi; k++)
                {
                    for ( j = yLo; j <= yHi; j++)
                    { 
                        cell = j+1;
                        for ( i = xLo; i <= xHi; i++)
                        {
                            *data_FC[i][j][k][BOTTOM][m]    = *data_FC[i][cell][k][BOTTOM][m]
                            - BC_Values[wall][var][m] * delY;
                            
#if 0                            
/*`==========TESTING==========*/ 
                        for(f = faceLo ; f <= faceHi; f ++)
                        {            
                            *data_FC[i][j][k][f][m] = data_CC[m][i][j][k];
                        }
 /*==========TESTING==========`*/
#endif
                        }
                    }
                }
            }
            
            /*__________________________________
            *  RIGHT of domain
            *___________________________________*/                        
            if (wall == RIGHT) 
            {                     
                for ( k = zLo; k <= zHi; k++)
                {
                    for ( j = yLo; j <= yHi; j++)
                    { 
                        for ( i = xLo; i <= xHi; i++)
                        {
                            cell = i-1;
                            *data_FC[i][j][k][RIGHT][m] = *data_FC[cell][j][k][RIGHT][m]
                            + BC_Values[wall][var][m] * delX;
                            /* *data_FC[i][j][k][TOP][m]   = *data_FC[cell][j][k][TOP][m]; */
#if 0
/*`==========TESTING==========*/ 
                        for(f = faceLo ; f <= faceHi; f ++)
                        {            
                            *data_FC[i][j][k][f][m] = data_CC[m][i][j][k];
                        }
 /*==========TESTING==========`*/
#endif
                        }
                    }
                }
            }
            
            /*__________________________________
            *  LEFT of domain
            *___________________________________*/                        
            if (wall == LEFT) 
            {                      
                for ( k = zLo; k <= zHi; k++)
                {
                    for ( j = yLo; j <= yHi; j++)
                    { 
                        for ( i = xLo; i <= xHi; i++)
                        {
                            cell = i+1;
                            *data_FC[i][j][k][LEFT][m]  = *data_FC[cell][j][k][LEFT][m]
                            - BC_Values[wall][var][m] * delX;
                            /* *data_FC[i][j][k][TOP][m]   = *data_FC[cell][j][k][TOP][m]; */
#if 0
/*`==========TESTING==========*/ 
                        for(f = faceLo ; f <= faceHi; f ++)
                        {            
                            *data_FC[i][j][k][f][m] = data_CC[m][i][j][k];
                        }
 /*==========TESTING==========`*/
#endif 
                        }
                    }
                }
            }
            /*__________________________________
            *  FR0NT of domain
            *___________________________________*/                        
            if (wall == FRONT) 
            {       
                Message(1,"File: boundary_cond_FC","Function: set_Neumann_BC_FC",
                "Error: Need to test to make sure this is right");
                         
                for ( k = zLo; k <= zHi; k++)
                {
                    cell = k-1;
                    for ( j = yLo; j <= yHi; j++)
                    { 
                        for ( i = xLo; i <= xHi; i++)
                        {
                            *data_FC[i][j][k][FRONT][m] = *data_FC[i][j][cell][FRONT][m]
                            + BC_Values[wall][var][m] * delZ;
#if 0
/*`==========TESTING==========*/ 
                        for(f = faceLo ; f <= faceHi; f ++)
                        {            
                            *data_FC[i][j][k][f][m] = data_CC[m][i][j][k];
                        }
 /*==========TESTING==========`*/
#endif 
                        }
                    }
                }
            }
            /*__________________________________
            *  FR0NT of domain
            *___________________________________*/                        
            if (wall == BACK) 
            {          
                Message(1,"File: boundary_cond_FC","Function: set_Neumann_BC_FC",
                "Error: Need to test to make sure this is right");           
                for ( k = zLo; k <= zHi; k++)
                {
                    cell = k+1;
                    for ( j = yLo; j <= yHi; j++)
                    { 
                        for ( i = xLo; i <= xHi; i++)
                        {
                            *data_FC[i][j][k][BACK][m] = *data_FC[i][j][cell][BACK][m]
                            - BC_Values[wall][var][m] * delZ;
/*`==========TESTING==========*/ 
                        for(f = faceLo ; f <= faceHi; f ++)
                        {            
                            *data_FC[i][j][k][f][m] = data_CC[m][i][j][k];
                        }
 /*==========TESTING==========`*/
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
/*`==========TESTING==========*/ 
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
 /*==========TESTING==========`*/
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
