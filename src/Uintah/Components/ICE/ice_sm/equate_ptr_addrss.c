/* 
======================================================================*/
#include "functionDeclare.h"
#include "parameters.h"
#include "switches.h"
#include "macros.h"
/* ---------------------------------------------------------------------

 Function:  equate_ptr_addresses_adjacent_cell_faces--MISC: equates the pointer address of face-centered variables.
 Filename:  equate_ptr_addrss.c
 Purpose:   Set the pointer address of adjacent cell faces equal to 
            each other.
            For example set the address of 
                 [i][j][k][RIGHT][m] = [i-1][j][k][LEFT][m] 
            and similarly for the other faces.
            
 Steps
    1)      Equate the face address on all cells except for those 
            along the top and right layer of cells.
              
    2)      Equate the face addresses in the top and right layer of cells
     
 History: 
 Version       Programmer         Date       Description                      
     -------   ----------         ----       -----------                 
        1.0     Todd Harman       010/13/99                                
                                 ________ 
                                /  1    /|
                               /_______/ |
                              |       | ______(3)
                       (4)____| I,J,K |  |     
                              |       | /      
                              |_______|/
                                  |               (6) = back face
                                 (2)              (5) = front face  
--------------------------------------------------
 WARNING: (1/18/00)
 You MUST set the top and bottom faces like this,
    (*)_FC[i][j+1][k][BOTTOM] = (*)_FC[i][j][k][TOP]; 
 If you set them the other way,
    (*)_FC[i][j][k][TOP]      = (*)_FC[i][j+1][k][BOTTOM];  
  the code will core dump when you deallocate that array.
---------------------------------------------------------------------*/ 

void equate_ptr_addresses_adjacent_cell_faces(
        double  *****x_FC,              /*--------pointer-------------------*/
        double  *****y_FC,              /* (*)_FC(i,j,k,face)               */
        double  *****z_FC,
        double  ******uvel_FC,          /* (*)vel_FC(i,j,k,face,material)   (IN/OUT)*/
        double  ******vvel_FC,          /*                                  (IN/OUT)*/
        double  ******wvel_FC,          /*                                  (IN/OUT)*/
        double  ******press_FC,         /*                                  (IN/OUT)*/
        double  ******tau_x_FC,         /* *x-stress component at each face (IN/OUT)*/
        double  ******tau_y_FC,         /* *y-stress component at each face (IN/OUT)*/
        double  ******tau_z_FC,         /* *z-stress component at each face (IN/OUT)*/
        int     nMaterials)
{
    int i,j,k,m;

/*__________________________________
*   Step 1:
* Now make sure that the face centered
* values know about each other.
* for example 
* [i][j][k][RIGHT][m] = [i-1][j][k][LEFT][m]
*   This covers all of the cells except 
*   a single layer on the top and right
*   sides.
*___________________________________*/  
    for ( m = 1; m <= nMaterials; m++)
    {
        for ( k = 0; k <= Z_MAX_LIM-1; k++)
        {
            for ( j = 0; j <= Y_MAX_LIM-1; j++)
            {
                for ( i = 0; i <= X_MAX_LIM-1; i++)
                {
                    /*__________________________________
                    *   Equate left and right sides
                    *___________________________________*/
                    x_FC[i][j][k][RIGHT]            = x_FC[i+1][j][k][LEFT];
                    y_FC[i][j][k][RIGHT]            = y_FC[i+1][j][k][LEFT];
                    z_FC[i][j][k][RIGHT]            = z_FC[i+1][j][k][LEFT];
                    press_FC[i][j][k][RIGHT][m]     = press_FC[i+1][j][k][LEFT][m];
                    tau_x_FC[i][j][k][RIGHT][m]     = tau_x_FC[i+1][j][k][LEFT][m];
                    tau_y_FC[i][j][k][RIGHT][m]     = tau_y_FC[i+1][j][k][LEFT][m];
                    tau_z_FC[i][j][k][RIGHT][m]     = tau_z_FC[i+1][j][k][LEFT][m];
                    uvel_FC[i][j][k][RIGHT][m]      = uvel_FC[i+1][j][k][LEFT][m];
                    vvel_FC[i][j][k][RIGHT][m]      = vvel_FC[i+1][j][k][LEFT][m];
                    wvel_FC[i][j][k][RIGHT][m]      = wvel_FC[i+1][j][k][LEFT][m];
                    /*__________________________________
                    *   Equate top and bottom sides
                    *___________________________________*/
                    x_FC[i][j+1][k][BOTTOM]		= x_FC[i][j][k][TOP];	
                    y_FC[i][j+1][k][BOTTOM]		= y_FC[i][j][k][TOP];	
                    z_FC[i][j+1][k][BOTTOM]		= z_FC[i][j][k][TOP];	
                    press_FC[i][j+1][k][BOTTOM][m]	= press_FC[i][j][k][TOP][m];
                    tau_x_FC[i][j+1][k][BOTTOM][m]	= tau_x_FC[i][j][k][TOP][m];
                    tau_y_FC[i][j+1][k][BOTTOM][m]	= tau_y_FC[i][j][k][TOP][m];
                    tau_z_FC[i][j+1][k][BOTTOM][m] 	= tau_z_FC[i][j][k][TOP][m];
                    uvel_FC[i][j+1][k][BOTTOM][m]	= uvel_FC[i][j][k][TOP][m];
                    vvel_FC[i][j+1][k][BOTTOM][m]	= vvel_FC[i][j][k][TOP][m];
                    wvel_FC[i][j+1][k][BOTTOM][m]	= wvel_FC[i][j][k][TOP][m];
                    /*__________________________________
                    *   Equate  front and back sides
                    *___________________________________*/
                    x_FC[i][j][k][FRONT]             = x_FC[i][j][k+1][BACK];
                    y_FC[i][j][k][FRONT]             = y_FC[i][j][k+1][BACK];
                    z_FC[i][j][k][FRONT]             = z_FC[i][j][k+1][BACK];
                    press_FC[i][j][k][FRONT][m]      = press_FC[i][j][k+1][BACK][m];
                    tau_x_FC[i][j][k][FRONT][m]      = tau_x_FC[i][j][k+1][BACK][m];
                    tau_y_FC[i][j][k][FRONT][m]      = tau_y_FC[i][j][k+1][BACK][m];
                    tau_z_FC[i][j][k][FRONT][m]      = tau_z_FC[i][j][k+1][BACK][m];                
                    uvel_FC[i][j][k][FRONT][m]       = uvel_FC[i][j][k+1][BACK][m];
                    vvel_FC[i][j][k][FRONT][m]       = vvel_FC[i][j][k+1][BACK][m];
                    wvel_FC[i][j][k][FRONT][m]       = wvel_FC[i][j][k+1][BACK][m];


                }
            }
        }  
    }  
    
/*__________________________________
*   Step 2:
*   Now equate the addresses along
*   the right, top and front layer of cells
*___________________________________*/ 
    for ( m = 1; m <= nMaterials; m++)
    { 
        for ( k = 0; k <= Z_MAX_LIM-1; k++)
        {
            for ( j = Y_MAX_LIM; j <= Y_MAX_LIM; j++)
            {
                for ( i = 0; i <= X_MAX_LIM-1; i++)
                {
                    /*__________________________________
                    *   Equate left and right faces
                    *___________________________________*/
                    x_FC[i][j][k][RIGHT]            = x_FC[i+1][j][k][LEFT];
                    y_FC[i][j][k][RIGHT]            = y_FC[i+1][j][k][LEFT];
                    z_FC[i][j][k][RIGHT]            = z_FC[i+1][j][k][LEFT];
                    press_FC[i][j][k][RIGHT][m]     = press_FC[i+1][j][k][LEFT][m];
                    tau_x_FC[i][j][k][RIGHT][m]     = tau_x_FC[i+1][j][k][LEFT][m];
                    tau_y_FC[i][j][k][RIGHT][m]     = tau_y_FC[i+1][j][k][LEFT][m];
                    tau_z_FC[i][j][k][RIGHT][m]     = tau_z_FC[i+1][j][k][LEFT][m];
                    uvel_FC[i][j][k][RIGHT][m]      = uvel_FC[i+1][j][k][LEFT][m];
                    vvel_FC[i][j][k][RIGHT][m]      = vvel_FC[i+1][j][k][LEFT][m];
                    wvel_FC[i][j][k][RIGHT][m]      = wvel_FC[i+1][j][k][LEFT][m];
                 }
            }
        }
    }
    /*__________________________________
    *   RIGHT layer of cells
    *___________________________________*/
    for ( m = 1; m <= nMaterials; m++)
    {
        for ( k = 0; k <= Z_MAX_LIM-1; k++)
        {
            for ( j = 0; j <= Y_MAX_LIM-1; j++)
            {
                for ( i = X_MAX_LIM; i <= X_MAX_LIM; i++)
                {

                    /*__________________________________
                    *   Equate top and bottom faces
                    *___________________________________*/
                    x_FC[i][j+1][k][BOTTOM]              = x_FC[i][j][k][TOP];
                    y_FC[i][j+1][k][BOTTOM]              = y_FC[i][j][k][TOP];
                    z_FC[i][j+1][k][BOTTOM]              = z_FC[i][j][k][TOP];
                    press_FC[i][j+1][k][BOTTOM][m]       = press_FC[i][j][k][TOP][m];
                    tau_x_FC[i][j+1][k][BOTTOM][m]       = tau_x_FC[i][j][k][TOP][m];
                    tau_y_FC[i][j+1][k][BOTTOM][m]       = tau_y_FC[i][j][k][TOP][m];
                    tau_z_FC[i][j+1][k][BOTTOM][m]       = tau_z_FC[i][j][k][TOP][m];
                     uvel_FC[i][j+1][k][BOTTOM][m]       = uvel_FC[i][j][k][TOP][m]; 
                     vvel_FC[i][j+1][k][BOTTOM][m]       = vvel_FC[i][j][k][TOP][m];
                     wvel_FC[i][j+1][k][BOTTOM][m]       = wvel_FC[i][j][k][TOP][m];
                }
            }
        }
    }
    /*__________________________________
    *   BACK layer of cells
    *___________________________________*/
    for ( m = 1; m <= nMaterials; m++)
    {
        for ( k = Z_MAX_LIM; k <= Z_MAX_LIM; k++)
        {
            for ( j = 0; j <= Y_MAX_LIM-1; j++)
            {
                for ( i = 0; i <= X_MAX_LIM-1; i++)
                {
                    /*__________________________________
                    *   Equate left and right faces
                    *___________________________________*/
                    x_FC[i][j][k][RIGHT]            = x_FC[i+1][j][k][LEFT];
                    y_FC[i][j][k][RIGHT]            = y_FC[i+1][j][k][LEFT];
                    z_FC[i][j][k][RIGHT]            = z_FC[i+1][j][k][LEFT];
                    press_FC[i][j][k][RIGHT][m]     = press_FC[i+1][j][k][LEFT][m];
                    tau_x_FC[i][j][k][RIGHT][m]     = tau_x_FC[i+1][j][k][LEFT][m];
                    tau_y_FC[i][j][k][RIGHT][m]     = tau_y_FC[i+1][j][k][LEFT][m];
                    tau_z_FC[i][j][k][RIGHT][m]     = tau_z_FC[i+1][j][k][LEFT][m];
                    uvel_FC[i][j][k][RIGHT][m]      = uvel_FC[i+1][j][k][LEFT][m];
                    vvel_FC[i][j][k][RIGHT][m]      = vvel_FC[i+1][j][k][LEFT][m];
                    wvel_FC[i][j][k][RIGHT][m]      = wvel_FC[i+1][j][k][LEFT][m];
                    /*__________________________________
                    *   Equate top and bottom faces 
                    *___________________________________*/
                    x_FC[i][j+1][k][BOTTOM]		= x_FC[i][j][k][TOP];	
                    y_FC[i][j+1][k][BOTTOM]		= y_FC[i][j][k][TOP];	
                    z_FC[i][j+1][k][BOTTOM]		= z_FC[i][j][k][TOP];	
                    press_FC[i][j+1][k][BOTTOM][m]	= press_FC[i][j][k][TOP][m];
                    tau_x_FC[i][j+1][k][BOTTOM][m]	= tau_x_FC[i][j][k][TOP][m];
                    tau_y_FC[i][j+1][k][BOTTOM][m]	= tau_y_FC[i][j][k][TOP][m];
                    tau_z_FC[i][j+1][k][BOTTOM][m] 	= tau_z_FC[i][j][k][TOP][m];
                    uvel_FC[i][j+1][k][BOTTOM][m]	= uvel_FC[i][j][k][TOP][m];
                    vvel_FC[i][j+1][k][BOTTOM][m]	= vvel_FC[i][j][k][TOP][m];
                    wvel_FC[i][j+1][k][BOTTOM][m]	= wvel_FC[i][j][k][TOP][m];

                }
            }
        }
    }
    
              
}
/*STOP_DOC*/        
        
