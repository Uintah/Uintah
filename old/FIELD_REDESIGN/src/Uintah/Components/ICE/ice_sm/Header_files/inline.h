#ifndef __INLINE_H
#define __INLINE_H
#ifdef __sgi
    #define inline __inline
#endif
/* ---------------------------------------------------------------------
 Function:  find_edge_vel__edge_is_parallel_w_X_axis--SOURCE: Step 4,compute edge velocities
 Filename:  inline.h
 Purpose:   Compute the velocity at the edge of a cell.  The edge
            must be parallel with the x-axis.  The edge velocity is simply
            the average of the velocities in the four surrounding cell that share
            the edge.
            
 Version       Programmer         Date       Description                      
     -------   ----------         ----       -----------                 
        1.0     Todd Harman       1/19/00  
             
 Implementation Note:
            This function has been inlined to speed it up since it is called 
            for every cell.                            
 ---------------------------------------------------------------------  */        
inline double find_edge_vel__edge_is_parallel_w_X_axis(
        double ****data_CC,
        int     i,
        int     j,
        int     k,
        int     m )             
{ 
    double temp;         
      temp =    0.25 * (data_CC[m][i][j][k]     + data_CC[m][i][j][k+1]  
                     +  data_CC[m][i][j+1][k]   + data_CC[i][j+1][k+1][m]);
    return temp;
}
/*STOP_DOC*/
/* ---------------------------------------------------------------------
 Function:  find_edge_vel__edge_is_parallel_w_Y_axis--SOURCE: Step 4,compute edge velocities
 Filename:  inline.h
 Purpose:   Compute the velocity at the edge of a cell.  The edge
            must be parallel with the Y-axis.  The edge velocity is simply
            the average of the velocities in the four surrounding cell that share
            the edge.
            
 Version       Programmer         Date       Description                      
     -------   ----------         ----       -----------                 
        1.0     Todd Harman       1/19/00  
             
 Implementation Note:
            This function has been inlined to speed it up since it is called 
            for every cell.                            
 ---------------------------------------------------------------------  */        
inline double find_edge_vel__edge_is_parallel_w_Y_axis(
        double ****data_CC,
        int     i,
        int     j,
        int     k,
        int     m )
{     
    double temp;
         
      temp =    0.25 * (data_CC[m][i][j][k]     + data_CC[m][i+1][j][k]
                     +  data_CC[m][i][j][k+1]   + data_CC[i+1][j][k+1][m]);
    return temp;
}
/*STOP_DOC*/

/* ---------------------------------------------------------------------
 Function:  find_edge_vel__edge_is_parallel_w_Z_axis--SOURCE: Step 4,compute edge velocities
 Filename:  inline.h
 Purpose:   compute the velocity at the edge of the cell.  The edge
            must be parallel with the z-axis.  The edge velocity is simply
            the average of the velocities in the four surrounding cells that share
            the edge.

 Version       Programmer         Date       Description                      
     -------   ----------         ----       -----------                 
        1.0     Todd Harman       1/19/00  
             
 Implementation Note:
            This function has been inlined to speed it up since it is called 
            for every cell.                            
 ---------------------------------------------------------------------  */        
inline double find_edge_vel__edge_is_parallel_w_Z_axis(
        double ****data_CC,
        int     i,
        int     j,
        int     k,
        int     m )
{
    double temp;          
      temp =  0.25 * (data_CC[m][i][j][k]     + data_CC[m][i+1][j][k]        
                   +  data_CC[m][i][j+1][k]   + data_CC[m][i+1][j+1][k]);
    return temp;
}
/*STOP_DOC*/

#include <math.h>
#include <assert.h>
#include "functionDeclare.h"
#include "parameters.h"
#include "switches.h"
#include "macros.h"
/* ---------------------------------------------------------------------
 Function:  find_transport_property_FC--SOURCE: Step 4,used to compute the source terms
 Filename:  inline.h
 Purpose:   compute the face-centered transport properties 

 Version       Programmer         Date       Description                      
     -------   ----------         ----       -----------                 
        1.0     Todd Harman       1/19/00       
 Implementation Note:
 This function has been inlined to speed it up since it is called 
 for every cell.                            
 ---------------------------------------------------------------------  */        
inline void find_transport_property_FC(
        int     i,
        int     j,
        int     k,
        int     m,
        double ****data_CC,             /* cell-centered data               */
        double *data_FC)                /* face-centered results            */
{        
    /*__________________________________
    *   Right Face
    *___________________________________*/   
    assert( data_CC[m][i][j][k] + data_CC[m][i+1][j][k] > SMALL_NUM);           
    data_FC[RIGHT]  = (2.0 * data_CC[m][i][j][k] * data_CC[m][i+1][j][k])/
                            (data_CC[m][i][j][k] + data_CC[m][i+1][j][k]);
     /*__________________________________
     *  Left Face
     *___________________________________*/     
    assert( data_CC[m][i][j][k] + data_CC[m][i-1][j][k] > SMALL_NUM);          
    data_FC[LEFT]   = (2.0 * data_CC[m][i][j][k] * data_CC[m][i-1][j][k])/
                            (data_CC[m][i][j][k] + data_CC[m][i-1][j][k]);
                    
     /*__________________________________
     *  Top Face
     *___________________________________*/     
    assert( data_CC[m][i][j][k] + data_CC[m][i][j+1][k] > SMALL_NUM);          
    data_FC[TOP]    = (2.0 * data_CC[m][i][j][k] * data_CC[m][i][j+1][k])/
                            (data_CC[m][i][j][k] + data_CC[m][i][j+1][k]);
                    
     /*__________________________________
     *  Bottom Face
     *___________________________________*/     
    assert( data_CC[m][i][j][k] + data_CC[m][i][j-1][k] > SMALL_NUM);          
    data_FC[BOTTOM] = (2.0 * data_CC[m][i][j][k] * data_CC[m][i][j-1][k])/
                            (data_CC[m][i][j][k] + data_CC[m][i][j-1][k]);
#if(N_DIMENSIONS == 3)
     /*__________________________________
     *  Front Face
     *___________________________________*/     
    assert( data_CC[m][i][j][k] + data_CC[m][i][j][k+1] > SMALL_NUM);          
    data_FC[FRONT]  = (2.0 * data_CC[m][i][j][k] * data_CC[m][i][j][k+1])/
                            (data_CC[m][i][j][k] + data_CC[m][i][j][k+1]);
                    
     /*__________________________________
     *  Back Face
     *___________________________________*/     
    assert( data_CC[m][i][j][k] + data_CC[m][i][j][k-1] > SMALL_NUM);          
    data_FC[BACK]   = (2.0 * data_CC[m][i][j][k] * data_CC[m][i][j][k-1])/
                            (data_CC[m][i][j][k] + data_CC[m][i][j][k-1]);
#endif
}
/*STOP_DOC*/


#include <math.h>
#include <assert.h>
#include "functionDeclare.h"
#include "parameters.h"
#include "switches.h"
#include "macros.h"
/* ---------------------------------------------------------------------
 Function:  divergence_of_velocity_for_tau_terms_FC--SOURCE: Step 4,used to compute the viscous source terms
 Filename:  inline.h
 Purpose:   compute the face-centered transport properties 

 Version       Programmer         Date       Description                      
     -------   ----------         ----       -----------                 
        1.0     Todd Harman       1/19/00       
 Implementation Note:
 This function has been inlined to speed it up since it is called 
 for every cell.                            
 ---------------------------------------------------------------------  */        
inline void divergence_of_velocity_for_tau_terms_FC(
        int     i,
        int     j,
        int     k,
        double delX,                        /* Cell width                       */
        double delY,                        /* Cell Width in the y dir          */
        double delZ,                        /* Cell width in the z dir          */
        double ****uvel_CC,                 /* cell-centered velocity in the    */
        double ****vvel_CC,                 /* x, y and z directions            */
        double ****wvel_CC,
        double *div_vel_FC,                 /* face-centered results            */
        int     m)
{
   double   term1, term2, term3,
            uvel_ED_top_right_z     = 0.0,         /* edge velocities                  */
            uvel_ED_top_left_z      = 0.0,
            uvel_ED_bottom_right_z  = 0.0,
            uvel_ED_bottom_left_z   = 0.0,
 
            uvel_ED_front_right_y   = 0.0,
            uvel_ED_front_left_y    = 0.0,
            uvel_ED_back_right_y    = 0.0,
            uvel_ED_back_left_y     = 0.0,
 
            vvel_ED_right_top_z     = 0.0,
            vvel_ED_left_top_z      = 0.0,
            vvel_ED_right_bottom_z  = 0.0,
            vvel_ED_left_bottom_z   = 0.0,

            vvel_ED_back_top_x      = 0.0,
            vvel_ED_back_bottom_x   = 0.0,
            vvel_ED_front_top_x     = 0.0,
            vvel_ED_front_bottom_x  = 0.0,

            wvel_ED_right_front_y   = 0.0,
            wvel_ED_left_front_y    = 0.0,
            wvel_ED_right_back_y    = 0.0,
            wvel_ED_left_back_y     = 0.0,

            wvel_ED_top_front_x     = 0.0,
            wvel_ED_top_back_x      = 0.0,
            wvel_ED_bottom_front_x  = 0.0,
            wvel_ED_bottom_back_x   = 0.0;
/*__________________________________
*   bullet proofing
*___________________________________*/
#if (N_DIMENSIONS == 2 )
    assert( delX > SMALL_NUM && delY > SMALL_NUM);
    delZ = 1.0;
#endif    
#if N_DIMENSIONS == 3
    assert( delZ > SMALL_NUM )
#endif
/*______________________________________________________________________
*   First compute all of the edge velocities.  The edge velocity is simply
*   the average of the velocities in the four surrounding cell that share
*   the edge. 
*_______________________________________________________________________*/ 

    uvel_ED_top_right_z     = find_edge_vel__edge_is_parallel_w_X_axis( uvel_CC, i,  j,  k,  m);
    uvel_ED_top_left_z      = find_edge_vel__edge_is_parallel_w_X_axis( uvel_CC, i-1,j,  k,  m);
    uvel_ED_bottom_right_z  = find_edge_vel__edge_is_parallel_w_X_axis( uvel_CC, i,  j-1,k,  m);
    uvel_ED_bottom_left_z   = find_edge_vel__edge_is_parallel_w_X_axis( uvel_CC, i-1,j-1,k,  m);
    
    uvel_ED_front_right_y   = find_edge_vel__edge_is_parallel_w_Y_axis( uvel_CC, i,  j,  k,  m);
    uvel_ED_front_left_y    = find_edge_vel__edge_is_parallel_w_Y_axis( uvel_CC, i-1,j,  k,  m);
    uvel_ED_back_right_y    = find_edge_vel__edge_is_parallel_w_Y_axis( uvel_CC, i,  j,  k-1,m);
    uvel_ED_back_left_y     = find_edge_vel__edge_is_parallel_w_Y_axis( uvel_CC, i-1,j,  k-1,m);
       
    vvel_ED_right_top_z     = find_edge_vel__edge_is_parallel_w_Z_axis( vvel_CC, i,  j,  k,  m);
    vvel_ED_left_top_z      = find_edge_vel__edge_is_parallel_w_Z_axis( vvel_CC, i-1,j,  k,  m);
    vvel_ED_right_bottom_z  = find_edge_vel__edge_is_parallel_w_Z_axis( vvel_CC, i,  j-1,k,  m);
    vvel_ED_left_bottom_z   = find_edge_vel__edge_is_parallel_w_Z_axis( vvel_CC, i-1,j-1,k,  m);

    vvel_ED_back_top_x      = find_edge_vel__edge_is_parallel_w_X_axis( vvel_CC, i,  j,  k-1,m);
    vvel_ED_back_bottom_x   = find_edge_vel__edge_is_parallel_w_X_axis( vvel_CC, i,  j-1,k-1,m);
    vvel_ED_front_top_x     = find_edge_vel__edge_is_parallel_w_X_axis( vvel_CC, i,  j,  k,  m);
    vvel_ED_front_bottom_x  = find_edge_vel__edge_is_parallel_w_X_axis( vvel_CC, i,  j-1,k,  m);
    
    wvel_ED_right_front_y   = find_edge_vel__edge_is_parallel_w_Y_axis( wvel_CC, i,  j,  k,  m);
    wvel_ED_left_front_y    = find_edge_vel__edge_is_parallel_w_Y_axis( wvel_CC, i-1,j,  k,  m);
    wvel_ED_right_back_y    = find_edge_vel__edge_is_parallel_w_Y_axis( wvel_CC, i,  j,  k-1,m);
    wvel_ED_left_back_y     = find_edge_vel__edge_is_parallel_w_Y_axis( wvel_CC, i-1,j,  k-1,m); 
    
    wvel_ED_top_front_x     = find_edge_vel__edge_is_parallel_w_X_axis( wvel_CC, i,  j,  k,  m);
    wvel_ED_top_back_x      = find_edge_vel__edge_is_parallel_w_X_axis( wvel_CC, i,  j,  k-1,m);
    wvel_ED_bottom_front_x  = find_edge_vel__edge_is_parallel_w_X_axis( wvel_CC, i,  j-1,k,  m);
    wvel_ED_bottom_back_x   = find_edge_vel__edge_is_parallel_w_X_axis( wvel_CC, i  ,j-1,k-1,m);


    /*__________________________________
    *   Right Face
    *___________________________________*/   
    term1 = (uvel_CC[m][i+1][j][k]  - uvel_CC[m][i][j][k] )     /delX; 
    term2 = (vvel_ED_right_top_z    - vvel_ED_right_bottom_z )  /delY;         
    term3 = (wvel_ED_right_front_y  - wvel_ED_right_back_y )    /delZ; 
    div_vel_FC[RIGHT]  =   term1 + term2 + term3;
    
    /*__________________________________
    *  Left Face
    *___________________________________*/
    term1 = (uvel_CC[m][i][j][k]    - uvel_CC[m][i-1][j][k] )   /delX; 
    term2 = (vvel_ED_left_top_z     - vvel_ED_left_bottom_z )   /delY;         
    term3 = (wvel_ED_left_front_y   - wvel_ED_left_back_y )     /delZ; 
    div_vel_FC[LEFT]  =   term1 + term2 + term3;
                    
     /*__________________________________
     *  Top Face
     *___________________________________*/     
    term1 = (uvel_ED_top_right_z    - uvel_ED_top_left_z )      /delX; 
    term2 = (vvel_CC[m][i][j+1][k]  - vvel_CC[m][i][j][k] )     /delY;         
    term3 = (wvel_ED_top_front_x    - wvel_ED_top_back_x )      /delZ; 
    div_vel_FC[TOP]  =   term1 + term2 + term3;
                
     /*__________________________________
     *  Bottom Face
     *___________________________________*/ 
    term1 = (uvel_ED_bottom_right_z - uvel_ED_bottom_left_z )   /delX; 
    term2 = (vvel_CC[m][i][j][k]    - vvel_CC[m][i][j-1][k] )   /delY;         
    term3 = (wvel_ED_bottom_front_x - wvel_ED_bottom_back_x )   /delZ; 
    div_vel_FC[BOTTOM]  =   term1 + term2 + term3;
    
     /*__________________________________
     *  Front Face
     *___________________________________*/   
#if(N_DIMENSIONS == 3)  
    term1 = (uvel_ED_front_right_y  - uvel_ED_front_left_y  )    /delX; 
    term2 = (vvel_ED_front_top_x    - vvel_ED_front_bottom_x )  /delY;         
    term3 = (wvel_CC[i][j][k+1]     - wvel_CC[i][j][k])         /delZ; 
    div_vel_FC[FRONT]  =   term1 + term2 + term3;                    

     /*__________________________________
     *  Back Face
     *___________________________________*/     
    term1 = (uvel_ED_back_right_y  - uvel_ED_back_left_y )      /delX; 
    term2 = (vvel_ED_back_top_x    - vvel_ED_back_bottom_x )    /delY;         
    term3 = (wvel_CC[i][j][k]      - wvel_CC[i][j][k-1])        /delZ; 
    div_vel_FC[BACK]  =   term1 + term2 + term3;                    

#endif
/*__________________________________
*   Quite fullwarn compiler remarks
*___________________________________*/
    uvel_ED_front_right_y   = uvel_ED_front_right_y;    uvel_ED_front_left_y  = uvel_ED_front_left_y;
    uvel_ED_back_right_y    = uvel_ED_back_right_y;     uvel_ED_back_left_y   = uvel_ED_back_left_y;
    vvel_ED_back_top_x      = vvel_ED_back_top_x;       vvel_ED_back_bottom_x = vvel_ED_back_bottom_x;
    vvel_ED_front_top_x     = vvel_ED_front_top_x;      vvel_ED_front_bottom_x=vvel_ED_front_bottom_x;
}
/*STOP_DOC*/
#endif      /*__INLINE_H*/

