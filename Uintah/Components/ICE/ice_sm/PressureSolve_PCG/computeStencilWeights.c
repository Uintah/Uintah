
/* 
 ======================================================================*/
#include <math.h> 
#include "pcgmg.h"
#include "parameters.h"
#include "functionDeclare.h"
#include "switches.h"
#include "macros.h"
/*---------------------------------------------------------------------  
 Function:  calc_delPress_Stencil_Weights_Neuman--PRESS: 
 Filename:  ComputeStencilWeights.c
 Purpose:   
            Compute the stencil weights 
    To be filled in
    
 Computational Domain:          Interior cells 
 Ghostcell data dependency:     

 References:
             
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       12/30/99
 ---------------------------------------------------------------------  */
void calc_delPress_Stencil_Weights_Neuman(
        int     xLoLimit,                   /* x-array lower limit              */
        int     yLoLimit,                   /* y-array lower limit              */
        int     zLoLimit,                   /* z-array lower limit              */
        int     xHiLimit,                   /* x-array upper limit              */
        int     yHiLimit,                   /* y-array upper limit              */
        int     zHiLimit,                   /* z-array upper limit              */
        double  delX,                       
        double  delY,                       
        double  delZ,                       
        double  delt,
        int     ***BC_types,                /* array containing the different   (INPUT) */
                                            /* types of boundary conditions     (INPUT) */
                                            /* BC_types[wall][variable]=type    (INPUT) */        
        double  ****rho_CC,                 /* Cell-centered density            (INPUT) */
        double  ****speedSound,             /* speed of sound (x,y,z, material) (INPUT) */
        int     nMaterials,                 /* number of materials              (INPUT) */
        stencilMatrix* stencil,             /* stencil                          (OUTPUT)*/
        Mat*    A)
{
    int         i,  j,  k,  indx,   mat;
    int         n,  m,  N,  ierr,   should_I_leave;
    int         wall,   wallLo,     wallHi;
    double      beta_n,                     /* temporary Variables              */
                beta_s, 
                beta_e, 
                beta_w, 
                beta_p;
                
    Scalar      an,                         /* stencil coefficients             */
                as, 
                ae, 
                aw, 
                ap; 
    void *ctx = (void*) stencil;
    Scalar  hx, hy;
    
/*__________________________________
*   Determine the looping indices
*   for multidimensional problems
*___________________________________*/
#if (N_DIMENSIONS == 1)  
        wallLo = LEFT;  wallHi = RIGHT;
#endif

#if (N_DIMENSIONS == 2) 
        wallLo = TOP;   wallHi = LEFT;
#endif
#if (N_DIMENSIONS == 3) 
        wallLo = TOP;   wallHi = BACK;
#endif    
/*__________________________________
*   Test to see if you should be in this function
*___________________________________*/
    should_I_leave = YES;
    for(m = 1; m <= nMaterials; m++)
    {
        for( wall = wallLo; wall <= wallHi; wall ++)
        {
            if(BC_types[wall][DELPRESS][m] == NEUMANN ) should_I_leave = NO;
        }
    }
    if (should_I_leave == YES) return;
/*__________________________________
*   initialize variables
*___________________________________*/
    m   = xHiLimit - xLoLimit + 1;
    n   = yHiLimit - yLoLimit + 1; 
    N   = m * n;   
    hx  = 1.0/m, 
    hy  = 1.0/n;
    
    hx  = delX;
    hy  = delY;    
/*__________________________________
*   Create the StencilMatrix Object
*___________________________________*/ 
  stencil   = (stencilMatrix*) ctx;
  ierr      = MatCreateShell( PETSC_COMM_SELF, N, N, N, N, ctx, A );                        CHKERRQ(ierr);
  ierr      = MatShellSetOperation( *A, MATOP_MULT, (void*) stencilMult );                  CHKERRQ(ierr);

/*__________________________________
*    Define stencil structure member data.
*___________________________________*/
  stencil->m = m;
  stencil->n = n;

  ierr = VecCreateSeq( PETSC_COMM_SELF, N, &(stencil->ap) );                                CHKERRQ(ierr);
  ierr = VecCreateSeq( PETSC_COMM_SELF, N, &(stencil->ae) );                                CHKERRQ(ierr);
  ierr = VecCreateSeq( PETSC_COMM_SELF, N, &(stencil->aw) );                                CHKERRQ(ierr);
  ierr = VecCreateSeq( PETSC_COMM_SELF, N, &(stencil->an) );                                CHKERRQ(ierr);
  ierr = VecCreateSeq( PETSC_COMM_SELF, N, &(stencil->as) );                                CHKERRQ(ierr);
  
/*______________________________________________________________________
*   Set the stencil values Neuman Boundary Conditions
*_______________________________________________________________________*/
    indx = 0;
    /*__________________________________
    *   HARDWIRE THE K AND M FOR NOW
    *___________________________________*/
    k   = 1;
    mat = nMaterials;
    
    for ( j = yLoLimit; j <= yHiLimit; j++) 
    {
        for ( i = xLoLimit; i <= xHiLimit; i++) 
        {                     
            beta_n = delt * 2.0/( rho_CC[i][j][k][mat] + rho_CC[i][j+1][k][mat] );
            beta_s = delt * 2.0/( rho_CC[i][j][k][mat] + rho_CC[i][j-1][k][mat] );
            beta_e = delt * 2.0/( rho_CC[i][j][k][mat] + rho_CC[i+1][j][k][mat] );
            beta_w = delt * 2.0/( rho_CC[i][j][k][mat] + rho_CC[i-1][j][k][mat] );
            beta_p = delX * delY/(rho_CC[i][j][k][mat] * delt * pow(speedSound[i][j][k][mat],2) );
            
/*`==========TESTING==========*/ 
#if switchDebug_pcgmg_test        
            beta_n = 1.0;
            beta_s = 1.0;
            beta_e = 1.0;
            beta_w = 1.0;
            beta_p = 0.0;
#endif
 /*==========TESTING==========`*/
    
            ae = -(hy/hx) * beta_e;
            aw = -(hy/hx) * beta_w;
            an = -(hx/hy) * beta_n;
            as = -(hx/hy) * beta_s;                    
            ap =   beta_p - (ae+aw+an+as);              /* center; conservative! */

            /*__________________________________
            * Neuman Boundary conditions
            *___________________________________*/
            if ( i == xLoLimit )                /* Left Side        */
            {
               aw = hy * beta_w;
               ap = beta_p - (ae + an + as);
            }
            else if ( i == xHiLimit )           /* Right Side        */
            {
               ae = hy * beta_e;
               ap = beta_p - (aw+an + as);
            }
            if ( j == yLoLimit )                /* Bottom           */
            {
               as = hx * beta_s;
               ap = beta_p -(ae + aw + an);
            }
            else if ( j == yHiLimit )           /* Top              */
            {
               an = hx * beta_n;
               ap = beta_p - (ae + aw + as);
            }
            /*__________________________________
            * Take care of the corner cells
            *___________________________________*/
            if ( i == xLoLimit && j == yLoLimit )       /* Lower Left Corner */
            {
               ap = beta_p - (ae + an);
               as = hx * beta_s;
               aw = hy * beta_w;
            }
            else if ( i == xLoLimit && j == yHiLimit )  /* Upper Left Corner */
            {
               ap = beta_p - (ae + as);
               an = hx * beta_n;
               aw = hy * beta_w;
            }
            else  if ( i == xHiLimit && j == yLoLimit )   /* Lower Right Corner */
            {
               ap = beta_p - (aw + an);
               as = hx * beta_s;
               ae = hy * beta_e;
            }
            else if ( i == xHiLimit && j == yHiLimit )  /* Upper Right Corner */
            {
               ap = beta_p - (aw + as);
               an = hx * beta_n;
               ae = hy * beta_e;
            }
            /*__________________________________
            * Finally set the stencil
            *___________________________________*/
            VecSetValue( stencil->ap, indx, ap, INSERT_VALUES );
            VecSetValue( stencil->ae, indx, ae, INSERT_VALUES );
            VecSetValue( stencil->aw, indx, aw, INSERT_VALUES );
            VecSetValue( stencil->an, indx, an, INSERT_VALUES );
            VecSetValue( stencil->as, indx, as, INSERT_VALUES );
            indx++;
        }
    }  

/*__________________________________
*   For testing and debugging
*___________________________________*/  
#if switchDebug_pcgmg_test
    #define switchInclude_stencil_test_code 1
    #include "testcode_PressureSolve.i"
    #undef switchInclude_stencil_test_code
#endif 
/*__________________________________
*   Assemble the matrix
*___________________________________*/
  ierr = MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY);                                          CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY);                                            CHKERRQ(ierr);

/*__________________________________
*   Quite fullwarn remarks in a way that
*   is compiler independent
*___________________________________*/  
    delZ = delZ;        zLoLimit = zLoLimit;        zHiLimit = zHiLimit;        ierr = ierr;
/*STOP_DOC*/
}



/* 
 ======================================================================*/
#include <math.h>
#include "pcgmg.h"
#include "parameters.h"
#include "functionDeclare.h"
#include "switches.h"
#include "macros.h"
/*---------------------------------------------------------------------  
 Function:  calc_delPress_Stencil_Weights_Dirichlet--PRESS: 
 Filename:  ComputeStencilWeights.c
 Purpose:   
            Compute the stencil weights 
    To be filled in
    
 Computational Domain:          Interior cells 
 Ghostcell data dependency:     

 References:
             
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       12/30/99
 ---------------------------------------------------------------------  */
void calc_delPress_Stencil_Weights_Dirichlet(
        int     xLoLimit,                   /* x-array lower limit              */
        int     yLoLimit,                   /* y-array lower limit              */
        int     zLoLimit,                   /* z-array lower limit              */
        int     xHiLimit,                   /* x-array upper limit              */
        int     yHiLimit,                   /* y-array upper limit              */
        int     zHiLimit,                   /* z-array upper limit              */
        double  delX,                       
        double  delY,                       
        double  delZ,                       
        double  delt,
        int     ***BC_types,                /* array containing the different   (INPUT) */
                                            /* types of boundary conditions     (INPUT) */
                                            /* BC_types[wall][variable]=type    (INPUT) */
        double  ****rho_CC,                 /* Cell-centered density            (INPUT) */
        double  ****speedSound,             /* speed of sound (x,y,z, material) (INPUT) */
        int     nMaterials,
        stencilMatrix* stencil,
        Mat*    A)
{
    int         i,  j,  k,  indx,   mat;
    int         n,  m,  N,  ierr,   should_I_leave;
    int         wall,   wallLo,     wallHi;
    double      beta_n, 
                beta_s, 
                beta_e, 
                beta_w, 
                beta_p;
    Scalar      an, 
                as, 
                ae, 
                aw, 
                ap; 
    void *ctx = (void*) stencil;
    Scalar  hx, hy;
/*__________________________________
*   Determine the looping indices
*   for multidimensional problems
*___________________________________*/
#if (N_DIMENSIONS == 1)  
        wallLo = LEFT;  wallHi = RIGHT;
#endif

#if (N_DIMENSIONS == 2) 
        wallLo = TOP;   wallHi = LEFT;
#endif
#if (N_DIMENSIONS == 3) 
        wallLo = TOP;   wallHi = BACK;
#endif      
/*__________________________________
*   Test to see if you should be in this function
*___________________________________*/
    should_I_leave = YES;
    for(m = 1; m <= nMaterials; m++)
    {
        for( wall = wallLo; wall <= wallHi; wall ++)
        {
            if(BC_types[wall][DELPRESS][m] == DIRICHLET ) should_I_leave = NO;
           
        }
    }
    if (should_I_leave == YES) return;
/*__________________________________
*   initialize variables
*___________________________________*/
    m   = xHiLimit - xLoLimit + 1;
    n   = yHiLimit - yLoLimit + 1; 
    N   = m * n;   
    hx  = 1.0/m, 
    hy  = 1.0/n;    
    
    hx  = delX;
    hy  = delY; 
/*__________________________________
*   Create the StencilMatrix Object
*___________________________________*/ 
  stencil   = (stencilMatrix*) ctx;
  ierr      = MatCreateShell( PETSC_COMM_SELF, N, N, N, N, ctx, A );                        CHKERRQ(ierr);
  ierr      = MatShellSetOperation( *A, MATOP_MULT, (void*) stencilMult );                  CHKERRQ(ierr);

/*__________________________________
*    Define stencil structure member data.
*___________________________________*/
  stencil->m = m;
  stencil->n = n;

  ierr = VecCreateSeq( PETSC_COMM_SELF, N, &(stencil->ap) );                                CHKERRQ(ierr);
  ierr = VecCreateSeq( PETSC_COMM_SELF, N, &(stencil->ae) );                                CHKERRQ(ierr);
  ierr = VecCreateSeq( PETSC_COMM_SELF, N, &(stencil->aw) );                                CHKERRQ(ierr);
  ierr = VecCreateSeq( PETSC_COMM_SELF, N, &(stencil->an) );                                CHKERRQ(ierr);
  ierr = VecCreateSeq( PETSC_COMM_SELF, N, &(stencil->as) );                                CHKERRQ(ierr);
  
/*______________________________________________________________________
*   Set the stencil values Dirichlet Boundary Conditions
*_______________________________________________________________________*/

    indx = 0;
    /*__________________________________
    *   HARDWIRE THE K AND M FOR NOW
    *___________________________________*/
    k   = 1;
    mat = nMaterials;
    for ( j = yLoLimit; j <= yHiLimit; j++) 
    {
        for ( i = xLoLimit; i <= xHiLimit; i++) 
        {
        
            beta_n = delt * 2.0/( rho_CC[i][j][k][mat] + rho_CC[i][j+1][k][mat] );
            beta_s = delt * 2.0/( rho_CC[i][j][k][mat] + rho_CC[i][j-1][k][mat] );
            beta_e = delt * 2.0/( rho_CC[i][j][k][mat] + rho_CC[i+1][j][k][mat] );
            beta_w = delt * 2.0/( rho_CC[i][j][k][mat] + rho_CC[i-1][j][k][mat] );
            beta_p = delX * delY/(rho_CC[i][j][k][mat] * delt * pow(speedSound[i][j][k][mat],2) );
            

/*`==========TESTING==========*/ 
#if switchDebug_pcgmg_test
            beta_n = 1.0;
            beta_s = 1.0;
            beta_e = 1.0;
            beta_w = 1.0;
            beta_p = 0.0;
#endif
 /*==========TESTING==========`*/
    
            ae =  -(hy/hx) * beta_e;
            aw =  -(hy/hx) * beta_w;
            an =  -(hx/hy) * beta_n;
            as =  -(hx/hy) * beta_s;
            ap =   beta_p - (ae+aw+an+as);              /* center; conservative! */
      
            /*__________________________________
            * Dirichlet Boundary conditions
            *___________________________________*/
            if ( i == xLoLimit)                     /* Left Side        */
            {
                aw  = -(8.0/3.0) * (hy/hx) * beta_w;
                ae  = -(hy/hx)   * ( (1.0/3.0)*beta_w + beta_e  );
            } 
            else if ( i == xHiLimit )              /* right Side        */ 
            {
                ae  = -(8.0/3.0) * (hy/hx) * beta_e;
                aw  = -(hy/hx)   * ( (1.0/3.0)*beta_e + beta_w);
            }  
            if ( j == yLoLimit )                     /* Bottom           */
            { 
                as  = -(8.0/3.0) * (hy/hx) * beta_s;
                an  = -(hy/hx)   * ( (1.0/3.0)*beta_s + beta_n);

            } 
            else if ( j == yHiLimit )              /* Top              */ 
            {
                an  = -(8.0/3.0) * (hy/hx)* beta_n;
                as  = -hy/hx     * ( (1.0/3.0)*beta_n + beta_s);
            }
            /*__________________________________
            * Take care of the corner cells
            *___________________________________*/
            if ( i == xLoLimit && j == yLoLimit )       /* Lower Left Corner */
            {               
                as  = -(8.0/3.0) * (hy/hx) * beta_s;
                an  = -(hy/hx)   * ((1.0/3.0)*beta_s + beta_n);
               
                aw  = -(8.0/3.0) * (hy/hx) * beta_w;
                ae  = -hy/hx     * (beta_e + (1.0/3.0)*beta_w);
            } 
            else if ( i == xLoLimit && j == yHiLimit )  /* Upper Left Corner */
            {                
                as  = -(hy/hx)   * ((1.0/3.0)*beta_n + beta_s);
                an  = -(8.0/3.0) * (hy/hx) * beta_n;
               
                aw  = -(8.0/3.0) * (hy/hx) * beta_w;
                ae  = -hy/hx     * (beta_e + (1.0/3.0)*beta_w);
            } 
            else  if ( i == xHiLimit && j == yLoLimit )   /* Lower Right Corner */
            {
                as  = -(8.0/3.0) * (hy/hx) * beta_s;
                an  = -hy/hx     * ((1.0/3.0)*beta_s + beta_n);

                aw  = -hy/hx     * ((1.0/3.0)*beta_e  + beta_w);
                ae  = -(8.0/3.0) * (hy/hx) * beta_e;

            } 
            else if ( i == xHiLimit && j == yHiLimit )  /* Upper Right Corner */
            {
                as  = -(hy/hx)   * ((1.0/3.0)*beta_n + beta_s);
                an  = -(8.0/3.0) * (hy/hx) * beta_n;

                aw  = -(hy/hx)   * ((1.0/3.0)*beta_e + beta_w);
                ae  = -(8.0/3.0) * (hy/hx) * beta_w;
            }
            /*__________________________________
            * Finally set the stencil
            *___________________________________*/
            VecSetValue( stencil->ap, indx, ap, INSERT_VALUES ); 
            VecSetValue( stencil->ae, indx, ae, INSERT_VALUES ); 
            VecSetValue( stencil->aw, indx, aw, INSERT_VALUES ); 
            VecSetValue( stencil->an, indx, an, INSERT_VALUES ); 
            VecSetValue( stencil->as, indx, as, INSERT_VALUES ); 
            indx++;
    }
  }
/*__________________________________
*   Testing
*___________________________________*/  
#if switchDebug_pcgmg_test
    #define switchInclude_stencil_test_code 1
    #include "testcode_PressureSolve.i"
    #undef switchInclude_stencil_test_code
#endif 
/*__________________________________
*   Assemble the matrix
*___________________________________*/
  ierr = MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY);                                          CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY);                                            CHKERRQ(ierr);
/*__________________________________
*   Quite fullwarn remarks in a way that
*   is compiler independent
*___________________________________*/  
    delZ = delZ;        zLoLimit = zLoLimit;        zHiLimit = zHiLimit;        ierr = ierr;
 /*STOP_DOC*/
  }
  
