/* 
 ======================================================================*/
#include <math.h>
#include <assert.h>
#include <limits.h>
#include "nrutil+.h"
#include "parameters.h"
#include "functionDeclare.h"
#include "switches.h"
#include "macros.h"

#include "pcgmg.h"
#include "sles.h"
/*---------------------------------------------------------------------  
 Function:  compute_delta_Press_Using_PCGMG--PRESS: main driver for computing delta_pressure.
 Filename:  pressure_PCG.c
 Purpose:  
    To be filled in
    
 Computational Domain:          Interior cells 
 Ghostcell data dependency:     

 References:
             
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       12/17/99
 ---------------------------------------------------------------------  */

 void   compute_delta_Press_Using_PCGMG(
        int     xLoLimit,                   /* x-array lower limit              */
        int     yLoLimit,                   /* y-array lower limit              */
        int     zLoLimit,                   /* z-array lower limit              */
        int     xHiLimit,                   /* x-array upper limit              */
        int     yHiLimit,                   /* y-array upper limit              */
        int     zHiLimit,                   /* z-array upper limit              */
        double  delX,                       /* distance/cell, xdir              (INPUT) */
        double  delY,                       /* distance/cell, ydir              (INPUT) */
        double  delZ,                       /* distance/cell, zdir              (INPUT) */
        double  delt,
        double  ****rho_CC,                 /* Cell-centered density            */
        double  ****speedSound,             /* speed of sound (x,y,z, material) */
            /*------to be treated as pointers---*/
                                            /*______(x,y,z,face, material)______*/
        double  ******uvel_FC,              /* u-face-centered velocity         */
        double  ******vvel_FC,              /* *v-face-centered velocity        */
        double  ******wvel_FC,              /* w face-centered velocity         */
        double  ****delPress_CC,            /* change in delta p                */
        double  ****press_CC,
        int     ***BC_types,                /* array containing the different   (INPUT) */
                                            /* types of boundary conditions             */
                                            /* BC_types[wall][variable]=type            */
        int     nMaterials)
 {
    static int    initialized=0;        /* Flag for initializing Petsc      */
    char*         args[2];
    char**        argv;
    int           argc=0;
    
    Scalar        *array;    
    
  UserCtx userctx;
    int           ierr, m, n, i, j, k, N, index, its, mat;
    double        enorm;
    Scalar        hx, hy, x, y, v, MINUSONE = -1.0;
    Vec           error, solution;

    KSP          kspctx;                /* Krylov subspace method context       */    
/*__________________________________
*   Plotting variables
*___________________________________*/
#if switchDebug_pressure_PCG 
    double
                ***plot_1,              /* testing array                        */      
                ***plot_2,              /* testing array                        */
                ***plot_3,              /* testing array                        */      
                ***plot_4,              /* testing array                        */ 
                ***plot_5,              /* testing array                        */ 
                ***plot_6;
    Scalar      *array_1,               /* used to convert from petsc to        */
                *array_2,               /* to my data arrays for plotting only*/
                *array_3,
                *array_4,
                *array_5,
                *array_6;
                       
    #include "plot_declare_vars.h" 
    plot_1  = darray_3d(0, X_MAX_LIM,   0, Y_MAX_LIM, 0, Z_MAX_LIM);    
    plot_2  = darray_3d(0, X_MAX_LIM,   0, Y_MAX_LIM, 0, Z_MAX_LIM);
    plot_3  = darray_3d(0, X_MAX_LIM,   0, Y_MAX_LIM, 0, Z_MAX_LIM);    
    plot_4  = darray_3d(0, X_MAX_LIM,   0, Y_MAX_LIM, 0, Z_MAX_LIM);
    plot_5  = darray_3d(0, X_MAX_LIM,   0, Y_MAX_LIM, 0, Z_MAX_LIM); 
    plot_6  = darray_3d(0, X_MAX_LIM,   0, Y_MAX_LIM, 0, Z_MAX_LIM);
 
#endif
/*______________________________________________________________________
*   initialize variables and allocate some temp memory
*_______________________________________________________________________*/
    mat = 1;
    m   = xHiLimit - xLoLimit + 1;
    n   = yHiLimit - yLoLimit + 1;
    N   = m*n;
/*__________________________________
*   Initialize the PETSC libraries
*   Do this only once
*___________________________________*/ 
  if(!initialized)
  {
     PetscInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL); 

     initialized= 1;
  }
/*__________________________________
*   Initialize the solver and compute
*   the stencil matrix
*___________________________________*/
    ierr = initializeLinearSolver(
                        xLoLimit,       yLoLimit,       zLoLimit,             
                        xHiLimit,       yHiLimit,       zHiLimit,             
                        delX,           delY,           delZ,                 
                        delt,           rho_CC,         speedSound,       
                        BC_types,       nMaterials,     &userctx);                          CHKERRA(ierr);
                        
/*__________________________________
* Calculate the rhs
*___________________________________*/
    calc_delPress_RHS(
                        xLoLimit,       yLoLimit,       zLoLimit,             
                        xHiLimit,       yHiLimit,       zHiLimit,             
                        delX,           delY,           delZ,               
                        uvel_FC,        vvel_FC,        wvel_FC,
                        &userctx,       &solution,     
                        nMaterials);

/*__________________________________
*   Solve the system
*___________________________________*/     
    ierr = SLESGetKSP(userctx.sles,&kspctx);                                                CHKERRA(ierr);
     ierr = KSPSetTolerances(kspctx,RELATIVE_TOLERANCE,
                                    ABSOLUTE_TOLERANCE, DIV_TOLERANCE, MAX_ITERATION);        CHKERRA(ierr);

    ierr = SLESSolve(userctx.sles, userctx.b, userctx.x, &its);                             CHKERRQ(ierr);
  
/*__________________________________
*   Compute error using test code
*___________________________________*/
#if switchDebug_pcgmg_test
    #define switchInclude_compute_error 1
    #include "testcode_PressureSolve.i"
    #undef switchInclude_compute_error
#endif 

/*__________________________________
*   Now extract the values of delPress
*   from PETSC
*___________________________________*/
    mat = 1;
    ierr = VecGetArray(userctx.x,&array);                                                   CHKERRA(ierr);


    for ( k = zLoLimit; k <= zHiLimit; k++)
    {
        for ( j = yLoLimit; j <= yHiLimit; j++)
        {
            for ( i = xLoLimit; i <= xHiLimit; i++)
            { 
            /*__________________________________
            * map a 3d array to a 1d vector
            *___________________________________*/
            index = (i-xLoLimit) + (j-yLoLimit)*(xHiLimit-xLoLimit+1);                  /* 2D       */
            /* index = index + (k - zLoLimit)*(xHiLimit-xLoLimit+1)*(yHiLimit-yLoLimit+1); */ /* 3D       */ 
            
            delPress_CC[i][j][k][mat]   = array[index];
            press_CC[i][j][k][mat]      = press_CC[i][j][k][mat]    + array[index];
             
            }
        }
    }
    ierr = VecRestoreArray(userctx.x,&array);                                               CHKERRA(ierr);

/*__________________________________
*   Plotting section
*___________________________________*/ 
#if switchDebug_pressure_PCG
         #define switchInclude_pressure_PCG 1
         #include "debugcode.i"
         free_darray_3d( plot_1,0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
         free_darray_3d( plot_2,0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
         free_darray_3d( plot_3,0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
         free_darray_3d( plot_4,0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
         free_darray_3d( plot_5,0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
         free_darray_3d( plot_6,0, X_MAX_LIM, 0, Y_MAX_LIM, 0, Z_MAX_LIM);
         #undef switchInclude_pressure_PCG
#endif
/*__________________________________
*   Deallocate memory
*___________________________________*/
#if switchDebug_pcgmg_test
    PetscFree(solution);
    PetscFree(error);
#endif

    ierr = FinalizeLinearSolver(&userctx);                                                  CHKERRA(ierr);
 /*__________________________________
*   Quite fullwarn remarks in a way that
*   is compiler independent
*___________________________________*/
    delX= delX;        
    delY= delY;        
    delZ= delZ; 
    ierr= ierr;
    hx  = hx;
    hy  = hy;
    x   = x;
    y   = y;
    v   = v;
    N   = N;
    args[1]= args[1];
    zLoLimit= zLoLimit;                    
    zHiLimit= zHiLimit;
    enorm   = enorm;
    error   = error;
    MINUSONE = MINUSONE;   
/*STOP_DOC*/      
}







/*_______________________________________________________________________*/
#undef __FUNC__  
#define __FUNC__ "InitializeLinearSolver"
/*---------------------------------------------------------------------  
 Function:  initializeLinearSolver--PRESS: to be filled in
 Filename:  pressure_PCG.c
 Purpose:  
    To be filled in
    
 Computational Domain:          Interior cells 
 Ghostcell data dependency:     

 References:
             
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       12/30/99
 ---------------------------------------------------------------------  */
int initializeLinearSolver(
        int     xLoLimit,                   /* x-array lower limit              */
        int     yLoLimit,                   /* y-array lower limit              */
        int     zLoLimit,                   /* z-array lower limit              */
        int     xHiLimit,                   /* x-array upper limit              */
        int     yHiLimit,                   /* y-array upper limit              */
        int     zHiLimit,                   /* z-array upper limit              */
        double  delX,                       /* distance/cell, xdir              (INPUT) */
        double  delY,                       /* distance/cell, ydir              (INPUT) */
        double  delZ,                       /* distance/cell, zdir              (INPUT) */
        double  delt,
        double  ****rho_CC,                 /* Cell-centered density            (INPUT) */
        double  ****speedSound,             /* speed of sound (x,y,z, material) (INPUT) */
        int     ***BC_types,                /* array containing the different   (INPUT) */
                                            /* types of boundary conditions             */
                                            /* BC_types[wall][variable]=type            */
        int     nMaterials,
        UserCtx *userctx)
{
    int m, n, N, flg, ierr;
    Mat *A = &(userctx->A);
    stencilMatrix* stencil = &(userctx->stencil);
    PC pc;
/*______________________________________________________________________
*   Initialize variables
*_______________________________________________________________________*/
    m = xHiLimit - xLoLimit + 1;
    n = yHiLimit - yLoLimit + 1; 
    userctx->m   = m;
    userctx->n   = n;
    userctx->hx2 = m*m;
    userctx->hy2 = n*n; 
    N            = m*n;

    /*__________________________________
    *   Create the vectors for
    *   the solution and the source
    *___________________________________*/
    ierr = VecCreateSeq(PETSC_COMM_SELF, N, &userctx->b);                                   CHKERRQ(ierr);
    ierr = VecDuplicate(userctx->b, &userctx->x);                                           CHKERRQ(ierr);

    /*__________________________________
    *   Create the solver
    *___________________________________*/
    ierr = SLESCreate(PETSC_COMM_SELF, &userctx->sles);                                     CHKERRQ(ierr);
    
    /*__________________________________
    *   Initialize the stencilMatrix
    *   The BC_types array will be used to
    *   determine which typ of BC to implement
    *___________________________________*/
    calc_delPress_Stencil_Weights_Dirichlet(
                        xLoLimit,       yLoLimit,       zLoLimit,             
                        xHiLimit,       yHiLimit,       zHiLimit,             
                        delX,           delY,           delZ,
                        delt,           BC_types,
                        rho_CC,         speedSound,       
                        nMaterials,     stencil,        A);
                        
    calc_delPress_Stencil_Weights_Neuman(
                        xLoLimit,       yLoLimit,       zLoLimit,             
                        xHiLimit,       yHiLimit,       zHiLimit,             
                        delX,           delY,           delZ,                
                        delt,           BC_types, 
                        rho_CC,         speedSound,     nMaterials,     
                        stencil,        A);
   
                        
/*______________________________________________________________________
*           REGISTER PHASE 
*   Register the stencil Matrix with the solver 
*   Eventually move this to a function outside the main loop
*_______________________________________________________________________*/
    ierr = SLESSetOperators(userctx->sles, *A, *A, SAME_NONZERO_PATTERN);                   CHKERRQ(ierr);

    /*__________________________________
    *   Register the 
    *   MultigridPreconditioner with the solver
    *___________________________________*/
    ierr = SLESGetPC(userctx->sles, &pc);                                                   CHKERRQ(ierr);
    ierr = PCSetType(pc, PCSHELL);                                                          CHKERRQ(ierr);

    ierr = MultigridPreconditionerCreate( &(userctx->MGshell) );                            CHKERRA(ierr);
    userctx->MGshell->m = m;
    userctx->MGshell->n = n;
    ierr = OptionsHasName(PETSC_NULL, "-levels", &flg);                                     CHKERRA(ierr);
  
    if ( flg ) 
    {
        ierr = OptionsGetInt(PETSC_NULL, "-levels", &(userctx->MGshell->mgLevels), &flg);   CHKERRQ(ierr);
    }
  
    ierr = PCShellSetApply(pc, MultigridPreconditionerApply, (void*) userctx->MGshell );    CHKERRA(ierr);
                            
    ierr = MultigridPreconditionerSetUp( userctx->MGshell, *A, userctx->x );                CHKERRQ(ierr); 
/*__________________________________
*   Quite fullwarn remarks in a way that
*   is compiler independent
*___________________________________*/
    ierr = ierr;
  return 0;
 /*STOP_DOC*/
}


#undef __FUNC__  
#define __FUNC__  "FinalizeLinearSolver"
/*---------------------------------------------------------------------  
 Function:  finalizeLinearSolver--PRESS: Deallocate memory used by Petsc
 Filename:  pressure_PCG.c
 Purpose:  
    To be filled in 
             
 History:
    Version   Programmer         Date       Description
    -------   ----------         ----       -----------
       1.0     Todd Harman       12/30/99
 ---------------------------------------------------------------------  */
int FinalizeLinearSolver(UserCtx *userctx)
{
  int ierr;
/*__________________________________
*   Deallocate memory
*___________________________________*/
  ierr = SLESDestroy(userctx->sles);                                                        CHKERRQ(ierr);
  ierr = VecDestroy(userctx->x);                                                            CHKERRQ(ierr);
  ierr = VecDestroy(userctx->b);                                                            CHKERRQ(ierr);  
  ierr = MatDestroy(userctx->A);                                                            CHKERRQ(ierr);
  ierr = MultigridPreconditionerDestroy(userctx->MGshell);                                  CHKERRQ(ierr);
  ierr = ierr;
  return 0;
/*STOP_DOC*/
}
