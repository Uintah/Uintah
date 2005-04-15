
/* $Id$ */

#include "sles.h"
#include <math.h>

#include "stencilShell.h"
#include "multigridShell.h"

/*
    User-defined context that contains all the data structures used
    in the linear solution process.
*/
typedef struct {
   Vec                     x, b;      /* solution vector, right-hand-side vector */
   Mat                     A;         /* sparse matrix */
   stencilMatrix           stencil;   /* stencil representation of A, stored as context in shell */
   SLES                    sles;      /* linear solver context */
   MultigridPreconditioner *MGshell;  /* preconditioner shell */
   int                     m, n;      /* grid dimensions */
   Scalar                  hx2, hy2;  /* 1/(m+1)*(m+1) and 1/(n+1)*(n+1) */
} UserCtx;

extern int InitializeLinearSolver(int, int, UserCtx *);
extern int FinalizeLinearSolver(UserCtx *);
extern int Solve(UserCtx *userctx, Scalar *b, Scalar *x);

#undef __FUNC__  
#define __FUNC__ "main"




/*______________________________________________________________________
*
*_______________________________________________________________________*/
int main(int argc,char **args)
{
  UserCtx userctx;
  int     ierr, m = 4, n = 4, flg, i, j, k, N, its;
  Scalar  hx, hy, x, y, v, MINUSONE = -1.0;
  Vec error, solution;
  double  enorm;
    
    /*__________________________________
    *   Step 1--Initialize the PETSC libraries
    *___________________________________*/ 
  PetscInitialize(&argc, &args, (char *)0, (char *)0);

  /*
     Set the grid size from command line arguments.
  */
  ierr = OptionsGetInt(PETSC_NULL, "-m", &m, &flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL, "-n", &n, &flg); CHKERRA(ierr);

/*__________________________________
*   STEPS 2 through 5
*___________________________________*/
  ierr = InitializeLinearSolver(m, n, &userctx); CHKERRA(ierr);
  N    = m*n;

  /*
     Fill the right-hand-side b[] and the solution with a known problem for testing.
  */
  ierr = VecCreateSeq(PETSC_COMM_SELF, N, &solution); CHKERRQ(ierr);
  hx = 1.0/m; 
  hy = 1.0/n;
  y  = hy*0.5;
  k  = 0;
  for ( j=0; j<n; j++ ) {
    x = hx*0.5;
    for ( i=0; i<m; i++ ) {
      v = (4.0*pow(x,3.0) - 6.0*pow(x,2.0) + 1.0)*(4.0*pow(y,3.0) - 6.0*pow(y,2.0) + 1.0);
      VecSetValue( solution, k, v, INSERT_VALUES );
      v = -12.0*hx*hy*((2.0*x - 1.0)*(4.0*pow(y,3.0) - 6.0*pow(y,2.0) + 1.0) +
                       (4.0*pow(x,3.0) - 6.0*pow(x,2.0) + 1.0)*(2.0*y - 1.0));
      VecSetValue( userctx.b, k, v, INSERT_VALUES );
      x += hx;
      k++;
    }
    y += hy;
  }

  /* 
     Set runtime options, e.g.,
        -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
     These options will override those specified above as long as
     SLESSetFromOptions() is called _after_ any other customization
     routines.
 
     Run the program with the option -help to see all the possible
     linear solver options.
  */
  ierr = SLESSetFromOptions(userctx.sles); CHKERRQ(ierr);

/*__________________________________
*   Step 11--Solve the system
*___________________________________*/
  ierr = SLESSolve(userctx.sles, userctx.b, userctx.x, &its); CHKERRQ(ierr);
  
  /*
      Compute error.
  */
  ierr = VecCreateSeq(PETSC_COMM_SELF, N, &error); CHKERRQ(ierr);
  ierr = VecWAXPY( &MINUSONE, userctx.x, solution, error ); CHKERRQ(ierr);
  ierr = VecNorm(error, NORM_INFINITY, &enorm );

  printf("m %d n %d iterations %d error norm %g\n", m, n, its, enorm);

/*__________________________________
*   Step 12 Deallocate memory
*___________________________________*/
  PetscFree(solution);
  PetscFree(error);
  ierr = FinalizeLinearSolver(&userctx); CHKERRA(ierr);
  PetscFinalize();

  return 0;
}


/*______________________________________________________________________
*
*_______________________________________________________________________*/
#undef __FUNC__  
#define __FUNC__ "InitializeLinearSolver"
/* ------------------------------------------------------------------------*/
int InitializeLinearSolver(int m, int n, UserCtx *userctx)
{
  int N, flg, ierr;
  Mat *A = &(userctx->A);
  stencilMatrix* stencil = &(userctx->stencil);
  PC pc;

  /*
     Computational grid is an m x n cell-centered grid.
  */
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
    *   Details are burried within
    *___________________________________*/
    ierr = defineStencilOperator( m, n, stencil, A );                                       CHKERRQ(ierr);

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

  return 0;
}




/*______________________________________________________________________
*    FinalizeLinearSolver(UserCtx *userctx)   
*_______________________________________________________________________*/
#undef __FUNC__  
#define __FUNC__  "FinalizeLinearSolver"
/* ------------------------------------------------------------------------*/
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

  return 0;
}
