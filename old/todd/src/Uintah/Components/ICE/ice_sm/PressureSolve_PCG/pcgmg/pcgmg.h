/*______________________________________________________________________
*      Context and functions needed for shell matrix.
*_______________________________________________________________________*/
#ifndef STENCIL_SHELL_H
    #define STENCIL_SHELL_H

    #include "sles.h"

    #define FORT_STENCILMULT stencilmult_

    typedef struct {
       Vec as;
       Vec aw;
       Vec ap;
       Vec ae;
       Vec an;
       int m;
       int n;
    } stencilMatrix;
    extern int stencilMult( Mat A, Vec x, Vec y );
    extern int stencilSetValues( Mat A, int row, int* nrows, int col, int* ncols, Scalar* values, InsertMode mode );
    extern int defineStencilOperator( int m, int n, stencilMatrix* stencil, Mat* A );
extern void FORT_STENCILMULT( Scalar*, Scalar*, Scalar*, Scalar*,Scalar*, Scalar*, Scalar*, int*, int* );
 
#endif


/*______________________________________________________________________
*     Context and functions needed for shell preconditioner.
*_______________________________________________________________________*/

#ifndef MULTIGRID_SHELL_H
    #define MULTIGRID_SHELL_H

    #include "sles.h"
    #include "stencilShell.h"

    #define FORT_PSET pset_
    #define FORT_PSOL psol_

    typedef struct {
       int m;
       int n;
       int mgLevels;
    } MultigridPreconditioner;

    extern int MultigridPreconditionerCreate( MultigridPreconditioner** );
    extern int MultigridPreconditionerSetUp( MultigridPreconditioner*, Mat, Vec );
    extern int MultigridPreconditionerApply( void*, Vec, Vec );
    extern int MultigridPreconditionerDestroy( MultigridPreconditioner* );
    extern void FORT_PSET( int*, int*, int*, Scalar*, Scalar*, Scalar*, Scalar*, Scalar* );
    extern void FORT_PSOL( Scalar*, Scalar*, int*, int* );

#endif

/*______________________________________________________________________
*   Header stuff from the main program
*_______________________________________________________________________*/

typedef struct {
   Vec                     x, b;      /* solution vector, right-hand-side vector                */
   Mat                     A;         /* sparse matrix                                          */
   stencilMatrix           stencil;   /* stencil representation of A, stored as context in shell*/
   SLES                    sles;      /* linear solver context                                  */
   MultigridPreconditioner *MGshell;  /* preconditioner shell                                   */
   int                     m, n;      /* grid dimensions                                        */
   Scalar                  hx2, hy2;  /* 1/(m+1)*(m+1) and 1/(n+1)*(n+1)                        */
} UserCtx;


extern int FinalizeLinearSolver(UserCtx *);
extern int Solve(UserCtx *userctx, Scalar *b, Scalar *x);
