
/* $Id$ */

/*
   Functions that define interface between multigrid preconditioner and PETSc.
*/

#include "pcgmg.h"

/* ------------------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "MultigridPreconditionerCreate"

int MultigridPreconditionerCreate(MultigridPreconditioner **MGshell)
{
  /*
    Allocate and initialize space for shell preconditioner.
  */
  MultigridPreconditioner *newctx = PetscNew(MultigridPreconditioner); CHKPTRQ(newctx);
  newctx->m = 0;
  newctx->n = 0;
  newctx->mgLevels = 0;
  *MGshell = newctx;

  return 0;
}

/* ------------------------------------------------------------------- */
#undef __FUNC__  
#define __FUNC__ "MultigridPreconditionerSetUp"

int MultigridPreconditionerSetUp(MultigridPreconditioner *MGshell, Mat pmat, Vec x)
{
  int ierr;
  int m = MGshell->m;
  int n = MGshell->n;
  int levels = MGshell->mgLevels;

  void *ctx = 0;

  Scalar *pap;
  Scalar *pae;
  Scalar *paw;
  Scalar *pan;
  Scalar *pas;

  stencilMatrix *stencil;

  /*
    Grab user context associated with matrix.
  */
  ierr = MatShellGetContext( pmat, &ctx );
  stencil = (stencilMatrix*) ctx;

  /*
    Grab pointers to vectors containing stencil weights.
  */
  ierr = VecGetArray( stencil->ap, &pap ); CHKERRQ(ierr);
  ierr = VecGetArray( stencil->ae, &pae ); CHKERRQ(ierr);
  ierr = VecGetArray( stencil->aw, &paw ); CHKERRQ(ierr);
  ierr = VecGetArray( stencil->an, &pan ); CHKERRQ(ierr);
  ierr = VecGetArray( stencil->as, &pas ); CHKERRQ(ierr);
  
  /*
    Initialize preconditioner.
  */
  FORT_PSET( &m, &n, &levels, pap, pae, paw, pan, pas ); 
  
  /*
    Restore data to respective vectors.
  */
  ierr = VecRestoreArray( stencil->ap, &pap ); CHKERRQ(ierr);
  ierr = VecRestoreArray( stencil->ae, &pae ); CHKERRQ(ierr);
  ierr = VecRestoreArray( stencil->aw, &paw ); CHKERRQ(ierr);
  ierr = VecRestoreArray( stencil->an, &pan ); CHKERRQ(ierr);
  ierr = VecRestoreArray( stencil->as, &pas ); CHKERRQ(ierr);
 /*__________________________________
*   Quite fullwarn remarks in a way that
*   is compiler independent
*___________________________________*/
    ierr = ierr;
  return 0;
}

/* ------------------------------------------------------------------- */
#undef __FUNC__  
#define __FUNC__ "MultigridPreconditionerApply"

int MultigridPreconditionerApply(void *ctx, Vec x, Vec y)
{
  int ierr;
  int m;
  int n;
  MultigridPreconditioner *MGshell = (MultigridPreconditioner*) ctx;
  Scalar *px, *py;

  m = MGshell->m; n = MGshell->n;
 
  /* 
    Grab pointers of data arrays associated with input arguments.
  */
  ierr = VecGetArray(x, &px); CHKERRQ(ierr);
  ierr = VecGetArray(y, &py); CHKERRQ(ierr);

  /*
    Perform stencil-based multigrid preconditioning.
  */
  FORT_PSOL( px, py, &m, &n ); 

  /*
    Restore data to respective vectors.
  */
  ierr = VecRestoreArray(x, &px); CHKERRQ(ierr);
  ierr = VecRestoreArray(y, &py); CHKERRQ(ierr);
 /*__________________________________
*   Quite fullwarn remarks in a way that
*   is compiler independent
*___________________________________*/
    ierr = ierr;

  return 0;
}

/* ------------------------------------------------------------------- */
#undef __FUNC__  
#define __FUNC__ "MultigridPreconditionerDestroy"

int MultigridPreconditionerDestroy(MultigridPreconditioner *MGshell)
{
  PetscFree(MGshell);

  return 0;
}
