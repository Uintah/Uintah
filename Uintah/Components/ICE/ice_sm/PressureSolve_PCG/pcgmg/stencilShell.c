
/* $Id$ */

/*
   Functions that define a restricted stencil matrix type.
*/
#include "stencilShell.h"
/* -------------------------------------------------------------------*/
#undef __FUNC__
#define __FUNC__ "defineStencilOperator"

int defineStencilOperator( int m, int n, stencilMatrix* stencil, Mat* A )
{
  int i, j, k, ierr;
  int N = m*n;
  Scalar an, as, ae, aw, ap;

  void *ctx = (void*) stencil;

  Scalar hx = 1.0/m, hy = 1.0/n;

    /*__________________________________
    *   Step 3--Create the StencilMatrix Object
    *___________________________________*/ 
  stencil   = (stencilMatrix*) ctx;
  
  ierr      = MatCreateShell( PETSC_COMM_SELF, N, N, N, N, ctx, A );                        CHKERRQ(ierr);
  ierr      = MatShellSetOperation( *A, MATOP_MULT, (void*) stencilMult );                  CHKERRQ(ierr);

  /*
    Define stencil structure member data.
  */
  stencil->m = m;
  stencil->n = n;

  ierr = VecCreateSeq( PETSC_COMM_SELF, N, &(stencil->ap) );                                CHKERRQ(ierr);
  ierr = VecCreateSeq( PETSC_COMM_SELF, N, &(stencil->ae) );                                CHKERRQ(ierr);
  ierr = VecCreateSeq( PETSC_COMM_SELF, N, &(stencil->aw) );                                CHKERRQ(ierr);
  ierr = VecCreateSeq( PETSC_COMM_SELF, N, &(stencil->an) );                                CHKERRQ(ierr);
  ierr = VecCreateSeq( PETSC_COMM_SELF, N, &(stencil->as) );                                CHKERRQ(ierr);

  /* 
    Embedding boundary processing within the sweep over the grid, 
    is not very efficient, since it requires testing whether every
    grid cell is a boundary cell.  A more efficient strategy is to
    set every stencil weights in every cell regardless of whether it
    is a boundary cell, then follow with cleanup loops that properly
    set the boundary cells.  For now, this will do.
  */

  k = 0;
  for ( j=0; j<n; j++ ) {
    for ( i=0; i<m; i++ ) {
      ae = -hy/hx;         /* east   */
      aw = -hy/hx;         /* west   */
      an = -hx/hy;         /* north  */
      as = -hx/hy;         /* south  */
      ap = -(ae+aw+an+as); /* center; conservative! */
      if ( i == 0 ) {
         aw = hy;
         ap = -(ae+an+as);
      } else if ( i == m-1 ) {
         ae = hy;
         ap = -(aw+an+as);
      }  
      if ( j == 0 ) {
         as = hx;
         ap = -(ae+aw+an);
      } else if ( j == n-1 ) {
         an = hx;
         ap = -(ae+aw+as);
      }
      if ( i == 0 && j == 0 ) {
         ap = -(ae+an);
      } else if ( i == 0 && j == n-1 ) {
         ap = -(ae+as);
      } else  if ( i == m-1 && j == 0 ) {
         ap = -(aw+an);
      } else if ( i == m-1 && j == n-1 ) {
         ap = -(aw+as);
      }
      VecSetValue( stencil->ap, k, ap, INSERT_VALUES ); 
      VecSetValue( stencil->ae, k, ae, INSERT_VALUES ); 
      VecSetValue( stencil->aw, k, aw, INSERT_VALUES ); 
      VecSetValue( stencil->an, k, an, INSERT_VALUES ); 
      VecSetValue( stencil->as, k, as, INSERT_VALUES ); 
      k++;
    }
  }

  ierr = MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY);                                          CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY);                                            CHKERRQ(ierr);
 /*__________________________________
*   Quite fullwarn remarks in a way that
*   is compiler independent
*___________________________________*/
    ierr = ierr;
  return 0;
}





/*______________________________________________________________________
*
*_______________________________________________________________________*/

#include "pcgmg.h"
/* ------------------------------------------------------------------- */
#undef __FUNC__
#define __FUNC__ "stencilMult"

int stencilMult( Mat A, Vec x, Vec y )
{
  int ierr, m, n;

  Scalar *pap, *pae, *paw, *pan, *pas;
  Scalar *px, *py;

  void *ctx;

  stencilMatrix* stencil;

  /* 
    Grab context associated with stencil shell.
  */
  ierr = MatShellGetContext( A, &ctx ); CHKERRQ(ierr);
  stencil = (stencilMatrix *) ctx;

  m = stencil->m; n = stencil->n;

  /*
    Grab pointers of data arrays associated 
    with input arguments and stencil weights.
  */
  ierr = VecGetArray(x, &px); CHKERRQ(ierr);
  ierr = VecGetArray(y, &py); CHKERRQ(ierr);
  ierr = VecGetArray(stencil->ap, &pap); CHKERRQ(ierr);
  ierr = VecGetArray(stencil->ae, &pae); CHKERRQ(ierr);
  ierr = VecGetArray(stencil->aw, &paw); CHKERRQ(ierr);
  ierr = VecGetArray(stencil->an, &pan); CHKERRQ(ierr);
  ierr = VecGetArray(stencil->as, &pas); CHKERRQ(ierr);

  /*
    Stencil form of matrix-vector product.
  */
  FORT_STENCILMULT( px, py, pap, pae, paw, pan, pas, &m, &n );

  /*
    Restore data to respective vectors.
  */
  ierr = VecRestoreArray(x, &px); CHKERRQ(ierr);
  ierr = VecRestoreArray(y, &py); CHKERRQ(ierr);
  ierr = VecRestoreArray(stencil->ap, &pap); CHKERRQ(ierr);
  ierr = VecRestoreArray(stencil->ae, &pae); CHKERRQ(ierr);
  ierr = VecRestoreArray(stencil->aw, &paw); CHKERRQ(ierr);
  ierr = VecRestoreArray(stencil->an, &pan); CHKERRQ(ierr);
  ierr = VecRestoreArray(stencil->as, &pas); CHKERRQ(ierr);
 /*__________________________________
*   Quite fullwarn remarks in a way that
*   is compiler independent
*___________________________________*/
    ierr = ierr;
  return 0;
}
