
/* $Id$ */

/* 
  Context and functions needed for shell preconditioner.
*/

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

