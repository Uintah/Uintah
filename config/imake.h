
/*
 * config/imake.h
 * 
 * Written by:
 *  Steven G. Parker
 *  Feb, 1994
 */

#include "../variant.h"

/* machine variants */
#ifdef SCI_MACHINE_SGI
#include "sgi.h"
#endif
#ifdef SCI_MACHINE_SOLARIS
#include "solaris.h"
#endif
#ifdef SCI_MACHINE_LINUX
#include "linux.h"
#endif

/* Compiler variants */
#ifdef SCI_COMPILER_CFRONT
#include "cfront.h"
#endif
#ifdef SCI_COMPILER_GNU
#include "gnu.h"
#endif

/* Optimization variants */
#ifdef SCI_VARIANT_DEBUG
#include "debug.h"
#endif
#ifdef SCI_VARIANT_OPT
#include "opt.h"
#endif


#ifdef SCI_COMPILER_CFRONT
#define OBJEXT1 nat
#endif
#ifdef SCI_COMPILER_GNU
#define OBJEXT1 gnu
#endif

#ifdef SCI_VARIANT_DEBUG
#define SCI_OBJEXT2 debug
#endif
#ifdef SCI_VARIANT_OPT
#define SCI_OBJEXT2 opt
#endif

