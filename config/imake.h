
/*
 * config/imake.h
 * 
 * Written by:
 *  Steven G. Parker
 *  Feb, 1994
 */

#include "variant.h"

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
#ifdef SCI_COMPILER_DELTA
#include "delta.h"
#endif

/* Optimization variants */
#ifdef SCI_VARIANT_DEBUG
#include "debug.h"
#endif
#ifdef SCI_VARIANT_OPT
#include "opt.h"
#endif

