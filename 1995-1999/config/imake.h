
/*
 * config/imake.h
 * 
 * Written by:
 *  Steven G. Parker
 *  Feb, 1994
 */

#include "config_imake.h"

/* machine variants */
#ifdef SCI_MACHINE_SGI
#include "sgi.h"
#endif
#ifdef SCI_MACHINE_Solaris
#include "solaris.h"
#endif
#ifdef SCI_MACHINE_Linux
#include "linux.h"
#endif

/* Compiler variants */
#ifdef SCI_COMPILER_CFront
#include "cfront.h"
#endif
#ifdef SCI_COMPILER_GNU
#include "gnu.h"
#endif
#ifdef SCI_COMPILER_Delta
#include "delta.h"
#endif

/* Optimization variants */
#ifdef SCI_VARIANT_Debug
#include "debug.h"
#endif
#ifdef SCI_VARIANT_Optimized
#include "opt.h"
#endif

