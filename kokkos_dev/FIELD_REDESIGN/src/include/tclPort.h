/*
 * tclPort.h --
 *
 *	This header file handles porting issues that occur because
 *	of differences between systems.  It reads in platform specific
 *	portability files.
 *
 * Copyright (c) 1994-1995 Sun Microsystems, Inc.
 *
 * See the file "license.terms" for information on usage and redistribution
 * of this file, and for a DISCLAIMER OF ALL WARRANTIES.
 *
 * RCS: @(#) $Id$
 */

#ifndef _TCLPORT
#define _TCLPORT

#if defined(__WIN32__) || defined(_WIN32)
#   include "../win/tclWinPort.h"
#else
#   if defined(MAC_TCL)
#	include "tclMacPort.h"
#    else
#	include "tclUnixPort.h"
#    endif
#endif

#endif /* _TCLPORT */
