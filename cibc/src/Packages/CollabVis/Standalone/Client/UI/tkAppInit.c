/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/* 
 * tkAppInit.c --
 *
 *	Provides a default version of the Tcl_AppInit procedure for
 *	use in wish and similar Tk-based applications.
 *
 * Copyright (c) 1993 The Regents of the University of California.
 * Copyright (c) 1994-1997 Sun Microsystems, Inc.
 *
 * See the file "license.terms" for information on usage and redistribution
 * of this file, and for a DISCLAIMER OF ALL WARRANTIES.
 *
 */

#include <sci_config.h>
#define BUILD_tcl
#include <tk.h>
#include "locale.h"
#include <itk.h>
#undef BUILD_tcl

#include <stdio.h>

#ifdef _WIN32
#define SHARE __declspec(dllexport)
#else
#define SHARE
#endif


#ifdef __cplusplus
extern "C" {
#endif
SHARE Tcl_Interp* the_interp;
#ifdef __cplusplus
}
#endif



/*
 * The following declarations refer to internal Tk routines.  These
 * interfaces are available for use, but are not supported.
 */

EXTERN void		TkConsoleCreate(void);
EXTERN int		TkConsoleInit(Tcl_Interp *interp);

extern int OpenGLCmd _ANSI_ARGS_((ClientData clientData,
	Tcl_Interp *interp, int argc, char **argv));
extern int BevelCmd _ANSI_ARGS_((ClientData clientData,
	Tcl_Interp *interp, int argc, char **argv));
/* extern int Tk_RangeCmd _ANSI_ARGS_((ClientData clientData, */
/* 	Tcl_Interp *interp, int argc, char **argv)); */
extern int Tk_CursorCmd _ANSI_ARGS_((ClientData clientData,
	Tcl_Interp *interp, int argc, char **argv));
extern int BLineInit _ANSI_ARGS_((void));

#ifdef _WIN32 
#define EXPORT  __declspec(dllimport)
#else
#define EXPORT
#endif

extern EXPORT int Blt_SafeInit _ANSI_ARGS_((Tcl_Interp *interp));
extern EXPORT int Blt_Init _ANSI_ARGS_((Tcl_Interp *interp));
extern int Table_Init _ANSI_ARGS_((Tcl_Interp* interp));

/* include tclInt.h for access to namespace API */
#include "tclInt.h"

/*
 * The following variable is a special hack that is needed in order for
 * Sun shared libraries to be used for Tcl.
 */

#ifndef _WIN32
extern int matherr();
int *tclDummyMathPtr = (int *) matherr;
#endif

#ifdef TK_TEST
EXTERN int		Tcltest_Init _ANSI_ARGS_((Tcl_Interp *interp));
EXTERN int		Tktest_Init _ANSI_ARGS_((Tcl_Interp *interp));
#endif /* TK_TEST */


static void (*wait_func)(void*);
static void* wait_func_data;
/*
 *----------------------------------------------------------------------
 *
 * main --
 *
 *	This is the main program for the application.
 *
 * Results:
 *	None: Tk_Main never returns here, so this procedure never
 *	returns either.
 *
 * Side effects:
 *	Whatever the application does.
 *
 *----------------------------------------------------------------------
 */

SHARE int
tkMain(argc, argv, nwait_func, nwait_func_data)
    int argc;			/* Number of command-line arguments. */
    char **argv;		/* Values of command-line arguments. */
    void (*nwait_func)(void*);
    void* nwait_func_data;
{
    wait_func=nwait_func;
    wait_func_data=nwait_func_data;
#ifdef _WIN32
    /*
     * Create the console channels and install them as the standard
     * channels.  All I/O will be discarded until TkConsoleInit is
     * called to attach the console to a text widget.
     */

		printf("Calling TkConsoleCreate\n");
    TkConsoleCreate();
#endif
    Tk_Main(argc, argv, Tcl_AppInit);
    return 0;			/* Needed only to prevent compiler warning. */
}

/*
 *----------------------------------------------------------------------
 *
 * Tcl_AppInit --
 *
 *	This procedure performs application-specific initialization.
 *	Most applications, especially those that incorporate additional
 *	packages, will have their own version of this procedure.
 *
 * Results:
 *	Returns a standard Tcl completion code, and leaves an error
 *	message in interp->result if an error occurs.
 *
 * Side effects:
 *	Depends on the startup script.
 *
 *----------------------------------------------------------------------
 */

SHARE int
Tcl_AppInit(interp)
    Tcl_Interp *interp;		/* Interpreter for application. */
{
    the_interp=interp;

    printf("Initializing the tcl packages: ");
    fflush(stdout);

    printf("tcl, ");
    if (Tcl_Init(interp) == TCL_ERROR) {
      printf("Tcl_Init() failed\n");
	return TCL_ERROR;
    }
    fflush(stdout);
    printf("tk, ");
    if (Tk_Init(interp) == TCL_ERROR) {
      printf("Tk_Init() failed.  Is the DISPLAY environment variable set properly?\n");
	return TCL_ERROR;
    }
    fflush(stdout);
    Tcl_StaticPackage(interp, "Tk", Tk_Init, Tk_SafeInit);

#ifdef _WIN32
    /*
     * Initialize the console only if we are running as an interactive
     * application.
     */
	printf("Calling TkConsoleInit\n");
    if (TkConsoleInit(interp) == TCL_ERROR) {
		printf("Error in TkConsoleInit\n");
	   return TCL_ERROR;
    }
#endif


#ifdef TK_TEST
    if (Tktest_Init(interp) == TCL_ERROR) {
	return TCL_ERROR;
    }
    Tcl_StaticPackage(interp, "Tktest", Tktest_Init,
            (Tcl_PackageInitProc *) NULL);
#endif /* TK_TEST */


    /*
     * Call the init procedures for included packages.  Each call should
     * look like this:
     *
     * if (Mod_Init(interp) == TCL_ERROR) {
     *     return TCL_ERROR;
     * }
     *
     * where "Mod" is the name of the module.
     */
    printf("itcl, ");
    if (Itcl_Init(interp) == TCL_ERROR) {
      printf("Itcl_Init() failed\n");
        return TCL_ERROR;
    }
    fflush(stdout);
    printf("itk, ");
    if (Itk_Init(interp) == TCL_ERROR) {
      printf("Itk_Init() failed\n");
        return TCL_ERROR;
    }
    fflush(stdout);
    printf("blt, ");
    if (Blt_Init(interp) == TCL_ERROR) {
      printf("Blt_Init() failed\n");
	return TCL_ERROR;
    }

    Tcl_StaticPackage(interp, "Itcl", Itcl_Init, Itcl_SafeInit);
    Tcl_StaticPackage(interp, "Itk", Itk_Init, (Tcl_PackageInitProc *) NULL);
    Tcl_StaticPackage(interp, "BLT", Blt_Init, Blt_SafeInit);

    printf("Done.\n");
    fflush(stdout);

    /*
     *  This is itkwish, so import all [incr Tcl] commands by
     *  default into the global namespace.  Fix up the autoloader
     *  to do the same.
     */
    if (Tcl_Import(interp, Tcl_GetGlobalNamespace(interp),
            "::itk::*", /* allowOverwrite */ 1) != TCL_OK) {
        return TCL_ERROR;
    }

    if (Tcl_Import(interp, Tcl_GetGlobalNamespace(interp),
            "::itcl::*", /* allowOverwrite */ 1) != TCL_OK) {
        return TCL_ERROR;
    }

    if (Tcl_Eval(interp, "auto_mkindex_parser::slavehook { _%@namespace import -force ::itcl::* ::itk::* }") != TCL_OK) {
        return TCL_ERROR;
    }

    /*
     * Call Tcl_CreateCommand for application-specific commands, if
     * they weren't already created by the init procedures called above.
     */

    /*
     * Call Tcl_CreateCommand for application-specific commands, if
     * they weren't already created by the init procedures called above.
     */

#ifdef _WIN32
#define PARAMETERTYPE int*
#else
#define PARAMETERTYPE
#endif

    printf("Adding SCI extensions to tcl: ");
    fflush(stdout);

#ifdef SCI_OPENGL
    printf("OpenGL widget, ");
    Tcl_CreateCommand(interp, "opengl", OpenGLCmd, (ClientData) Tk_MainWindow(interp),
		      (void (*)(PARAMETERTYPE)) NULL);
#endif
    fflush(stdout);
    printf("bevel widget, ");
    Tcl_CreateCommand(interp, "bevel", BevelCmd, (ClientData) Tk_MainWindow(interp),
		      (void (*)(PARAMETERTYPE)) NULL);
/*     Tcl_CreateCommand(interp, "range", Tk_RangeCmd, (ClientData) Tk_MainWindow(interp), */
/*                       (void (*)(PARAMETERTYPE)) NULL); */
    fflush(stdout);
    printf("cursor, ");
    Tcl_CreateCommand(interp, "cursor", Tk_CursorCmd,
		      (ClientData) Tk_MainWindow(interp), NULL);

    printf("Done.\n");
    fflush(stdout);

    /*
     * Initialize the BLine Canvas item
     */

    BLineInit();

    /*
     * Specify a user-specific startup file to invoke if the application
     * is run interactively.  Typically the startup file is "~/.apprc"
     * where "app" is the name of the application.  If this line is deleted
     * then no user-specific startup file will be run under any conditions.
     */

    Tcl_SetVar(interp, "tcl_rcFileName", "~/.scirc", TCL_GLOBAL_ONLY);
    (*wait_func)(wait_func_data);
    return TCL_OK;
}












