/* 
 * tkAppInit.c --
 *
 *	Provides a default version of the Tcl_AppInit procedure for
 *	use in wish and similar Tk-based applications.
 *
 * Copyright (c) 1993 The Regents of the University of California.
 * All rights reserved.
 *
 * Permission is hereby granted, without written agreement and without
 * license or royalty fees, to use, copy, modify, and distribute this
 * software and its documentation for any purpose, provided that the
 * above copyright notice and the following two paragraphs appear in
 * all copies of this software.
 * 
 * IN NO EVENT SHALL THE UNIVERSITY OF CALIFORNIA BE LIABLE TO ANY PARTY FOR
 * DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT
 * OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF THE UNIVERSITY OF
 * CALIFORNIA HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * THE UNIVERSITY OF CALIFORNIA SPECIFICALLY DISCLAIMS ANY WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
 * AND FITNESS FOR A PARTICULAR PURPOSE.  THE SOFTWARE PROVIDED HEREUNDER IS
 * ON AN "AS IS" BASIS, AND THE UNIVERSITY OF CALIFORNIA HAS NO OBLIGATION TO
 * PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
 */

#ifndef lint
static char rcsid[] = "$Header$ SPRITE (Berkeley)";
#endif /* not lint */

#include "tk.h"
#include "itcl.h"
#include <config.h>

Tcl_Interp* the_interp;

extern int OpenGLCmd _ANSI_ARGS_((ClientData clientData,
	Tcl_Interp *interp, int argc, char **argv));
extern int BevelCmd _ANSI_ARGS_((ClientData clientData,
	Tcl_Interp *interp, int argc, char **argv));
extern int Tk_RangeCmd _ANSI_ARGS_((ClientData clientData,
	Tcl_Interp *interp, int argc, char **argv));
extern int BLineInit _ANSI_ARGS_((void));
extern int Blt_Init _ANSI_ARGS_((Tcl_Interp* interp));
extern int Table_Init _ANSI_ARGS_((Tcl_Interp* interp));


static void (*wait_func)(void*);
static void* wait_func_data;

int tkMain(argc, argv, nwait_func, nwait_func_data)
    int argc;				/* Number of arguments. */
    char **argv;			/* Array of argument strings. */
    void (*nwait_func)(void*);
    void* nwait_func_data;
{
    wait_func=nwait_func;
    wait_func_data=nwait_func_data;
    Tk_Main(argc, argv, Tcl_AppInit);
    return 0;
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

int
Tcl_AppInit(interp)
    Tcl_Interp *interp;		/* Interpreter for application. */
{
    Tk_Window main;
    Visual* visual;
    int depth;
    Colormap colormap;

    the_interp=interp;

    main = Tk_MainWindow(interp);

    /* Use a truecolor visual if one is available */
    visual = Tk_GetVisual(interp, main, "best", &depth, &colormap);
    if (visual == NULL) {
	return TCL_ERROR;
    }
    if (!Tk_SetWindowVisual(main, visual, (unsigned) depth, colormap)) {
	return TCL_ERROR;
    }

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

    if (Tcl_Init(interp) == TCL_ERROR) {
	return TCL_ERROR;
    }
    if (Tk_Init(interp) == TCL_ERROR) {
	return TCL_ERROR;
    }
    /*
     *  Add [incr Tcl] facilities...
     */
    if (Itcl_Init(interp) == TCL_ERROR) {
	return TCL_ERROR;
    }

    /*
     * Add BLT....
     */
    if (Blt_Init(interp) == TCL_ERROR) {
	return TCL_ERROR;
    }

    /*
     * Add the table extensions
     */
    if(Table_Init(interp) == TCL_ERROR) {
       return TCL_ERROR ;
     }

    /*
     * Call Tcl_CreateCommand for application-specific commands, if
     * they weren't already created by the init procedures called above.
     */
#ifdef SCI_OPENGL
    Tcl_CreateCommand(interp, "opengl", OpenGLCmd, (ClientData) main,
		      (void (*)()) NULL);
#endif
    Tcl_CreateCommand(interp, "bevel", BevelCmd, (ClientData) main,
		      (void (*)()) NULL);
    Tcl_CreateCommand(interp, "range", Tk_RangeCmd, (ClientData) main,
                      (void (*)()) NULL);


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

    tcl_RcFileName = "~/.scirc";

    (*wait_func)(wait_func_data);
    return TCL_OK;
}
