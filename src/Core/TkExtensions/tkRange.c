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
 * tkRange.c --
 *
 *	This module implements a range widget for the Tk toolkit.
 *	A range displays a slider that can be adjusted to change a
 *	value range;  it also displays numeric labels and a textual label,
 *	if desired.
 *	
 *	The modifications to use floating-point values are based on
 *	an implementation by Paul Mackerras.  The -variable option
 *	is due to Henning Schulzrinne.  All of these are used with
 *	permission.
 *
 * Copyright (c) 1990-1994 The Regents of the University of California.
 * Copyright (c) 1994-1995 Sun Microsystems, Inc.
 *
 * See the file "license.terms" for information on usage and redistribution
 * of this file, and for a DISCLAIMER OF ALL WARRANTIES.
 *
 *
 * Originally based on the tkScale widget, and then totally modified by...
 * David Weinstein 
 * February 1995
 * January 1999
 * Copyright SCI
 *
 * Changes Log
 * ~~~~~~~ ~~~
 * 9/95  Added nonZero flag to indicate whether min and max must be distinct
 */

#include <sci_config.h>

#include "tkPort.h"
#include "default.h"
#include "tkInt.h"
#include "tclMath.h"
#include "tkRange.h"

static Tk_ConfigSpec configSpecs[] = {
    {TK_CONFIG_BORDER, "-activebackground", "activeBackground", "Foreground",
	DEF_RANGE_ACTIVE_BG_COLOR, Tk_Offset(TkRange, activeBorder),
	TK_CONFIG_COLOR_ONLY},
    {TK_CONFIG_BORDER, "-activebackground", "activeBackground", "Foreground",
	DEF_RANGE_ACTIVE_BG_MONO, Tk_Offset(TkRange, activeBorder),
	TK_CONFIG_MONO_ONLY},
    {TK_CONFIG_BORDER, "-background", "background", "Background",
	DEF_RANGE_BG_COLOR, Tk_Offset(TkRange, bgBorder),
	TK_CONFIG_COLOR_ONLY},
    {TK_CONFIG_BORDER, "-background", "background", "Background",
	DEF_RANGE_BG_MONO, Tk_Offset(TkRange, bgBorder),
	TK_CONFIG_MONO_ONLY},
    {TK_CONFIG_DOUBLE, "-bigincrement", "bigIncrement", "BigIncrement",
	DEF_RANGE_BIG_INCREMENT, Tk_Offset(TkRange, bigIncrement), 0},
    {TK_CONFIG_SYNONYM, "-bd", "borderWidth", (char *) NULL,
	(char *) NULL, 0, 0},
    {TK_CONFIG_SYNONYM, "-bg", "background", (char *) NULL,
	(char *) NULL, 0, 0},
    {TK_CONFIG_PIXELS, "-borderwidth", "borderWidth", "BorderWidth",
	DEF_RANGE_BORDER_WIDTH, Tk_Offset(TkRange, borderWidth), 0},
    {TK_CONFIG_STRING, "-command", "command", "Command",
	DEF_RANGE_COMMAND, Tk_Offset(TkRange, command), TK_CONFIG_NULL_OK},
    {TK_CONFIG_ACTIVE_CURSOR, "-cursor", "cursor", "Cursor",
	DEF_RANGE_CURSOR, Tk_Offset(TkRange, cursor), TK_CONFIG_NULL_OK},
    {TK_CONFIG_INT, "-digits", "digits", "Digits",
	DEF_RANGE_DIGITS, Tk_Offset(TkRange, digits), 0},
    {TK_CONFIG_SYNONYM, "-fg", "foreground", (char *) NULL,
	(char *) NULL, 0, 0},
    {TK_CONFIG_FONT, "-font", "font", "Font",
	DEF_RANGE_FONT, Tk_Offset(TkRange, tkfont),
	0},
    {TK_CONFIG_COLOR, "-foreground", "foreground", "Foreground",
	DEF_RANGE_FG_COLOR, Tk_Offset(TkRange, textColorPtr),
	TK_CONFIG_COLOR_ONLY},
    {TK_CONFIG_COLOR, "-foreground", "foreground", "Foreground",
	DEF_RANGE_FG_MONO, Tk_Offset(TkRange, textColorPtr),
	TK_CONFIG_MONO_ONLY},
    {TK_CONFIG_DOUBLE, "-from", "from", "From",
	DEF_RANGE_FROM, Tk_Offset(TkRange, fromValue), 0},
    {TK_CONFIG_COLOR, "-highlightbackground", "highlightBackground",
	"HighlightBackground", DEF_RANGE_HIGHLIGHT_BG,
	Tk_Offset(TkRange, highlightBgColorPtr), 0},
    {TK_CONFIG_COLOR, "-highlightcolor", "highlightColor", "HighlightColor",
	DEF_RANGE_HIGHLIGHT, Tk_Offset(TkRange, highlightColorPtr), 0},
    {TK_CONFIG_PIXELS, "-highlightthickness", "highlightThickness",
	"HighlightThickness",
	DEF_RANGE_HIGHLIGHT_WIDTH, Tk_Offset(TkRange, highlightWidth), 0},
    {TK_CONFIG_STRING, "-label", "label", "Label",
	DEF_RANGE_LABEL, Tk_Offset(TkRange, label), TK_CONFIG_NULL_OK},
    {TK_CONFIG_PIXELS, "-length", "length", "Length",
	DEF_RANGE_LENGTH, Tk_Offset(TkRange, length), 0},
    {TK_CONFIG_UID, "-orient", "orient", "Orient",
	DEF_RANGE_ORIENT, Tk_Offset(TkRange, orientUid), 0},
    {TK_CONFIG_BOOLEAN, "-nonzero", "nonZero", "NonZero",
        DEF_RANGE_NON_ZERO, Tk_Offset(TkRange, nonZero), 0},
    {TK_CONFIG_RELIEF, "-relief", "relief", "Relief",
	DEF_RANGE_RELIEF, Tk_Offset(TkRange, relief), 0},
    {TK_CONFIG_INT, "-repeatdelay", "repeatDelay", "RepeatDelay",
	DEF_RANGE_REPEAT_DELAY, Tk_Offset(TkRange, repeatDelay), 0},
    {TK_CONFIG_INT, "-repeatinterval", "repeatInterval", "RepeatInterval",
	DEF_RANGE_REPEAT_INTERVAL, Tk_Offset(TkRange, repeatInterval), 0},
    {TK_CONFIG_DOUBLE, "-resolution", "resolution", "Resolution",
	DEF_RANGE_RESOLUTION, Tk_Offset(TkRange, resolution), 0},
    {TK_CONFIG_BOOLEAN, "-showvalue", "showValue", "ShowValue",
	DEF_RANGE_SHOW_VALUE, Tk_Offset(TkRange, showValue), 0},
    {TK_CONFIG_PIXELS, "-sliderlength", "sliderLength", "SliderLength",
	DEF_RANGE_SLIDER_LENGTH, Tk_Offset(TkRange, sliderLength), 0},
    {TK_CONFIG_RELIEF, "-sliderrelief", "sliderRelief", "SliderRelief",
	DEF_RANGE_SLIDER_RELIEF, Tk_Offset(TkRange, sliderRelief),
	TK_CONFIG_DONT_SET_DEFAULT},
    {TK_CONFIG_UID, "-state", "state", "State",
	DEF_RANGE_STATE, Tk_Offset(TkRange, state), 0},
    {TK_CONFIG_STRING, "-takefocus", "takeFocus", "TakeFocus",
	DEF_RANGE_TAKE_FOCUS, Tk_Offset(TkRange, takeFocus),
	TK_CONFIG_NULL_OK},
    {TK_CONFIG_DOUBLE, "-tickinterval", "tickInterval", "TickInterval",
	DEF_RANGE_TICK_INTERVAL, Tk_Offset(TkRange, tickInterval), 0},
    {TK_CONFIG_DOUBLE, "-to", "to", "To",
	DEF_RANGE_TO, Tk_Offset(TkRange, toValue), 0},
    {TK_CONFIG_COLOR, "-troughcolor", "troughColor", "Background",
	DEF_RANGE_TROUGH_COLOR, Tk_Offset(TkRange, troughColorPtr),
	TK_CONFIG_COLOR_ONLY},
    {TK_CONFIG_COLOR, "-troughcolor", "troughColor", "Background",
	DEF_RANGE_TROUGH_MONO, Tk_Offset(TkRange, troughColorPtr),
	TK_CONFIG_MONO_ONLY},
    {TK_CONFIG_COLOR, "-rangecolor", "rangeColor", "RangeColor",
	DEF_RANGE_RANGE_COLOR, Tk_Offset(TkRange, rangeColorPtr),
	TK_CONFIG_COLOR_ONLY},
    {TK_CONFIG_COLOR, "-rangecolor", "rangeColor", "RangeColor",
	DEF_RANGE_TROUGH_MONO, Tk_Offset(TkRange, rangeColorPtr),
	TK_CONFIG_MONO_ONLY},
    {TK_CONFIG_STRING, "-var_min", "var_min", "Var_min",
	DEF_RANGE_VARIABLE, Tk_Offset(TkRange, min_varName), TK_CONFIG_NULL_OK},
    {TK_CONFIG_STRING, "-var_max", "var_max", "Var_max",
	DEF_RANGE_VARIABLE, Tk_Offset(TkRange, max_varName), TK_CONFIG_NULL_OK},
    {TK_CONFIG_PIXELS, "-width", "width", "Width",
	DEF_RANGE_WIDTH, Tk_Offset(TkRange, width), 0},
    {TK_CONFIG_END, (char *) NULL, (char *) NULL, (char *) NULL,
	(char *) NULL, 0, 0}
};

/*
 * Forward declarations for procedures defined later in this file:
 */

static void		ComputeFormat _ANSI_ARGS_((TkRange *rangePtr));
static void		ComputeRangeGeometry _ANSI_ARGS_((TkRange *rangePtr));
static int		ConfigureRange _ANSI_ARGS_((Tcl_Interp *interp,
			    TkRange *rangePtr, int argc, char **argv,
			    int flags));
static void		DestroyRange _ANSI_ARGS_((char *memPtr));
static void		RangeEventProc _ANSI_ARGS_((ClientData clientData,
			    XEvent *eventPtr));
static char *		RangeVarMinProc _ANSI_ARGS_((ClientData clientData,
			    Tcl_Interp *interp, char *name1, char *name2,
			    int flags));
static char *		RangeVarMaxProc _ANSI_ARGS_((ClientData clientData,
			    Tcl_Interp *interp, char *name1, char *name2,
			    int flags));
static int		RangeWidgetCmd _ANSI_ARGS_((ClientData clientData,
			    Tcl_Interp *interp, int argc, char **argv));
static void		RangeWorldChanged _ANSI_ARGS_((
			    ClientData instanceData));
/*
 * The structure below defines scale class behavior by means of procedures
 * that can be invoked from generic window code.
 */

static TkClassProcs rangeClass = {
    NULL,			/* createProc. */
    RangeWorldChanged,		/* geometryProc. */
    NULL			/* modalProc. */
};


/*
 *--------------------------------------------------------------
 *
 * Tk_RangeCmd --
 *
 *	This procedure is invoked to process the "range" Tcl
 *	command.  See the user documentation for details on what
 *	it does.
 *
 * Results:
 *	A standard Tcl result.
 *
 * Side effects:
 *	See the user documentation.
 *
 *--------------------------------------------------------------
 */

int
Tk_RangeCmd(clientData, interp, argc, argv)
    ClientData clientData;		/* Main window associated with
				 * interpreter. */
    Tcl_Interp *interp;		/* Current interpreter. */
    int argc;			/* Number of arguments. */
    char **argv;		/* Argument strings. */
{
    Tk_Window tkwin = (Tk_Window) clientData;
    register TkRange *rangePtr;
    Tk_Window new;

    if (argc < 2) {
	Tcl_AppendResult(interp, "wrong # args: should be \"",
		argv[0], " pathName ?options?\"", (char *) NULL);
	return TCL_ERROR;
    }

    new = Tk_CreateWindowFromPath(interp, tkwin, argv[1], (char *) NULL);
    if (new == NULL) {
	return TCL_ERROR;
    }
    rangePtr = TkpCreateRange(new);

    /*
     * Initialize fields that won't be initialized by ConfigureRange,
     * or which ConfigureRange expects to have reasonable values
     * (e.g. resource pointers).
     */

    rangePtr = (TkRange *) ckalloc(sizeof(TkRange));
    rangePtr->tkwin = new;
    rangePtr->display = Tk_Display(new);
    rangePtr->interp = interp;
    rangePtr->orientUid = NULL;
    rangePtr->vertical = 0;
    rangePtr->width = 0;
    rangePtr->length = 0;
    rangePtr->min_value = 0;
    rangePtr->max_value = 0;
    rangePtr->min_varName = NULL;
    rangePtr->max_varName = NULL;
    rangePtr->fromValue = 0;
    rangePtr->toValue = 0;
    rangePtr->tickInterval = 0;
    rangePtr->resolution = 1;
    rangePtr->bigIncrement = 0.0;
    rangePtr->command = NULL;
    rangePtr->repeatDelay = 0;
    rangePtr->repeatInterval = 0;
    rangePtr->label = NULL;
    rangePtr->labelLength = 0;
    //    rangePtr->state = tkNormalUid;
    rangePtr->borderWidth = 0;
    rangePtr->bgBorder = NULL;
    rangePtr->activeBorder = NULL;
    rangePtr->troughColorPtr = NULL;
    rangePtr->rangeColorPtr = NULL;
    rangePtr->troughGC = None;
    rangePtr->rangeGC = None;
    rangePtr->copyGC = None;
    rangePtr->tkfont = NULL;
    rangePtr->textColorPtr = NULL;
    rangePtr->textGC = None;
    rangePtr->relief = TK_RELIEF_FLAT;
    rangePtr->highlightWidth = 0;
    rangePtr->highlightColorPtr = NULL;
    rangePtr->highlightGC = None;
    rangePtr->inset = 0;
    rangePtr->sliderLength = 0;
    rangePtr->showValue = 0;
    rangePtr->nonZero = 0;
    rangePtr->horizLabelY = 0;
    rangePtr->horizValueY = 0;
    rangePtr->horizTroughY = 0;
    rangePtr->horizTickY = 0;
    rangePtr->vertTickRightX = 0;
    rangePtr->vertValueRightX = 0;
    rangePtr->vertTroughX = 0;
    rangePtr->vertLabelX = 0;
    rangePtr->cursor = None;
    rangePtr->flags = NEVER_SET;

    Tk_SetClass(rangePtr->tkwin, "Range");
    TkSetClassProcs(rangePtr->tkwin, &rangeClass, (ClientData) rangePtr);
    Tk_CreateEventHandler(rangePtr->tkwin,
	    ExposureMask|StructureNotifyMask|FocusChangeMask,
	    RangeEventProc, (ClientData) rangePtr);
    if (ConfigureRange(interp, rangePtr, argc-2, argv+2, 0) != TCL_OK) {
	goto error;
    }

    interp->result = Tk_PathName(rangePtr->tkwin);
    return TCL_OK;

    error:
    Tk_DestroyWindow(rangePtr->tkwin);
    return TCL_ERROR;
}

/*
 *--------------------------------------------------------------
 *
 * RangeWidgetCmd --
 *
 *	This procedure is invoked to process the Tcl command
 *	that corresponds to a widget managed by this module.
 *	See the user documentation for details on what it does.
 *
 * Results:
 *	A standard Tcl result.
 *
 * Side effects:
 *	See the user documentation.
 *
 *--------------------------------------------------------------
 */

static int
RangeWidgetCmd(clientData, interp, argc, argv)
    ClientData clientData;		/* Information about range
					 * widget. */
    Tcl_Interp *interp;			/* Current interpreter. */
    int argc;				/* Number of arguments. */
    char **argv;			/* Argument strings. */
{
    register TkRange *rangePtr = (TkRange *) clientData;
    int result = TCL_OK;
    size_t length;
    int c;

    if (argc < 2) {
	Tcl_AppendResult(interp, "wrong # args: should be \"",
		argv[0], " option ?arg arg ...?\"", (char *) NULL);
	return TCL_ERROR;
    }
    Tcl_Preserve((ClientData) rangePtr);
    c = argv[1][0];
    length = strlen(argv[1]);
    if ((c == 'c') && (strncmp(argv[1], "cget", length) == 0)
	    && (length >= 2)) {
	if (argc != 3) {
	    Tcl_AppendResult(interp, "wrong # args: should be \"",
		    argv[0], " cget option\"",
		    (char *) NULL);
	    goto error;
	}
	result = Tk_ConfigureValue(interp, rangePtr->tkwin, configSpecs,
		(char *) rangePtr, argv[2], 0);
    } else if ((c == 'c') && (strncmp(argv[1], "configure", length) == 0)
	    && (length >= 3)) {
	if (argc == 2) {
	    result = Tk_ConfigureInfo(interp, rangePtr->tkwin, configSpecs,
		    (char *) rangePtr, (char *) NULL, 0);
	} else if (argc == 3) {
	    result = Tk_ConfigureInfo(interp, rangePtr->tkwin, configSpecs,
		    (char *) rangePtr, argv[2], 0);
	} else {
	    result = ConfigureRange(interp, rangePtr, argc-2, argv+2,
		    TK_CONFIG_ARGV_ONLY);
	}
    } else if ((c == 'c') && (strncmp(argv[1], "coordsMin", length) == 0)
	    && (length >= 3)) {
	int x, y ;
	double value;

	if ((argc != 2) && (argc != 3)) {
	    Tcl_AppendResult(interp, "wrong # args: should be \"",
		    argv[0], " coordsMin ?value?\"", (char *) NULL);
	    goto error;
	}
	if (argc == 3) {
	    if (Tcl_GetDouble(interp, argv[2], &value) != TCL_OK) {
		goto error;
	    }
	} else {
	    value = rangePtr->min_value;
	}
	if (rangePtr->vertical) {
	    x = rangePtr->vertTroughX + rangePtr->width/2
		    + rangePtr->borderWidth;
	    y = TkpRangeValueToPixel(rangePtr, value);
	} else {
	    x = TkpRangeValueToPixel(rangePtr, value);
	    y = rangePtr->horizTroughY + rangePtr->width/2
		    + rangePtr->borderWidth;
	}
	sprintf(interp->result, "%d %d", x, y);
    } else if ((c == 'c') && (strncmp(argv[1], "coordsMax", length) == 0)
	    && (length >= 3)) {
	int x, y ;
	double value;

	if ((argc != 2) && (argc != 3)) {
	    Tcl_AppendResult(interp, "wrong # args: should be \"",
		    argv[0], " coordsMax ?value?\"", (char *) NULL);
	    goto error;
	}
	if (argc == 3) {
	    if (Tcl_GetDouble(interp, argv[2], &value) != TCL_OK) {
		goto error;
	    }
	} else {
	    value = rangePtr->max_value;
	}
	if (rangePtr->vertical) {
	    x = rangePtr->vertTroughX + rangePtr->width/2
		    + rangePtr->borderWidth;
	    y = TkpRangeValueToPixel(rangePtr, value);
	} else {
	    x = TkpRangeValueToPixel(rangePtr, value);
	    y = rangePtr->horizTroughY + rangePtr->width/2
		    + rangePtr->borderWidth;
	}
	sprintf(interp->result, "%d %d", x, y);
    } else if ((c == 'f') && (strncmp(argv[1], "from", length) == 0)) {
	double value;

	if (argc != 2) {
	    Tcl_AppendResult(interp, "wrong # args: should be \"",
		    argv[0], " from\"", (char *) NULL);
	    goto error;
	}
	value = rangePtr->fromValue;
	sprintf(interp->result, rangePtr->format, value);
    } else if ((c == 'g') && (strncmp(argv[1], "get", length) == 0)) {
	double value;
	int x, y;

	if (argc != 4) {
	    Tcl_AppendResult(interp, "wrong # args: should be \"",
		    argv[0], " get x y\"", (char *) NULL);
	    goto error;
	}
	if ((Tcl_GetInt(interp, argv[2], &x) != TCL_OK)
	    || (Tcl_GetInt(interp, argv[3], &y) != TCL_OK)) {
	    return TCL_ERROR;
	}
	value = TkRangePixelToValue(rangePtr, x, y);
	sprintf(interp->result, rangePtr->format, value);
    } else if ((c == 'g') && (strncmp(argv[1], "getMin", length) == 0)) {
	double value;

	if (argc != 2) {
	    Tcl_AppendResult(interp, "wrong # args: should be \"",
		    argv[0], " getMin\"", (char *) NULL);
	    goto error;
	}
	value = rangePtr->min_value;
	sprintf(interp->result, rangePtr->format, value);
    } else if ((c == 'g') && (strncmp(argv[1], "getMax", length) == 0)) {
	double value;

	if (argc != 2) {
	    Tcl_AppendResult(interp, "wrong # args: should be \"",
		    argv[0], " getMax\"", (char *) NULL);
	    goto error;
	}
	value = rangePtr->max_value;
	sprintf(interp->result, rangePtr->format, value);
    } else if ((c == 'i') && (strncmp(argv[1], "identify", length) == 0)) {
	int x, y, thing;

	if (argc != 4) {
	    Tcl_AppendResult(interp, "wrong # args: should be \"",
		    argv[0], " identify x y\"", (char *) NULL);
	    goto error;
	}
	if ((Tcl_GetInt(interp, argv[2], &x) != TCL_OK)
		|| (Tcl_GetInt(interp, argv[3], &y) != TCL_OK)) {
	    return TCL_ERROR;
	}
	thing = TkpRangeElement(rangePtr, x,y);
	switch (thing) {
	    case TROUGH1:	interp->result = "trough1";	break;
	    case RANGE:		interp->result = "range";	break;
	    case TROUGH2:	interp->result = "trough2";	break;
	    case MIN_SLIDER:    interp->result = "min_slider";  break;
	    case MAX_SLIDER:    interp->result = "max_slider";  break;	    
	 }
    } else if ((c == 's') && (strncmp(argv[1], "setMin", length) == 0)) {
	double value;

	if (argc != 3) {
	    Tcl_AppendResult(interp, "wrong # args: should be \"",
		    argv[0], " setMin value\"", (char *) NULL);
	    goto error;
	}
	if (Tcl_GetDouble(interp, argv[2], &value) != TCL_OK) {
	    goto error;
	}
	if (rangePtr->state != tkDisabledUid) {
	    TkpSetRangeMinValue(rangePtr, value, 1, 1);
	}
    } else if ((c == 's') && (strncmp(argv[1], "setMax", length) == 0)) {
	double value;

	if (argc != 3) {
	    Tcl_AppendResult(interp, "wrong # args: should be \"",
		    argv[0], " setMax value\"", (char *) NULL);
	    goto error;
	}
	if (Tcl_GetDouble(interp, argv[2], &value) != TCL_OK) {
	    goto error;
	}
	if (rangePtr->state != tkDisabledUid) {
	    TkpSetRangeMaxValue(rangePtr, value, 1, 1);
	}
    } else if ((c == 's') && (strncmp(argv[1], "setMinMax", length) == 0)) {
	double val1;
	double val2;

	if (argc != 4) {
	    Tcl_AppendResult(interp, "wrong # args: should be \"",
		    argv[0], " setMinMax valMin valMax\"", (char *) NULL);
	    goto error;
	}
	if (Tcl_GetDouble(interp, argv[2], &val1) != TCL_OK) {
	    goto error;
	}
	if (Tcl_GetDouble(interp, argv[3], &val2) != TCL_OK) {
	    goto error;
	}
	if (rangePtr->state != tkDisabledUid) {
	    TkpSetRangeMinValue(rangePtr, val1, 1, 1);
	    TkpSetRangeMaxValue(rangePtr, val2, 1, 1);
	}
    } else if ((c == 's') && (strncmp(argv[1], "sliderLength", length) == 0)) {
	sprintf(interp->result, "%d", rangePtr->sliderLength/2);
    } else if ((c == 't') && (strncmp(argv[1], "to", length) == 0)) {
	double value;

	if (argc != 2) {
	    Tcl_AppendResult(interp, "wrong # args: should be \"",
		    argv[0], " to\"", (char *) NULL);
	    goto error;
	}
	value = rangePtr->toValue;
	sprintf(interp->result, rangePtr->format, value);
    } else {
	Tcl_AppendResult(interp, "bad option \"", argv[1],
		"\": must be cget, configure, coords, get, identify, or set",
		(char *) NULL);
	goto error;
    }
    Tk_Release((ClientData) rangePtr);
    return result;

    error:
    Tk_Release((ClientData) rangePtr);
    return TCL_ERROR;
}

/*
 *----------------------------------------------------------------------
 *
 * DestroyRange --
 *
 *	This procedure is invoked by Tk_EventuallyFree or Tk_Release
 *	to clean up the internal structure of a button at a safe time
 *	(when no-one is using it anymore).
 *
 * Results:
 *	None.
 *
 * Side effects:
 *	Everything associated with the range is freed up.
 *
 *----------------------------------------------------------------------
 */

static void
DestroyRange(memPtr)
    char *memPtr;	/* Info about range widget. */
{
    register TkRange *rangePtr = (TkRange *) memPtr;

    /*
     * Free up all the stuff that requires special handling, then
     * let Tk_FreeOptions handle all the standard option-related
     * stuff.
     */

    if (rangePtr->min_varName != NULL) {
	Tcl_UntraceVar(rangePtr->interp, rangePtr->min_varName,
		TCL_GLOBAL_ONLY|TCL_TRACE_WRITES|TCL_TRACE_UNSETS,
		RangeVarMinProc, (ClientData) rangePtr);
    }
    if (rangePtr->max_varName != NULL) {
	Tcl_UntraceVar(rangePtr->interp, rangePtr->max_varName,
		TCL_GLOBAL_ONLY|TCL_TRACE_WRITES|TCL_TRACE_UNSETS,
		RangeVarMaxProc, (ClientData) rangePtr);
    }
    if (rangePtr->troughGC != None) {
	Tk_FreeGC(rangePtr->display, rangePtr->troughGC);
    }
    if (rangePtr->rangeGC != None) {
	Tk_FreeGC(rangePtr->display, rangePtr->rangeGC);
    }
    if (rangePtr->copyGC != None) {
	Tk_FreeGC(rangePtr->display, rangePtr->copyGC);
    }
    if (rangePtr->textGC != None) {
	Tk_FreeGC(rangePtr->display, rangePtr->textGC);
    }
    if (rangePtr->highlightGC != None) {
	Tk_FreeGC(rangePtr->display, rangePtr->highlightGC);
    }
    Tk_FreeOptions(configSpecs, (char *) rangePtr, rangePtr->display, 0);
    ckfree((char *) rangePtr);
}

/*
 *----------------------------------------------------------------------
 *
 * ConfigureRange --
 *
 *	This procedure is called to process an argv/argc list, plus
 *	the Tk option database, in order to configure (or
 *	reconfigure) a range widget.
 *
 * Results:
 *	The return value is a standard Tcl result.  If TCL_ERROR is
 *	returned, then interp->result contains an error message.
 *
 * Side effects:
 *	Configuration information, such as colors, border width,
 *	etc. get set for rangePtr;  old resources get freed,
 *	if there were any.
 *
 *----------------------------------------------------------------------
 */

static int
ConfigureRange(interp, rangePtr, argc, argv, flags)
    Tcl_Interp *interp;		/* Used for error reporting. */
    register TkRange *rangePtr;	/* Information about widget;  may or may
				 * not already have values for some fields. */
    int argc;			/* Number of valid entries in argv. */
    char **argv;		/* Arguments. */
    int flags;			/* Flags to pass to Tk_ConfigureWidget. */
{
    XGCValues gcValues;
    GC newGC;
    size_t length;

    /*
     * Eliminate any existing trace on a variable monitored by the range.
     */

    if (rangePtr->min_varName != NULL) {
	Tcl_UntraceVar(interp, rangePtr->min_varName, 
		TCL_GLOBAL_ONLY|TCL_TRACE_WRITES|TCL_TRACE_UNSETS,
		RangeVarMinProc, (ClientData) rangePtr);
    }
    if (rangePtr->max_varName != NULL) {
	Tcl_UntraceVar(interp, rangePtr->max_varName, 
		TCL_GLOBAL_ONLY|TCL_TRACE_WRITES|TCL_TRACE_UNSETS,
		RangeVarMaxProc, (ClientData) rangePtr);
    }

    if (Tk_ConfigureWidget(interp, rangePtr->tkwin, configSpecs,
	    argc, argv, (char *) rangePtr, flags) != TCL_OK) {
	return TCL_ERROR;
    }

    /*
     * If the range is tied to the value of a variable, then set up
     * a trace on the variable's value and set the range's value from
     * the value of the variable, if it exists.
     */

    if (rangePtr->min_varName != NULL) {
	char *stringValue, *end;
	double value;

	stringValue = Tcl_GetVar(interp, rangePtr->min_varName, TCL_GLOBAL_ONLY);
	if (stringValue != NULL) {
	    value = strtod(stringValue, &end);
	    if ((end != stringValue) && (*end == 0)) {
		rangePtr->min_value = value;
	    }
	}
	Tcl_TraceVar(interp, rangePtr->min_varName,
		TCL_GLOBAL_ONLY|TCL_TRACE_WRITES|TCL_TRACE_UNSETS,
		RangeVarMinProc, (ClientData) rangePtr);
    }
    if (rangePtr->max_varName != NULL) {
	char *stringValue, *end;
	double value;

	stringValue = Tcl_GetVar(interp, rangePtr->max_varName, TCL_GLOBAL_ONLY);
	if (stringValue != NULL) {
	    value = strtod(stringValue, &end);
	    if ((end != stringValue) && (*end == 0)) {
		rangePtr->max_value = value;
	    }
	}
	Tcl_TraceVar(interp, rangePtr->max_varName,
		TCL_GLOBAL_ONLY|TCL_TRACE_WRITES|TCL_TRACE_UNSETS,
		RangeVarMaxProc, (ClientData) rangePtr);
    }

    /*
     * Several options need special processing, such as parsing the
     * orientation and creating GCs.
     */

    length = strlen(rangePtr->orientUid);
    if (strncmp(rangePtr->orientUid, "vertical", length) == 0) {
	rangePtr->vertical = 1;
    } else if (strncmp(rangePtr->orientUid, "horizontal", length) == 0) {
	rangePtr->vertical = 0;
    } else {
	Tcl_AppendResult(interp, "bad orientation \"", rangePtr->orientUid,
		"\": must be vertical or horizontal", (char *) NULL);
	return TCL_ERROR;
    }

    /*
     * Make sure that the resolution is always positive and non-zero.
     */

    if (rangePtr->resolution <= 0) {
	rangePtr->resolution = 1;
    }

    rangePtr->fromValue = TkRoundToResolution(rangePtr, rangePtr->fromValue);
    rangePtr->toValue = TkRoundToResolution(rangePtr, rangePtr->toValue);
    if (rangePtr->toValue == rangePtr->fromValue)
	rangePtr->toValue+=rangePtr->resolution;
    rangePtr->tickInterval = TkRoundToResolution(rangePtr,
	    rangePtr->tickInterval);

    /*
     * Make sure that the tick interval has the right sign so that
     * addition moves from fromValue to toValue.
     */

    if ((rangePtr->tickInterval < 0)
	    ^ ((rangePtr->toValue - rangePtr->fromValue) <  0)) {
	rangePtr->tickInterval = -rangePtr->tickInterval;
    }

    /*
     * Set the range value to itself;  all this does is to make sure
     * that the range's value is within the new acceptable range for
     * the range and reflect the value in the associated variable,
     * if any.
     */

    ComputeFormat(rangePtr);
    TkpSetRangeMinValue(rangePtr, rangePtr->min_value, 1, 1);
    TkpSetRangeMaxValue(rangePtr, rangePtr->max_value, 1, 1);

    if (rangePtr->label != NULL) {
	rangePtr->labelLength = strlen(rangePtr->label);
    } else {
	rangePtr->labelLength = 0;
    }

/*     if ((rangePtr->state != tkNormalUid) */
/* 	    && (rangePtr->state != tkDisabledUid) */
/* 	    && (rangePtr->state != tkActiveUid)) { */
/* 	Tcl_AppendResult(interp, "bad state value \"", rangePtr->state, */
/* 		"\":  must be normal, active, or disabled", (char *) NULL); */
/* 	rangePtr->state = tkNormalUid; */
/* 	return TCL_ERROR; */
/*     } */

    Tk_SetBackgroundFromBorder(rangePtr->tkwin, rangePtr->bgBorder);

    if (rangePtr->highlightWidth < 0) {
	rangePtr->highlightWidth = 0;
    }
    rangePtr->inset = rangePtr->highlightWidth + rangePtr->borderWidth;
    RangeWorldChanged((ClientData) rangePtr);
    return TCL_OK;
}

/*
 *---------------------------------------------------------------------------
 *
 * RangeWorldChanged --
 *
 *      This procedure is called when the world has changed in some
 *      way and the widget needs to recompute all its graphics contexts
 *	and determine its new geometry.
 *
 * Results:
 *      None.
 *
 * Side effects:
 *      Range will be relayed out and redisplayed.
 *
 *---------------------------------------------------------------------------
 */
 
static void
RangeWorldChanged(instanceData)
    ClientData instanceData;	/* Information about widget. */
{
    XGCValues gcValues;
    GC gc;
    TkRange *rangePtr;

    rangePtr = (TkRange *) instanceData;

    gcValues.foreground = rangePtr->troughColorPtr->pixel;
    gc = Tk_GetGC(rangePtr->tkwin, GCForeground, &gcValues);
    if (rangePtr->troughGC != None) {
	Tk_FreeGC(rangePtr->display, rangePtr->troughGC);
    }
    rangePtr->troughGC = gc;

    gcValues.font = Tk_FontId(rangePtr->tkfont);
    gcValues.foreground = rangePtr->textColorPtr->pixel;
    gc = Tk_GetGC(rangePtr->tkwin, GCForeground | GCFont, &gcValues);
    if (rangePtr->textGC != None) {
	Tk_FreeGC(rangePtr->display, rangePtr->textGC);
    }
    rangePtr->textGC = gc;

    if (rangePtr->copyGC == None) {
	gcValues.graphics_exposures = False;
	rangePtr->copyGC = Tk_GetGC(rangePtr->tkwin, GCGraphicsExposures,
	    &gcValues);
    }
    rangePtr->inset = rangePtr->highlightWidth + rangePtr->borderWidth;

    /*
     * Recompute display-related information, and let the geometry
     * manager know how much space is needed now.
     */

    ComputeRangeGeometry(rangePtr);

    TkEventuallyRedrawRange(rangePtr, REDRAW_ALL);
}

/*
 *----------------------------------------------------------------------
 *
 * ComputeFormat --
 *
 *	This procedure is invoked to recompute the "format" field
 *	of a range's widget record, which determines how the value
 *	of the range is converted to a string.
 *
 * Results:
 *	None.
 *
 * Side effects:
 *	The format field of rangePtr is modified.
 *
 *----------------------------------------------------------------------
 */

static void
ComputeFormat(rangePtr)
    TkRange *rangePtr;			/* Information about range widget. */
{
    double maxValue, x;
    int mostSigDigit, numDigits, leastSigDigit, afterDecimal;
    int eDigits, fDigits;

    /*
     * Compute the displacement from the decimal of the most significant
     * digit required for any number in the range's range.
     */

    maxValue = fabs(rangePtr->fromValue);
    x = fabs(rangePtr->toValue);
    if (x > maxValue) {
	maxValue = x;
	if (maxValue == 0) {
	    maxValue = 1;
	}
    }
    mostSigDigit = floor(log10(maxValue));

    /*
     * If the number of significant digits wasn't specified explicitly,
     * compute it (it's the difference between the most significant
     * digit overall and the most significant digit of the resolution).
     */

    numDigits = rangePtr->digits;
    if (numDigits <= 0) {
	if  (rangePtr->resolution > 0) {
	    /*
	     * A resolution was specified for the range, so just use it.
	     */

	    leastSigDigit = (int) floor(log10(rangePtr->resolution));
	} else {
	    /*
	     * No resolution was specified, so compute the difference
	     * in value between adjacent pixels and use it for the least
	     * significant digit.
	     */

	    x = fabs(rangePtr->fromValue - rangePtr->toValue);
	    if (rangePtr->length > 0) {
		x /= rangePtr->length;
	    }
	    if (x > 0){
		leastSigDigit = (int) floor(log10(x));
	    } else {
		leastSigDigit = 0;
	    }
	}
	numDigits = mostSigDigit - leastSigDigit + 1;
	if (numDigits < 1) {
	    numDigits = 1;
	}
    }

    /*
     * Compute the number of characters required using "e" format and
     * "f" format, and then choose whichever one takes fewer characters.
     */

    eDigits = numDigits + 4;
    if (numDigits > 1) {
	eDigits++;			/* Decimal point. */
    }
    afterDecimal = numDigits - mostSigDigit - 1;
    if (afterDecimal < 0) {
	afterDecimal = 0;
    }
    fDigits = (mostSigDigit >= 0) ? mostSigDigit + afterDecimal : afterDecimal;
    if (afterDecimal > 0) {
	fDigits++;			/* Decimal point. */
    }
    if (mostSigDigit < 0) {
	fDigits++;			/* Zero to left of decimal point. */
    }
    if (fDigits <= eDigits) {
	sprintf(rangePtr->format, "%%.%df", afterDecimal);
    } else {
	sprintf(rangePtr->format, "%%.%de", numDigits-1);
    }
}

/*
 *----------------------------------------------------------------------
 *
 * ComputeRangeGeometry --
 *
 *	This procedure is called to compute various geometrical
 *	information for a range, such as where various things get
 *	displayed.  It's called when the window is reconfigured.
 *
 * Results:
 *	None.
 *
 * Side effects:
 *	Display-related numbers get changed in *rangePtr.  The
 *	geometry manager gets told about the window's preferred size.
 *
 *----------------------------------------------------------------------
 */

static void
ComputeRangeGeometry(rangePtr)
    register TkRange *rangePtr;		/* Information about widget. */
{
    char valueString[PRINT_CHARS];
    int tmp, valuePixels, x, y, extraSpace;
    Tk_FontMetrics fm;

    /*
     * Horizontal ranges are simpler than vertical ones because
     * all sizes are the same (the height of a line of text);
     * handle them first and then quit.
     */

    Tk_GetFontMetrics(rangePtr->tkfont, &fm);
    if (!rangePtr->vertical) {
	y = rangePtr->inset;
	extraSpace = 0;
	if (rangePtr->labelLength != 0) {
	    rangePtr->horizLabelY = y + SPACING;
	    y += fm.linespace + SPACING;
	    extraSpace = SPACING;
	}
	if (rangePtr->showValue) {
	    rangePtr->horizValueY = y + SPACING;
	    y += fm.linespace + SPACING;
	    extraSpace = SPACING;
	} else {
	    rangePtr->horizValueY = y;
	}
	y += extraSpace;
	rangePtr->horizTroughY = y;
	y += rangePtr->width + 2*rangePtr->borderWidth;
	if (rangePtr->tickInterval != 0) {
	    rangePtr->horizTickY = y + SPACING;
	    y += fm.linespace + 2*SPACING;
	}
	Tk_GeometryRequest(rangePtr->tkwin,
		rangePtr->length + 2*rangePtr->inset, y + rangePtr->inset);
	Tk_SetInternalBorder(rangePtr->tkwin, rangePtr->inset);
	return;
    }

    /*
     * Vertical range:  compute the amount of space needed to display
     * the ranges value by formatting strings for the two end points;
     * use whichever length is longer.
     */

    sprintf(valueString, rangePtr->format, rangePtr->fromValue);
    valuePixels = Tk_TextWidth(rangePtr->tkfont, valueString, -1);

    sprintf(valueString, rangePtr->format, rangePtr->toValue);
    tmp = Tk_TextWidth(rangePtr->tkfont, valueString, -1);
    if (valuePixels < tmp) {
	valuePixels = tmp;
    }

    /*
     * Assign x-locations to the elements of the range, working from
     * left to right.
     */

    x = rangePtr->inset;
    if ((rangePtr->tickInterval != 0) && (rangePtr->showValue)) {
	rangePtr->vertTickRightX = x + SPACING + valuePixels;
	rangePtr->vertValueRightX = rangePtr->vertTickRightX + valuePixels
		+ fm.ascent/2;
	x = rangePtr->vertValueRightX + SPACING;
    } else if (rangePtr->tickInterval != 0) {
	rangePtr->vertTickRightX = x + SPACING + valuePixels;
	rangePtr->vertValueRightX = rangePtr->vertTickRightX;
	x = rangePtr->vertTickRightX + SPACING;
    } else if (rangePtr->showValue) {
	rangePtr->vertTickRightX = x;
	rangePtr->vertValueRightX = x + SPACING + valuePixels;
	x = rangePtr->vertValueRightX + SPACING;
    } else {
	rangePtr->vertTickRightX = x;
	rangePtr->vertValueRightX = x;
    }
    rangePtr->vertTroughX = x;
    x += 2*rangePtr->borderWidth + rangePtr->width;
    if (rangePtr->labelLength == 0) {
	rangePtr->vertLabelX = 0;
    } else {
	rangePtr->vertLabelX = x + fm.ascent/2;
	x = rangePtr->vertLabelX + fm.ascent/2
		+ Tk_TextWidth(rangePtr->tkfont, rangePtr->label,
			rangePtr->labelLength);
    }
    Tk_GeometryRequest(rangePtr->tkwin, x + rangePtr->inset,
	    rangePtr->length + 2*rangePtr->inset);
    Tk_SetInternalBorder(rangePtr->tkwin, rangePtr->inset);
}

/*
 *--------------------------------------------------------------
 *
 * RangeEventProc --
 *
 *	This procedure is invoked by the Tk dispatcher for various
 *	events on ranges.
 *
 * Results:
 *	None.
 *
 * Side effects:
 *	When the window gets deleted, internal structures get
 *	cleaned up.  When it gets exposed, it is redisplayed.
 *
 *--------------------------------------------------------------
 */

static void
RangeEventProc(clientData, eventPtr)
    ClientData clientData;	/* Information about window. */
    XEvent *eventPtr;		/* Information about event. */
{
    TkRange *rangePtr = (TkRange *) clientData;

    if ((eventPtr->type == Expose) && (eventPtr->xexpose.count == 0)) {
	TkEventuallyRedrawRange(rangePtr, REDRAW_ALL);
    } else if (eventPtr->type == DestroyNotify) {
	if (rangePtr->tkwin != NULL) {
	    rangePtr->tkwin = NULL;
	    Tcl_DeleteCommandFromToken(rangePtr->interp, rangePtr->widgetCmd);
	}
	if (rangePtr->flags & REDRAW_ALL) {
	    Tcl_CancelIdleCall(TkpDisplayRange, (ClientData) rangePtr);
	}
	Tcl_EventuallyFree((ClientData) rangePtr, DestroyRange);
    } else if (eventPtr->type == ConfigureNotify) {
	ComputeRangeGeometry(rangePtr);
	TkEventuallyRedrawRange(rangePtr, REDRAW_ALL);
    } else if (eventPtr->type == FocusIn) {
	if (eventPtr->xfocus.detail != NotifyInferior) {
	    rangePtr->flags |= GOT_FOCUS;
	    if (rangePtr->highlightWidth > 0) {
		TkEventuallyRedrawRange(rangePtr, REDRAW_ALL);
	    }
	}
    } else if (eventPtr->type == FocusOut) {
	if (eventPtr->xfocus.detail != NotifyInferior) {
	    rangePtr->flags &= ~GOT_FOCUS;
	    if (rangePtr->highlightWidth > 0) {
		TkEventuallyRedrawRange(rangePtr, REDRAW_ALL);
	    }
	}
    }
}

/*
 *----------------------------------------------------------------------
 *
 * RangeCmdDeletedProc --
 *
 *	This procedure is invoked when a widget command is deleted.  If
 *	the widget isn't already in the process of being destroyed,
 *	this command destroys it.
 *
 * Results:
 *	None.
 *
 * Side effects:
 *	The widget is destroyed.
 *
 *----------------------------------------------------------------------
 */

static void
RangeCmdDeletedProc(clientData)
    ClientData clientData;	/* Pointer to widget record for widget. */
{
    TkRange *rangePtr = (TkRange *) clientData;
    Tk_Window tkwin = rangePtr->tkwin;

    /*
     * This procedure could be invoked either because the window was
     * destroyed and the command was then deleted (in which case tkwin
     * is NULL) or because the command was deleted, and then this procedure
     * destroys the widget.
     */

    if (tkwin != NULL) {
	rangePtr->tkwin = NULL;
	Tk_DestroyWindow(tkwin);
    }
}

/*
 *--------------------------------------------------------------
 *
 * TkEventuallyRedrawRange --
 *
 *	Arrange for part or all of a range widget to redrawn at
 *	the next convenient time in the future.
 *
 * Results:
 *	None.
 *
 * Side effects:
 *	If "what" is REDRAW_SLIDER then just the slider and the
 *	value readout will be redrawn;  if "what" is REDRAW_ALL
 *	then the entire widget will be redrawn.
 *
 *--------------------------------------------------------------
 */

void
TkEventuallyRedrawRange(rangePtr, what)
    register TkRange *rangePtr;	/* Information about widget. */
    int what;			/* What to redraw:  REDRAW_SLIDER
				 * or REDRAW_ALL. */
{
    if ((what == 0) || (rangePtr->tkwin == NULL)
	    || !Tk_IsMapped(rangePtr->tkwin)) {
	return;
    }
    if ((rangePtr->flags & REDRAW_ALL) == 0) {
	Tk_DoWhenIdle(TkpDisplayRange, (ClientData) rangePtr);
    }
    rangePtr->flags |= what;
}

/*
 *----------------------------------------------------------------------
 *
 * RangeVarMinProc --
 *
 *	This procedure is invoked by Tcl whenever someone modifies a
 *	variable associated with a range widget.
 *
 * Results:
 *	NULL is always returned.
 *
 * Side effects:
 *	The value displayed in the range will change to match the
 *	variable's new value.  If the variable has a bogus value then
 *	it is reset to the value of the range.
 *
 *----------------------------------------------------------------------
 */

    /* ARGSUSED */
static char *
RangeVarMinProc(clientData, interp, name1, name2, flags)
    ClientData clientData;	/* Information about button. */
    Tcl_Interp *interp;		/* Interpreter containing variable. */
    char *name1;		/* Name of variable. */
    char *name2;		/* Second part of variable name. */
    int flags;			/* Information about what happened. */
{
    register TkRange *rangePtr = (TkRange *) clientData;
    char *stringValue, *end, *result;
    double value;

    /*
     * If the variable is unset, then immediately recreate it unless
     * the whole interpreter is going away.
     */

    if (flags & TCL_TRACE_UNSETS) {
	if ((flags & TCL_TRACE_DESTROYED) && !(flags & TCL_INTERP_DESTROYED)) {
	    Tcl_TraceVar2(interp, name1, name2,
		    TCL_GLOBAL_ONLY|TCL_TRACE_WRITES|TCL_TRACE_UNSETS,
		    RangeVarMinProc, clientData);
	    rangePtr->flags |= NEVER_SET;
	    TkpSetRangeMinValue(rangePtr, rangePtr->min_value, 1, 0);
	}
	return (char *) NULL;
    }

    /*
     * If we came here because we updated the variable (in SetRangeMinValue),
     * then ignore the trace.  Otherwise update the range with the value
     * of the variable.
     */

    if (rangePtr->flags & SETTING_VAR) {
	return (char *) NULL;
    }
    result = NULL;
    stringValue = Tcl_GetVar(interp, rangePtr->min_varName, TCL_GLOBAL_ONLY);
    if (stringValue != NULL) {
	value = strtod(stringValue, &end);
	if ((end == stringValue) || (*end != 0)) {
	    result = "can't assign non-numeric value to range variable";
	} else {
	    rangePtr->min_value = TkRoundToResolution(rangePtr, value);
	}

	/*
	 * This code is a bit tricky because it sets the range's value before
	 * calling SetRangeMinValue.  This way, SetRangeMinValue won't bother to
	 * set the variable again or to invoke the -command.  However, it
	 * also won't redisplay the range, so we have to ask for that
	 * explicitly.
	 */

	TkpSetRangeMinValue(rangePtr, rangePtr->min_value, 1, 0);
	TkEventuallyRedrawRange(rangePtr, REDRAW_SLIDER);
    }
    return result;
}
/*
 *----------------------------------------------------------------------
 *
 * RangeVarMaxProc --
 *
 *	This procedure is invoked by Tcl whenever someone modifies a
 *	variable associated with a range widget.
 *
 * Results:
 *	NULL is always returned.
 *
 * Side effects:
 *	The value displayed in the range will change to match the
 *	variable's new value.  If the variable has a bogus value then
 *	it is reset to the value of the range.
 *
 *----------------------------------------------------------------------
 */

    /* ARGSUSED */
static char *
RangeVarMaxProc(clientData, interp, name1, name2, flags)
    ClientData clientData;	/* Information about button. */
    Tcl_Interp *interp;		/* Interpreter containing variable. */
    char *name1;		/* Name of variable. */
    char *name2;		/* Second part of variable name. */
    int flags;			/* Information about what happened. */
{
    register TkRange *rangePtr = (TkRange *) clientData;
    char *stringValue, *end, *result;
    double value;

    /*
     * If the variable is unset, then immediately recreate it unless
     * the whole interpreter is going away.
     */

    if (flags & TCL_TRACE_UNSETS) {
	if ((flags & TCL_TRACE_DESTROYED) && !(flags & TCL_INTERP_DESTROYED)) {
	    Tcl_TraceVar2(interp, name1, name2,
		    TCL_GLOBAL_ONLY|TCL_TRACE_WRITES|TCL_TRACE_UNSETS,
		    RangeVarMaxProc, clientData);
	    rangePtr->flags |= NEVER_SET;
	    TkpSetRangeMaxValue(rangePtr, rangePtr->max_value, 1, 0);
	}
	return (char *) NULL;
    }

    /*
     * If we came here because we updated the variable (in SetRangeMaxValue),
     * then ignore the trace.  Otherwise update the range with the value
     * of the variable.
     */

    if (rangePtr->flags & SETTING_VAR) {
	return (char *) NULL;
    }
    result = NULL;
    stringValue = Tcl_GetVar(interp, rangePtr->max_varName, TCL_GLOBAL_ONLY);
    if (stringValue != NULL) {
	value = strtod(stringValue, &end);
	if ((end == stringValue) || (*end != 0)) {
	    result = "can't assign non-numeric value to range variable";
	} else {
	    rangePtr->max_value = TkRoundToResolution(rangePtr, value);
	}

	/*
	 * This code is a bit tricky because it sets the range's value before
	 * calling SetRangeMaxValue.  This way, SetRangeMaxValue won't bother to
	 * set the variable again or to invoke the -command.  However, it
	 * also won't redisplay the range, so we have to ask for that
	 * explicitly.
	 */

	TkpSetRangeMaxValue(rangePtr, rangePtr->max_value, 1, 0);
	TkEventuallyRedrawRange(rangePtr, REDRAW_SLIDER);
    }

    return result;
}
