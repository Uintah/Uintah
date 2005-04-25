/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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

#include "tkPort.h"
#include "default.h"
#include "tkInt.h"
#include "tclMath.h"
#include "tkRange.h"


#include <sci_defs/config_defs.h> /* for HAVE_LIMITS etc, for tcl files */


/*
 * The following table defines the legal values for the -orient option.
 * It is used together with the "enum orient" declaration in tkRange.h.
 */

static char *orientStrings[] = {
    "horizontal", "vertical", (char *) NULL
};

/*
 * The following table defines the legal values for the -state option.
 * It is used together with the "enum state" declaration in tkRange.h.
 */

static char *stateStrings[] = {
    "active", "disabled", "normal", (char *) NULL
};

static Tk_OptionSpec optionSpecs[] = {
    {TK_OPTION_BORDER, "-activebackground", "activeBackground", "Foreground",
	DEF_RANGE_ACTIVE_BG_COLOR, -1, Tk_Offset(TkRange, activeBorder),
	0, (ClientData) DEF_RANGE_ACTIVE_BG_MONO, 0},
    {TK_OPTION_BORDER, "-background", "background", "Background",
	DEF_RANGE_BG_COLOR, -1, Tk_Offset(TkRange, bgBorder),
	0, (ClientData) DEF_RANGE_BG_MONO, 0},
    {TK_OPTION_DOUBLE, "-bigincrement", "bigIncrement", "BigIncrement",
        DEF_RANGE_BIG_INCREMENT, -1, Tk_Offset(TkRange, bigIncrement), 
        0, 0, 0},
    {TK_OPTION_SYNONYM, "-bd", (char *) NULL, (char *) NULL,
	(char *) NULL, 0, -1, 0, (ClientData) "-borderwidth", 0},
    {TK_OPTION_SYNONYM, "-bg", (char *) NULL, (char *) NULL,
	(char *) NULL, 0, -1, 0, (ClientData) "-background", 0},
    {TK_OPTION_PIXELS, "-borderwidth", "borderWidth", "BorderWidth",
	DEF_RANGE_BORDER_WIDTH, -1, Tk_Offset(TkRange, borderWidth), 
        0, 0, 0},
    {TK_OPTION_STRING, "-command", "command", "Command",
	DEF_RANGE_COMMAND, -1, Tk_Offset(TkRange, command),
	TK_OPTION_NULL_OK, 0, 0},
    {TK_OPTION_CURSOR, "-cursor", "cursor", "Cursor",
	DEF_RANGE_CURSOR, -1, Tk_Offset(TkRange, cursor),
	TK_OPTION_NULL_OK, 0, 0},
    {TK_OPTION_INT, "-digits", "digits", "Digits", 
	DEF_RANGE_DIGITS, -1, Tk_Offset(TkRange, digits), 
        0, 0, 0},
    {TK_OPTION_SYNONYM, "-fg", "foreground", (char *) NULL,
	(char *) NULL, 0, -1, 0, (ClientData) "-foreground", 0},
    {TK_OPTION_FONT, "-font", "font", "Font",
	DEF_RANGE_FONT, -1, Tk_Offset(TkRange, tkfont), 0, 0, 0},
    {TK_OPTION_COLOR, "-foreground", "foreground", "Foreground",
	DEF_RANGE_FG_COLOR, -1, Tk_Offset(TkRange, textColorPtr), 0, 
        (ClientData) DEF_RANGE_FG_MONO, 0},
    {TK_OPTION_COLOR, "-rangecolor", "rangeColor", "Foreground",
	DEF_RANGE_RANGE_COLOR, -1, Tk_Offset(TkRange, rangeColorPtr), 0, 
        (ClientData) DEF_RANGE_RANGE_MONO, 0},
    {TK_OPTION_DOUBLE, "-from", "from", "From", DEF_RANGE_FROM, -1, 
        Tk_Offset(TkRange, fromValue), 0, 0, 0},
    {TK_OPTION_BORDER, "-highlightbackground", "highlightBackground",
	"HighlightBackground", DEF_RANGE_HIGHLIGHT_BG_COLOR,
	-1, Tk_Offset(TkRange, highlightBorder), 
        0, (ClientData) DEF_RANGE_HIGHLIGHT_BG_MONO, 0},
    {TK_OPTION_COLOR, "-highlightcolor", "highlightColor", "HighlightColor",
	DEF_RANGE_HIGHLIGHT, -1, Tk_Offset(TkRange, highlightColorPtr),
	0, 0, 0},
    {TK_OPTION_PIXELS, "-highlightthickness", "highlightThickness",
	"HighlightThickness", DEF_RANGE_HIGHLIGHT_WIDTH, -1, 
	Tk_Offset(TkRange, highlightWidth), 0, 0, 0},
    {TK_OPTION_STRING, "-label", "label", "Label",
	DEF_RANGE_LABEL, -1, Tk_Offset(TkRange, label),
	TK_OPTION_NULL_OK, 0, 0},
    {TK_OPTION_PIXELS, "-length", "length", "Length",
	DEF_RANGE_LENGTH, -1, Tk_Offset(TkRange, length), 0, 0, 0},
    {TK_OPTION_STRING_TABLE, "-orient", "orient", "Orient",
        DEF_RANGE_ORIENT, -1, Tk_Offset(TkRange, orient), 
        0, (ClientData) orientStrings, 0},
    {TK_OPTION_BOOLEAN, "-nonzero", "nonZero", "NonZero",
        DEF_RANGE_NON_ZERO, -1, Tk_Offset(TkRange, nonZero),
        0, 0, 0},
    {TK_OPTION_RELIEF, "-relief", "relief", "Relief",
	DEF_RANGE_RELIEF, -1, Tk_Offset(TkRange, relief), 0, 0, 0},
    {TK_OPTION_INT, "-repeatdelay", "repeatDelay", "RepeatDelay",
        DEF_RANGE_REPEAT_DELAY, -1, Tk_Offset(TkRange, repeatDelay),
        0, 0, 0},
    {TK_OPTION_INT, "-repeatinterval", "repeatInterval", "RepeatInterval",
        DEF_RANGE_REPEAT_INTERVAL, -1, Tk_Offset(TkRange, repeatInterval),
        0, 0, 0},
    {TK_OPTION_DOUBLE, "-resolution", "resolution", "Resolution",
        DEF_RANGE_RESOLUTION, -1, Tk_Offset(TkRange, resolution),
        0, 0, 0},
    {TK_OPTION_BOOLEAN, "-showvalue", "showValue", "ShowValue",
        DEF_RANGE_SHOW_VALUE, -1, Tk_Offset(TkRange, showValue),
        0, 0, 0},
    {TK_OPTION_PIXELS, "-sliderlength", "sliderLength", "SliderLength",
        DEF_RANGE_SLIDER_LENGTH, -1, Tk_Offset(TkRange, sliderLength),
        0, 0, 0},
    {TK_OPTION_RELIEF, "-sliderrelief", "sliderRelief", "SliderRelief",
	DEF_RANGE_SLIDER_RELIEF, -1, Tk_Offset(TkRange, sliderRelief), 
        0, 0, 0},
    {TK_OPTION_STRING_TABLE, "-state", "state", "State",
        DEF_RANGE_STATE, -1, Tk_Offset(TkRange, state), 
        0, (ClientData) stateStrings, 0},
    {TK_OPTION_STRING, "-takefocus", "takeFocus", "TakeFocus",
	DEF_RANGE_TAKE_FOCUS, Tk_Offset(TkRange, takeFocusPtr), -1,
	TK_OPTION_NULL_OK, 0, 0},
    {TK_OPTION_DOUBLE, "-tickinterval", "tickInterval", "TickInterval",
        DEF_RANGE_TICK_INTERVAL, -1, Tk_Offset(TkRange, tickInterval),
        0, 0, 0},
    {TK_OPTION_DOUBLE, "-to", "to", "To",
        DEF_RANGE_TO, -1, Tk_Offset(TkRange, toValue), 0, 0, 0},
    {TK_OPTION_COLOR, "-troughcolor", "troughColor", "Background",
        DEF_RANGE_TROUGH_COLOR, -1, Tk_Offset(TkRange, troughColorPtr),
        0, (ClientData) DEF_RANGE_TROUGH_MONO, 0},
    {TK_OPTION_STRING, "-varmin", "varMin", "VarMin",
	DEF_RANGE_VARIABLE, Tk_Offset(TkRange, minVarNamePtr), -1,
	TK_OPTION_NULL_OK, 0, 0},
    {TK_OPTION_STRING, "-varmax", "varMax", "VarMax",
	DEF_RANGE_VARIABLE, Tk_Offset(TkRange, maxVarNamePtr), -1,
	TK_OPTION_NULL_OK, 0, 0},
    {TK_OPTION_PIXELS, "-width", "width", "Width",
	DEF_RANGE_WIDTH, -1, Tk_Offset(TkRange, width), 0, 0, 0},
    {TK_OPTION_END, (char *) NULL, (char *) NULL, (char *) NULL,
	(char *) NULL, 0, -1, 0, 0, 0}
};

/*
 * The following tables define the range widget commands and map the 
 * indexes into the string tables into a single enumerated type used 
 * to dispatch the range widget command.
 */

static char *commandNames[] = {
  "cget", "configure", "coordsMin", "coordsMax", "from", "to", "sliderLength",
  "get", "getMin", "getMax", "identify", "setMin", "setMax", "setMinMax", (char *) NULL
};

enum command {
    COMMAND_CGET, COMMAND_CONFIGURE, COMMAND_COORDSMIN, COMMAND_COORDSMAX, 
    COMMAND_FROM, COMMAND_TO, COMMAND_SLIDERLENGTH,
    COMMAND_GET, COMMAND_GETMIN, COMMAND_GETMAX,
    COMMAND_IDENTIFY, COMMAND_SETMIN, COMMAND_SETMAX, COMMAND_SETMINMAX
};

/*
 * Forward declarations for procedures defined later in this file:
 */

static void		ComputeFormat _ANSI_ARGS_((TkRange *rangePtr));
static void		ComputeRangeGeometry _ANSI_ARGS_((TkRange *rangePtr));
static int		ConfigureRange _ANSI_ARGS_((Tcl_Interp *interp,
			    TkRange *rangePtr, int objc,
			    Tcl_Obj *CONST objv[]));
static void		DestroyRange _ANSI_ARGS_((char *memPtr));
static void		RangeCmdDeletedProc _ANSI_ARGS_((
			    ClientData clientData));
static void		RangeEventProc _ANSI_ARGS_((ClientData clientData,
			    XEvent *eventPtr));
static char *		RangeVarMinProc _ANSI_ARGS_((ClientData clientData,
			    Tcl_Interp *interp, char *name1, char *name2,
			    int flags));
static char *		RangeVarMaxProc _ANSI_ARGS_((ClientData clientData,
			    Tcl_Interp *interp, char *name1, char *name2,
			    int flags));
static int		RangeWidgetObjCmd _ANSI_ARGS_((ClientData clientData,
			    Tcl_Interp *interp, int objc, 
			    Tcl_Obj *CONST objv[]));
static void		RangeWorldChanged _ANSI_ARGS_((
			    ClientData instanceData));
static void		RangeSetMinVariable _ANSI_ARGS_((TkRange *rangePtr));
static void		RangeSetMaxVariable _ANSI_ARGS_((TkRange *rangePtr));

/*
 * The structure below defines range class behavior by means of procedures
 * that can be invoked from generic window code.
 */

#if (TCL_MINOR_VERSION >= 4)
static Tk_ClassProcs rangeClass = {
#else 
static TkClassProcs rangeClass = {
#endif
    NULL,			/* createProc. */
    RangeWorldChanged,		/* geometryProc. */
    NULL			/* modalProc. */
};


/*
 *--------------------------------------------------------------
 *
 * Tk_RangeObjCmd --
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
Tk_RangeObjCmd(clientData, interp, objc, objv)
    ClientData clientData;	/* Either NULL or pointer to option table. */
    Tcl_Interp *interp;		/* Current interpreter. */
    int objc;			/* Number of arguments. */
    Tcl_Obj *CONST objv[];	/* Argument values. */
{
    register TkRange *rangePtr;
    Tk_OptionTable optionTable;
    Tk_Window tkwin;

    optionTable = (Tk_OptionTable) clientData;
    if (optionTable == NULL) {
	Tcl_CmdInfo info;
	char *name;

	/*
	 * We haven't created the option table for this widget class
	 * yet.  Do it now and save the table as the clientData for
	 * the command, so we'll have access to it in future
	 * invocations of the command.
	 */

	optionTable = Tk_CreateOptionTable(interp, optionSpecs);
	name = Tcl_GetString(objv[0]);
	Tcl_GetCommandInfo(interp, name, &info);
	info.objClientData = (ClientData) optionTable;
	Tcl_SetCommandInfo(interp, name, &info);
	}

    if (objc < 2) {
	Tcl_WrongNumArgs(interp, 1, objv, "pathName ?options?");
	return TCL_ERROR;
    }

    tkwin = Tk_CreateWindowFromPath(interp, Tk_MainWindow(interp),
				    Tcl_GetString(objv[1]), (char *) NULL);

    if (tkwin == NULL) {
	return TCL_ERROR;
    }

    Tk_SetClass(tkwin, "Range");
    rangePtr = TkpCreateRange(tkwin);

    /*
     * Initialize fields that won't be initialized by ConfigureRange,
     * or which ConfigureRange expects to have reasonable values
     * (e.g. resource pointers).
     */

    rangePtr->tkwin		= tkwin;
    rangePtr->display		= Tk_Display(tkwin);
    rangePtr->interp		= interp;
    rangePtr->widgetCmd		= Tcl_CreateObjCommand(interp,
	    Tk_PathName(rangePtr->tkwin), RangeWidgetObjCmd,
	    (ClientData) rangePtr, RangeCmdDeletedProc);
    rangePtr->optionTable	= optionTable;
    rangePtr->orient		= ORIENT_VERTICAL;
    rangePtr->width		= 0;
    rangePtr->length		= 0;
    rangePtr->minvalue		= 0.0;
    rangePtr->maxvalue		= 0.0;
    rangePtr->minVarNamePtr	= NULL;
    rangePtr->maxVarNamePtr	= NULL;
    rangePtr->fromValue		= 0.0;
    rangePtr->toValue		= 0.0;
    rangePtr->tickInterval	= 0.0;
    rangePtr->resolution	= 1.0;
    rangePtr->digits		= 0;
    rangePtr->bigIncrement	= 0.0;
    rangePtr->command		= NULL;
    rangePtr->repeatDelay	= 0;
    rangePtr->repeatInterval	= 0;
    rangePtr->label		= NULL;
    rangePtr->labelLength	= 0;
    rangePtr->state		= STATE_NORMAL;
    rangePtr->borderWidth	= 0;
    rangePtr->bgBorder		= NULL;
    rangePtr->activeBorder	= NULL;
    rangePtr->sliderRelief	= TK_RELIEF_RAISED;
    rangePtr->troughColorPtr	= NULL;
    rangePtr->troughGC		= None;
    rangePtr->copyGC		= None;
    rangePtr->tkfont		= NULL;
    rangePtr->textColorPtr	= NULL;
    rangePtr->rangeColorPtr	= NULL;
    rangePtr->textGC		= None;
    rangePtr->rangeGC		= None;
    rangePtr->relief		= TK_RELIEF_FLAT;
    rangePtr->highlightWidth	= 0;
    rangePtr->highlightBorder	= NULL;
    rangePtr->highlightColorPtr	= NULL;
    rangePtr->inset		= 0;
    rangePtr->sliderLength	= 0;
    rangePtr->showValue		= 0;
    rangePtr->nonZero           = 0;
    rangePtr->horizLabelY	= 0;
    rangePtr->horizValueY	= 0;
    rangePtr->horizTroughY	= 0;
    rangePtr->horizTickY	= 0;
    rangePtr->vertTickRightX	= 0;
    rangePtr->vertValueRightX	= 0;
    rangePtr->vertTroughX	= 0;
    rangePtr->vertLabelX	= 0;
    rangePtr->fontHeight	= 0;
    rangePtr->cursor		= None;
    rangePtr->takeFocusPtr	= NULL;
    rangePtr->flags		= NEVER_SET;

#if (TCL_MINOR_VERSION >= 4)
    Tk_SetClassProcs(rangePtr->tkwin, &rangeClass, (ClientData) rangePtr);
#else
    TkSetClassProcs(rangePtr->tkwin, &rangeClass, (ClientData) rangePtr);
#endif
    Tk_CreateEventHandler(rangePtr->tkwin,
	    ExposureMask|StructureNotifyMask|FocusChangeMask,
	    RangeEventProc, (ClientData) rangePtr);

    if ((Tk_InitOptions(interp, (char *) rangePtr, optionTable, tkwin) != TCL_OK) ||
	(ConfigureRange(interp, rangePtr, objc - 2, objv + 2) != TCL_OK)) {
      Tk_DestroyWindow(rangePtr->tkwin);
      return TCL_ERROR;
    }

    Tcl_SetResult(interp, Tk_PathName(rangePtr->tkwin), TCL_STATIC);
    return TCL_OK;
}


/*
 *--------------------------------------------------------------
 *
 * RangeWidgetObjCmd --
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
RangeWidgetObjCmd(clientData, interp, objc, objv)
    ClientData clientData;		/* Information about range
					 * widget. */
    Tcl_Interp *interp;			/* Current interpreter. */
    int objc;				/* Number of arguments. */
    Tcl_Obj *CONST objv[];		/* Argument strings. */
{
    TkRange *rangePtr = (TkRange *) clientData;
    Tcl_Obj *objPtr;
    int index, result;

    if (objc < 2) {
        Tcl_WrongNumArgs(interp, 1, objv, "option ?arg arg ...?");
	return TCL_ERROR;
    }
    result = Tcl_GetIndexFromObj(interp, objv[1], commandNames,
            "option", 0, &index);
    if (result != TCL_OK) {
	return result;
    }
    Tcl_Preserve((ClientData) rangePtr);

    switch (index) {
        case COMMAND_CGET: {
  	    if (objc != 3) {
	        Tcl_WrongNumArgs(interp, 1, objv, "cget option");
		goto error;
	    }
	    objPtr = Tk_GetOptionValue(interp, (char *) rangePtr,
		    rangePtr->optionTable, objv[2], rangePtr->tkwin);
	    if (objPtr == NULL) {
		 goto error;
	    } else {
		Tcl_SetObjResult(interp, objPtr);
	    }
	    break;
	}
        case COMMAND_CONFIGURE: {
	    if (objc <= 3) {
		objPtr = Tk_GetOptionInfo(interp, (char *) rangePtr,
			rangePtr->optionTable,
			(objc == 3) ? objv[2] : (Tcl_Obj *) NULL,
			rangePtr->tkwin);
		if (objPtr == NULL) {
		    goto error;
		} else {
		    Tcl_SetObjResult(interp, objPtr);
		}
	    } else {
		result = ConfigureRange(interp, rangePtr, objc-2, objv+2);
	    }
	    break;
	}
        case COMMAND_COORDSMIN: {
	    int x, y ;
	    double value;
	    char buf[TCL_INTEGER_SPACE * 2];

	    if ((objc != 2) && (objc != 3)) {
	        Tcl_WrongNumArgs(interp, 1, objv, "coords ?value?");
		goto error;
	    }
	    if (objc == 3) {
	        if (Tcl_GetDoubleFromObj(interp, objv[2], &value) 
                        != TCL_OK) {
		    goto error;
		}
	    } else {
	        value = rangePtr->minvalue;
	    }
	    if (rangePtr->orient == ORIENT_VERTICAL) {
	        x = rangePtr->vertTroughX + rangePtr->width/2
		        + rangePtr->borderWidth;
		y = TkRangeValueToPixel(rangePtr, value);
	    } else {
	        x = TkRangeValueToPixel(rangePtr, value);
		y = rangePtr->horizTroughY + rangePtr->width/2
                        + rangePtr->borderWidth;
	    }
	    sprintf(buf, "%d %d", x, y);
	    Tcl_SetResult(interp, buf, TCL_VOLATILE);
            break;
        }
        case COMMAND_COORDSMAX: {
	    int x, y ;
	    double value;
	    char buf[TCL_INTEGER_SPACE * 2];

	    if ((objc != 2) && (objc != 3)) {
	        Tcl_WrongNumArgs(interp, 1, objv, "coords ?value?");
		goto error;
	    }
	    if (objc == 3) {
	        if (Tcl_GetDoubleFromObj(interp, objv[2], &value) 
                        != TCL_OK) {
		    goto error;
		}
	    } else {
	        value = rangePtr->maxvalue;
	    }
	    if (rangePtr->orient == ORIENT_VERTICAL) {
	        x = rangePtr->vertTroughX + rangePtr->width/2
		        + rangePtr->borderWidth;
		y = TkRangeValueToPixel(rangePtr, value);
	    } else {
	        x = TkRangeValueToPixel(rangePtr, value);
		y = rangePtr->horizTroughY + rangePtr->width/2
                        + rangePtr->borderWidth;
	    }
	    sprintf(buf, "%d %d", x, y);
	    Tcl_SetResult(interp, buf, TCL_VOLATILE);
            break;
        }
        case COMMAND_FROM: {
	    double value;
	    char buf[TCL_DOUBLE_SPACE];

	    if ((objc != 2)) {
	        Tcl_WrongNumArgs(interp, 1, objv, "from");
		goto error;
	    }
	    value = rangePtr->fromValue;

	    sprintf(buf, rangePtr->format, value);
	    Tcl_SetResult(interp, buf, TCL_VOLATILE);	  
	  break;
	}
        case COMMAND_TO: {
	    double value;
	    char buf[TCL_DOUBLE_SPACE];

	    if ((objc != 2)) {
	        Tcl_WrongNumArgs(interp, 1, objv, "to");
		goto error;
	    }
	    value = rangePtr->toValue;

	    sprintf(buf, rangePtr->format, value);
	    Tcl_SetResult(interp, buf, TCL_VOLATILE);	 	  
	  break;
	}
        case COMMAND_SLIDERLENGTH: {
	  char buf[TCL_DOUBLE_SPACE];
	    sprintf(buf, rangePtr->format, rangePtr->sliderLength/2);
	    Tcl_SetResult(interp, buf, TCL_VOLATILE);
	  break;
	}
        case COMMAND_GET: {
	    double value;
	    int x, y;
	    char buf[TCL_DOUBLE_SPACE];

	    if (objc != 4) {
	        Tcl_WrongNumArgs(interp, 1, objv, "get x y");
		goto error;
	    }
	    if ((Tcl_GetIntFromObj(interp, objv[2], &x) != TCL_OK)
		|| (Tcl_GetIntFromObj(interp, objv[3], &y) 
		    != TCL_OK)) {
	      goto error;
	    }
	    value = TkRangePixelToValue(rangePtr, x, y);

	    sprintf(buf, rangePtr->format, value);
	    Tcl_SetResult(interp, buf, TCL_VOLATILE);
            break;
        }
        case COMMAND_GETMIN: {
	    double value;
	    char buf[TCL_DOUBLE_SPACE];

	    if (objc != 2) {
	        Tcl_WrongNumArgs(interp, 1, objv, "getMin");
		goto error;
	    }

	    value = rangePtr->minvalue;

	    sprintf(buf, rangePtr->format, value);
	    Tcl_SetResult(interp, buf, TCL_VOLATILE);
            break;
        }
        case COMMAND_GETMAX: {
	    double value;
	    char buf[TCL_DOUBLE_SPACE];

	    if (objc != 2) {
	        Tcl_WrongNumArgs(interp, 1, objv, "getMax");
		goto error;
	    }

	    value = rangePtr->maxvalue;

	    sprintf(buf, rangePtr->format, value);
	    Tcl_SetResult(interp, buf, TCL_VOLATILE);
            break;
        }
        case COMMAND_IDENTIFY: {
	    int x, y, thing;

	    if (objc != 4) {
	        Tcl_WrongNumArgs(interp, 1, objv, "identify x y");
		goto error;
	    }
	    if ((Tcl_GetIntFromObj(interp, objv[2], &x) != TCL_OK)
                    || (Tcl_GetIntFromObj(interp, objv[3], &y) != TCL_OK)) {
	        goto error;
	    }
	    thing = TkpRangeElement(rangePtr, x,y);
	    switch (thing) {
	    case TROUGH1:
	      Tcl_SetResult(interp, "trough1", TCL_STATIC);
	      break;
	    case RANGE:
	      Tcl_SetResult(interp, "range", TCL_STATIC);
	      break;
	    case TROUGH2:
	      Tcl_SetResult(interp, "trough2", TCL_STATIC);
	      break;
	    case MIN_SLIDER:
	      Tcl_SetResult(interp, "min_slider", TCL_STATIC);
	      break;
	    case MAX_SLIDER:
	      Tcl_SetResult(interp, "max_slider", TCL_STATIC);
	      break;
	    }
            break;
        }
        case COMMAND_SETMIN: {
	    double value;

	    if (objc != 3) {
	        Tcl_WrongNumArgs(interp, 1, objv, "setMin value");
		goto error;
	    }
	    if (Tcl_GetDoubleFromObj(interp, objv[2], &value) != TCL_OK) {
	        goto error;
	    }
	    if (rangePtr->state != STATE_DISABLED) {
	      TkRangeSetMinValue(rangePtr, value, 1, 1);
	    }
	    break;
        } 
        case COMMAND_SETMAX: {
	    double value;

	    if (objc != 3) {
	        Tcl_WrongNumArgs(interp, 1, objv, "setMax value");
		goto error;
	    }
	    if (Tcl_GetDoubleFromObj(interp, objv[2], &value) != TCL_OK) {
	        goto error;
	    }
	    if (rangePtr->state != STATE_DISABLED) {
	      TkRangeSetMaxValue(rangePtr, value, 1, 1);
	    }
	    break;
        } 
        case COMMAND_SETMINMAX: {
	    double minvalue;
	    double maxvalue;

	    if (objc != 4) {
	        Tcl_WrongNumArgs(interp, 1, objv, "setMinMax min max");
		goto error;
	    }
	    if (Tcl_GetDoubleFromObj(interp, objv[2], &minvalue) != TCL_OK) {
	        goto error;
	    }
	    if (Tcl_GetDoubleFromObj(interp, objv[3], &maxvalue) != TCL_OK) {
	        goto error;
	    }
	    if (rangePtr->state != STATE_DISABLED) {
	      TkRangeSetMinValue(rangePtr, minvalue, 1, 1);
	      TkRangeSetMaxValue(rangePtr, maxvalue, 1, 1);
	    }
	    break;
        } 
    }
    Tcl_Release((ClientData) rangePtr);
    return result;

    error:
    Tcl_Release((ClientData) rangePtr);
    return TCL_ERROR;
}

/*
 *----------------------------------------------------------------------
 *
 * DestroyRange --
 *
 *	This procedure is invoked by Tcl_EventuallyFree or Tcl_Release
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

    rangePtr->flags |= RANGE_DELETED;

    Tcl_DeleteCommandFromToken(rangePtr->interp, rangePtr->widgetCmd);
    if (rangePtr->flags & REDRAW_PENDING) {
	Tcl_CancelIdleCall(TkpDisplayRange, (ClientData) rangePtr);
    }

    /*
     * Free up all the stuff that requires special handling, then
     * let Tk_FreeOptions handle all the standard option-related
     * stuff.
     */

    if (rangePtr->minVarNamePtr != NULL) {
	Tcl_UntraceVar(rangePtr->interp, Tcl_GetString(rangePtr->minVarNamePtr),
		TCL_GLOBAL_ONLY|TCL_TRACE_WRITES|TCL_TRACE_UNSETS,
		RangeVarMinProc, (ClientData) rangePtr);
    }
    if (rangePtr->maxVarNamePtr != NULL) {
	Tcl_UntraceVar(rangePtr->interp, Tcl_GetString(rangePtr->maxVarNamePtr),
		TCL_GLOBAL_ONLY|TCL_TRACE_WRITES|TCL_TRACE_UNSETS,
		RangeVarMaxProc, (ClientData) rangePtr);
    }
    if (rangePtr->troughGC != None) {
	Tk_FreeGC(rangePtr->display, rangePtr->troughGC);
    }
    if (rangePtr->copyGC != None) {
	Tk_FreeGC(rangePtr->display, rangePtr->copyGC);
    }
    if (rangePtr->textGC != None) {
	Tk_FreeGC(rangePtr->display, rangePtr->textGC);
    }
    if (rangePtr->rangeGC != None) {
	Tk_FreeGC(rangePtr->display, rangePtr->rangeGC);
    }
    Tk_FreeConfigOptions((char *) rangePtr, rangePtr->optionTable,
	    rangePtr->tkwin);
    rangePtr->tkwin = NULL;
    TkpDestroyRange(rangePtr);
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
 *	returned, then the interp's result contains an error message.
 *
 * Side effects:
 *	Configuration information, such as colors, border width,
 *	etc. get set for rangePtr;  old resources get freed,
 *	if there were any.
 *
 *----------------------------------------------------------------------
 */

static int
ConfigureRange(interp, rangePtr, objc, objv)
    Tcl_Interp *interp;		/* Used for error reporting. */
    register TkRange *rangePtr;	/* Information about widget;  may or may
				 * not already have values for some fields. */
    int objc;			/* Number of valid entries in objv. */
    Tcl_Obj *CONST objv[];	/* Argument values. */
{
    Tk_SavedOptions savedOptions;
    Tcl_Obj *errorResult = NULL;
    int error;
    double oldMinValue = rangePtr->minvalue;
    double oldMaxValue = rangePtr->maxvalue;

    /*
     * Eliminate any existing trace on a variable monitored by the range.
     */
    if (rangePtr->minVarNamePtr != NULL) {
	Tcl_UntraceVar(interp, Tcl_GetString(rangePtr->minVarNamePtr),
		TCL_GLOBAL_ONLY|TCL_TRACE_WRITES|TCL_TRACE_UNSETS,
		RangeVarMinProc, (ClientData) rangePtr);
    }

    if (rangePtr->maxVarNamePtr != NULL) {
	Tcl_UntraceVar(interp, Tcl_GetString(rangePtr->maxVarNamePtr),
		TCL_GLOBAL_ONLY|TCL_TRACE_WRITES|TCL_TRACE_UNSETS,
		RangeVarMaxProc, (ClientData) rangePtr);
    }

    for (error = 0; error <= 1; error++) {
	if (!error) {
	    /*
	     * First pass: set options to new values.
	     */
	    if (Tk_SetOptions(interp, (char *) rangePtr,
		    rangePtr->optionTable, objc, objv,
		    rangePtr->tkwin, &savedOptions, (int *) NULL) != TCL_OK) {
		continue;
	    }
	} else {
	    /*
	     * Second pass: restore options to old values.
	     */
	    errorResult = Tcl_GetObjResult(interp);
	    Tcl_IncrRefCount(errorResult);
	    Tk_RestoreSavedOptions(&savedOptions);
	}

	/*
	 * If the range is tied to the value of a variable, then set 
	 * the range's value from the value of the variable, if it exists
	 * and it holds a valid double value.
	 */
	if (rangePtr->minVarNamePtr != NULL) {
	    double value;
	    Tcl_Obj *valuePtr;

	    valuePtr = Tcl_ObjGetVar2(interp, rangePtr->minVarNamePtr, NULL,
		    TCL_GLOBAL_ONLY);
	    if ((valuePtr != NULL) &&
		    (Tcl_GetDoubleFromObj(NULL, valuePtr, &value) == TCL_OK)) {
		rangePtr->minvalue = TkRangeRoundToResolution(rangePtr, value);
	    }
	}

	if (rangePtr->maxVarNamePtr != NULL) {
	    double value;
	    Tcl_Obj *valuePtr;

	    valuePtr = Tcl_ObjGetVar2(interp, rangePtr->maxVarNamePtr, NULL,
		    TCL_GLOBAL_ONLY);
	    if ((valuePtr != NULL) &&
		    (Tcl_GetDoubleFromObj(NULL, valuePtr, &value) == TCL_OK)) {
		rangePtr->maxvalue = TkRangeRoundToResolution(rangePtr, value);
	    }
	}

	/*
	 * Several options need special processing, such as parsing the
	 * orientation and creating GCs.
	 */

	rangePtr->fromValue = TkRangeRoundToResolution(rangePtr, 
                rangePtr->fromValue);
	rangePtr->toValue = TkRangeRoundToResolution(rangePtr, rangePtr->toValue);
	rangePtr->tickInterval = TkRangeRoundToResolution(rangePtr,
	        rangePtr->tickInterval);

	/*
	 * Make sure that the tick interval has the right sign so that
	 * addition moves from fromValue to toValue.
	 */

	if ((rangePtr->tickInterval < 0)
		^ ((rangePtr->toValue - rangePtr->fromValue) <  0)) {
	  rangePtr->tickInterval = -rangePtr->tickInterval;
	}

	ComputeFormat(rangePtr);

	rangePtr->labelLength = rangePtr->label ? strlen(rangePtr->label) : 0;

	Tk_SetBackgroundFromBorder(rangePtr->tkwin, rangePtr->bgBorder);

	if (rangePtr->highlightWidth < 0) {
	    rangePtr->highlightWidth = 0;
	}
	rangePtr->inset = rangePtr->highlightWidth + rangePtr->borderWidth;
	break;
    }

    if (!error) {
        Tk_FreeSavedOptions(&savedOptions);
    }

    /*
     * Set the range value to itself;  all this does is to make sure
     * that the range's value is within the new acceptable range for
     * the range.  We don't set the var here because we need to make
     * special checks for possibly changed varNamePtr.
     */

    TkRangeSetMinValue(rangePtr, rangePtr->minvalue, 0, 1);
    TkRangeSetMaxValue(rangePtr, rangePtr->maxvalue, 0, 1);

    /*
     * Reestablish the variable trace, if it is needed.
     */
    if (rangePtr->minVarNamePtr != NULL) {
	Tcl_Obj *valuePtr;

	/*
	 * Set the associated variable only when the new value differs
	 * from the current value, or the variable doesn't yet exist
	 */
	valuePtr = Tcl_ObjGetVar2(interp, rangePtr->minVarNamePtr, NULL,
		TCL_GLOBAL_ONLY);
	if ((valuePtr == NULL) || (rangePtr->minvalue != oldMinValue)
		|| (Tcl_GetDoubleFromObj(NULL, valuePtr, &oldMinValue) != TCL_OK)
		|| (rangePtr->minvalue != oldMinValue)) {
	    RangeSetMinVariable(rangePtr);
	}
        Tcl_TraceVar(interp, Tcl_GetString(rangePtr->minVarNamePtr),
	        TCL_GLOBAL_ONLY|TCL_TRACE_WRITES|TCL_TRACE_UNSETS,
	        RangeVarMinProc, (ClientData) rangePtr);
    }

    if (rangePtr->maxVarNamePtr != NULL) {
	Tcl_Obj *valuePtr;

	/*
	 * Set the associated variable only when the new value differs
	 * from the current value, or the variable doesn't yet exist
	 */
	valuePtr = Tcl_ObjGetVar2(interp, rangePtr->maxVarNamePtr, NULL,
		TCL_GLOBAL_ONLY);
	if ((valuePtr == NULL) || (rangePtr->maxvalue != oldMaxValue)
		|| (Tcl_GetDoubleFromObj(NULL, valuePtr, &oldMaxValue) != TCL_OK)
		|| (rangePtr->maxvalue != oldMaxValue)) {
	    RangeSetMaxVariable(rangePtr);
	}
        Tcl_TraceVar(interp, Tcl_GetString(rangePtr->maxVarNamePtr),
	        TCL_GLOBAL_ONLY|TCL_TRACE_WRITES|TCL_TRACE_UNSETS,
	        RangeVarMaxProc, (ClientData) rangePtr);
    }

    RangeWorldChanged((ClientData) rangePtr);
    if (error) {
        Tcl_SetObjResult(interp, errorResult);
	Tcl_DecrRefCount(errorResult);
	return TCL_ERROR;
    } else {
	return TCL_OK;
    }
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

    gcValues.foreground = rangePtr->rangeColorPtr->pixel;
    gc = Tk_GetGC(rangePtr->tkwin, GCForeground, &gcValues);
    if (rangePtr->rangeGC != None) {
	Tk_FreeGC(rangePtr->display, rangePtr->rangeGC);
    }
    rangePtr->rangeGC = gc;

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
    }
    if (maxValue == 0) {
	maxValue = 1;
    }
    mostSigDigit = (int) floor(log10(maxValue));

    /*
     * If the number of significant digits wasn't specified explicitly,
     * compute it. It's the difference between the most significant
     * digit needed to represent any number on the range and the
     * most significant digit of the smallest difference between
     * numbers on the range.  In other words, display enough digits so
     * that at least one digit will be different between any two adjacent
     * positions of the range.
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

    Tk_GetFontMetrics(rangePtr->tkfont, &fm);
    rangePtr->fontHeight = fm.linespace + SPACING;

    /*
     * Horizontal ranges are simpler than vertical ones because
     * all sizes are the same (the height of a line of text);
     * handle them first and then quit.
     */

    if (rangePtr->orient == ORIENT_HORIZONTAL) {
	y = rangePtr->inset;
	extraSpace = 0;
	if (rangePtr->labelLength != 0) {
	    rangePtr->horizLabelY = y + SPACING;
	    y += rangePtr->fontHeight;
	    extraSpace = SPACING;
	}
	if (rangePtr->showValue) {
	    rangePtr->horizValueY = y + SPACING;
	    y += rangePtr->fontHeight;
	    extraSpace = SPACING;
	} else {
	    rangePtr->horizValueY = y;
	}
	y += extraSpace;
	rangePtr->horizTroughY = y;
	y += rangePtr->width + 2*rangePtr->borderWidth;
	if (rangePtr->tickInterval != 0) {
	    rangePtr->horizTickY = y + SPACING;
	    y += rangePtr->fontHeight + SPACING;
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
	DestroyRange((char *) clientData);
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

    if (!(rangePtr->flags & RANGE_DELETED)) {
	rangePtr->flags |= RANGE_DELETED;
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
    if (!(rangePtr->flags & REDRAW_PENDING)) {
	rangePtr->flags |= REDRAW_PENDING;
	Tcl_DoWhenIdle(TkpDisplayRange, (ClientData) rangePtr);
    }
    rangePtr->flags |= what;
}

/*
 *--------------------------------------------------------------
 *
 * TkRangeRoundToResolution --
 *
 *	Round a given floating-point value to the nearest multiple
 *	of the range's resolution.
 *
 * Results:
 *	The return value is the rounded result.
 *
 * Side effects:
 *	None.
 *
 *--------------------------------------------------------------
 */

double
TkRangeRoundToResolution(rangePtr, value)
    TkRange *rangePtr;		/* Information about range widget. */
    double value;		/* Value to round. */
{
    double rem, new, tick;

    if (rangePtr->resolution <= 0) {
	return value;
    }
    tick = floor(value/rangePtr->resolution);
    new = rangePtr->resolution * tick;
    rem = value - new;
    if (rem < 0) {
	if (rem <= -rangePtr->resolution/2) {
	    new = (tick - 1.0) * rangePtr->resolution;
	}
    } else {
	if (rem >= rangePtr->resolution/2) {
	    new = (tick + 1.0) * rangePtr->resolution;
	}
    }
    return new;
}

/*
 *----------------------------------------------------------------------
 *
 * RangeVarProc --
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
    char *resultStr;
    double value;
    Tcl_Obj *valuePtr;
    int result;

    /*
     * If the variable is unset, then immediately recreate it unless
     * the whole interpreter is going away.
     */

    if (flags & TCL_TRACE_UNSETS) {
	if ((flags & TCL_TRACE_DESTROYED) && !(flags & TCL_INTERP_DESTROYED)) {
	    Tcl_TraceVar(interp, Tcl_GetString(rangePtr->minVarNamePtr),
		    TCL_GLOBAL_ONLY|TCL_TRACE_WRITES|TCL_TRACE_UNSETS,
		    RangeVarMinProc, clientData);
	    rangePtr->flags |= NEVER_SET;
	    TkRangeSetMinValue(rangePtr, rangePtr->minvalue, 1, 0);
	}
	return (char *) NULL;
    }

    /*
     * If we came here because we updated the variable (in TkRangeSetMinValue),
     * then ignore the trace.  Otherwise update the range with the value
     * of the variable.
     */

    if (rangePtr->flags & SETTING_VAR) {
	return (char *) NULL;
    }
    resultStr = NULL;
    valuePtr = Tcl_ObjGetVar2(interp, rangePtr->minVarNamePtr, NULL, 
            TCL_GLOBAL_ONLY);
    result = Tcl_GetDoubleFromObj(interp, valuePtr, &value);
    if (result != TCL_OK) {
        resultStr = "can't assign non-numeric value to range variable";
	RangeSetMinVariable(rangePtr);
    } else {
	rangePtr->minvalue = TkRangeRoundToResolution(rangePtr, value);

	/*
	 * This code is a bit tricky because it sets the range's value before
	 * calling TkRangeSetMinValue.  This way, TkRangeSetMinValue won't bother 
	 * to set the variable again or to invoke the -command.  However, it
	 * also won't redisplay the range, so we have to ask for that
	 * explicitly.
	 */

	TkRangeSetMinValue(rangePtr, rangePtr->minvalue, 1, 0);
    }
    TkEventuallyRedrawRange(rangePtr, REDRAW_SLIDER);

    return resultStr;
}

static char *
RangeVarMaxProc(clientData, interp, name1, name2, flags)
    ClientData clientData;	/* Information about button. */
    Tcl_Interp *interp;		/* Interpreter containing variable. */
    char *name1;		/* Name of variable. */
    char *name2;		/* Second part of variable name. */
    int flags;			/* Information about what happened. */
{
    register TkRange *rangePtr = (TkRange *) clientData;
    char *resultStr;
    double value;
    Tcl_Obj *valuePtr;
    int result;

    /*
     * If the variable is unset, then immediately recreate it unless
     * the whole interpreter is going away.
     */

    if (flags & TCL_TRACE_UNSETS) {
	if ((flags & TCL_TRACE_DESTROYED) && !(flags & TCL_INTERP_DESTROYED)) {
	    Tcl_TraceVar(interp, Tcl_GetString(rangePtr->maxVarNamePtr),
		    TCL_GLOBAL_ONLY|TCL_TRACE_WRITES|TCL_TRACE_UNSETS,
		    RangeVarMaxProc, clientData);
	    rangePtr->flags |= NEVER_SET;
	    TkRangeSetMaxValue(rangePtr, rangePtr->maxvalue, 1, 0);
	}
	return (char *) NULL;
    }

    /*
     * If we came here because we updated the variable (in TkRangeSetMaxValue),
     * then ignore the trace.  Otherwise update the range with the value
     * of the variable.
     */

    if (rangePtr->flags & SETTING_VAR) {
	return (char *) NULL;
    }
    resultStr = NULL;
    valuePtr = Tcl_ObjGetVar2(interp, rangePtr->maxVarNamePtr, NULL, 
            TCL_GLOBAL_ONLY);
    result = Tcl_GetDoubleFromObj(interp, valuePtr, &value);
    if (result != TCL_OK) {
        resultStr = "can't assign non-numeric value to range variable";
	RangeSetMaxVariable(rangePtr);
    } else {
	rangePtr->maxvalue = TkRangeRoundToResolution(rangePtr, value);

	/*
	 * This code is a bit tricky because it sets the range's value before
	 * calling TkRangeSetMaxValue.  This way, TkRangeSetMaxValue won't bother 
	 * to set the variable again or to invoke the -command.  However, it
	 * also won't redisplay the range, so we have to ask for that
	 * explicitly.
	 */

	TkRangeSetMaxValue(rangePtr, rangePtr->maxvalue, 1, 0);
    }
    TkEventuallyRedrawRange(rangePtr, REDRAW_SLIDER);

    return resultStr;
}

/*
 *--------------------------------------------------------------
 *
 * TkRangeSetValue --
 *
 *	This procedure changes the value of a range and invokes
 *	a Tcl command to reflect the current position of a range
 *
 * Results:
 *	None.
 *
 * Side effects:
 *	A Tcl command is invoked, and an additional error-processing
 *	command may also be invoked.  The range's slider is redrawn.
 *
 *--------------------------------------------------------------
 */

void
TkRangeSetMinValue(rangePtr, value, setVar, invokeCommand)
    register TkRange *rangePtr;	/* Info about widget. */
    double value;		/* New value for range.  Gets adjusted
				 * if it's off the range. */
    int setVar;			/* Non-zero means reflect new value through
				 * to associated variable, if any. */
    int invokeCommand;		/* Non-zero means invoked -command option
				 * to notify of new value, 0 means don't. */
{
    value = TkRangeRoundToResolution(rangePtr, value);
    if ((value < rangePtr->fromValue)
	    ^ (rangePtr->toValue < rangePtr->fromValue)) {
	value = rangePtr->fromValue;
    }
    if ((value > rangePtr->toValue)
	    ^ (rangePtr->toValue < rangePtr->fromValue)) {
	value = rangePtr->toValue;
    }
    if (rangePtr->flags & NEVER_SET) {
	rangePtr->flags &= ~NEVER_SET;
    } else if (rangePtr->minvalue == value) {
	return;
    }
    rangePtr->minvalue = value;
    if (invokeCommand) {
	rangePtr->flags |= INVOKE_COMMAND;
    }
    TkEventuallyRedrawRange(rangePtr, REDRAW_SLIDER);

    if (setVar && rangePtr->minVarNamePtr) {
	RangeSetMinVariable(rangePtr);
    }
}

void
TkRangeSetMaxValue(rangePtr, value, setVar, invokeCommand)
    register TkRange *rangePtr;	/* Info about widget. */
    double value;		/* New value for range.  Gets adjusted
				 * if it's off the range. */
    int setVar;			/* Non-zero means reflect new value through
				 * to associated variable, if any. */
    int invokeCommand;		/* Non-zero means invoked -command option
				 * to notify of new value, 0 means don't. */
{
    value = TkRangeRoundToResolution(rangePtr, value);
    if ((value < rangePtr->fromValue)
	    ^ (rangePtr->toValue < rangePtr->fromValue)) {
	value = rangePtr->fromValue;
    }
    if ((value > rangePtr->toValue)
	    ^ (rangePtr->toValue < rangePtr->fromValue)) {
	value = rangePtr->toValue;
    }
    if (rangePtr->flags & NEVER_SET) {
	rangePtr->flags &= ~NEVER_SET;
    } else if (rangePtr->maxvalue == value) {
	return;
    }
    rangePtr->maxvalue = value;
    if (invokeCommand) {
	rangePtr->flags |= INVOKE_COMMAND;
    }
    TkEventuallyRedrawRange(rangePtr, REDRAW_SLIDER);

    if (setVar && rangePtr->maxVarNamePtr) {
	RangeSetMaxVariable(rangePtr);
    }
}

/*
 *--------------------------------------------------------------
 *
 * RangeSetVariable --
 *
 *	This procedure sets the variable associated with a range, if any.
 *
 * Results:
 *	None.
 *
 * Side effects:
 *	Other write traces on the variable will trigger.
 *
 *--------------------------------------------------------------
 */


static void
RangeSetMinVariable(rangePtr)
    register TkRange *rangePtr;	/* Info about widget. */
{
    if (rangePtr->minVarNamePtr != NULL) {
	char string[PRINT_CHARS];
	sprintf(string, rangePtr->format, rangePtr->minvalue);
	rangePtr->flags |= SETTING_VAR;
	Tcl_ObjSetVar2(rangePtr->interp, rangePtr->minVarNamePtr, NULL,
		Tcl_NewStringObj(string, -1), TCL_GLOBAL_ONLY);
	rangePtr->flags &= ~SETTING_VAR;
    }
}

static void
RangeSetMaxVariable(rangePtr)
    register TkRange *rangePtr;	/* Info about widget. */
{
    if (rangePtr->maxVarNamePtr != NULL) {
	char string[PRINT_CHARS];
	sprintf(string, rangePtr->format, rangePtr->maxvalue);
	rangePtr->flags |= SETTING_VAR;
	Tcl_ObjSetVar2(rangePtr->interp, rangePtr->maxVarNamePtr, NULL,
		Tcl_NewStringObj(string, -1), TCL_GLOBAL_ONLY);
	rangePtr->flags &= ~SETTING_VAR;
    }
}

/*
 *----------------------------------------------------------------------
 *
 * TkRangePixelToValue --
 *
 *	Given a pixel within a range window, return the range
 *	reading corresponding to that pixel.
 *
 * Results:
 *	A double-precision range reading.  If the value is outside
 *	the legal range for the range then it's rounded to the nearest
 *	end of the range.
 *
 * Side effects:
 *	None.
 *
 *----------------------------------------------------------------------
 */

double
TkRangePixelToValue(rangePtr, x, y)
    register TkRange *rangePtr;		/* Information about widget. */
    int x, y;				/* Coordinates of point within
					 * window. */
{
    double value, pixelRange;

    if (rangePtr->orient == ORIENT_VERTICAL) {
	pixelRange = Tk_Height(rangePtr->tkwin) - rangePtr->sliderLength
		- 2*rangePtr->inset - 2*rangePtr->borderWidth;
	value = y;
    } else {
	pixelRange = Tk_Width(rangePtr->tkwin) - rangePtr->sliderLength
		- 2*rangePtr->inset - 2*rangePtr->borderWidth;
	value = x;
    }

    if (pixelRange <= 0) {
	/*
	 * Not enough room for the slider to actually slide:  just return
	 * the range's current value.
	 */

	return rangePtr->minvalue;
    }
    value -= rangePtr->sliderLength/2 + rangePtr->inset
		+ rangePtr->borderWidth;
    value /= pixelRange;
    if (value < 0) {
	value = 0;
    }
    if (value > 1) {
	value = 1;
    }
    value = rangePtr->fromValue +
		value * (rangePtr->toValue - rangePtr->fromValue);
    return TkRangeRoundToResolution(rangePtr, value);
}

/*
 *----------------------------------------------------------------------
 *
 * TkRangeValueToPixel --
 *
 *	Given a reading of the range, return the x-coordinate or
 *	y-coordinate corresponding to that reading, depending on
 *	whether the range is vertical or horizontal, respectively.
 *
 * Results:
 *	An integer value giving the pixel location corresponding
 *	to reading.  The value is restricted to lie within the
 *	defined range for the range.
 *
 * Side effects:
 *	None.
 *
 *----------------------------------------------------------------------
 */

int
TkRangeValueToPixel(rangePtr, value)
    register TkRange *rangePtr;		/* Information about widget. */
    double value;			/* Reading of the widget. */
{
    int y, pixelRange;
    double valueRange;

    valueRange = rangePtr->toValue - rangePtr->fromValue;
    pixelRange = ((rangePtr->orient == ORIENT_VERTICAL)
	    ? Tk_Height(rangePtr->tkwin) : Tk_Width(rangePtr->tkwin))
	- rangePtr->sliderLength - 2*rangePtr->inset - 2*rangePtr->borderWidth;
    if (valueRange == 0) {
	y = 0;
    } else {
	y = (int) ((value - rangePtr->fromValue) * pixelRange
		  / valueRange + 0.5);
	if (y < 0) {
	    y = 0;
	} else if (y > pixelRange) {
	    y = pixelRange;
	}
    }
    y += rangePtr->sliderLength/2 + rangePtr->inset + rangePtr->borderWidth;
    return y;
}
