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
 * Copyright SCI
 *
 * Changes Log
 * ~~~~~~~ ~~~
 * 9/95  Added nonZero flag to indicate whether min and max must be distinct
 */

static char sccsid[] = "@(#) tkRange.c 1.58 95/01/03 17:06:05";

#include "tkPort.h"
#include "default.h"
#include "tkRange.h"
#include "tkInt.h"
#include <math.h>

/*
 * A data structure of the following type is kept for each range
 * widget managed by this file:
 */

typedef struct {
    Tk_Window tkwin;		/* Window that embodies the range.  NULL
				 * means that the window has been destroyed
				 * but the data structures haven't yet been
				 * cleaned up.*/
    Display *display;		/* Display containing widget.  Used, among
				 * other things, so that resources can be
				 * freed even after tkwin has gone away. */
    Tcl_Interp *interp;		/* Interpreter associated with range. */
    Tk_Uid orientUid;		/* Orientation for window ("vertical" or
				 * "horizontal"). */
    int vertical;		/* Non-zero means vertical orientation,
				 * zero means horizontal. */
    int width;			/* Desired narrow dimension of range,
				 * in pixels. */
    int length;			/* Desired long dimension of range,
				 * in pixels. */
    double min_value;		/* Current min value of range. */
    double max_value;		/* Current max value of range. */
    char *min_varName;		/* Name of variable (malloc'ed) or NULL.
				 * If non-NULL, range's value tracks
				 * the contents of this variable and
				 * vice versa. */
    char *max_varName;		/* Name of variable (malloc'ed) or NULL.
				 * If non-NULL, range's value tracks
				 * the contents of this variable and
				 * vice versa. */
    double fromValue;		/* Value corresponding to left or top of
				 * range. */
    double toValue;		/* Value corresponding to right or bottom
				 * of range. */
    double tickInterval;	/* Distance between tick marks;  0 means
				 * don't display any tick marks. */
    double resolution;		/* All values are rounded to an even multiple
				 * of this value.  This value is always
				 * positive. */
    int digits;			/* Number of significant digits to print
				 * in values.  0 means we get to choose the
				 * number based on resolution and range. */
    char format[10];		/* Sprintf conversion specifier computed from
				 * digits and other information. */
    double bigIncrement;	/* Amount to use for large increments to
				 * range value.  (0 means we pick a value). */
    char *command;		/* Command prefix to use when invoking Tcl
				 * commands because the range value changed.
				 * NULL means don't invoke commands.
				 * Malloc'ed. */
    int repeatDelay;		/* How long to wait before auto-repeating
				 * on scrolling actions (in ms). */
    int repeatInterval;		/* Interval between autorepeats (in ms). */
    char *label;		/* Label to display above or to right of
				 * range;  NULL means don't display a
				 * label.  Malloc'ed. */
    int labelLength;		/* Number of non-NULL chars. in label. */
    Tk_Uid state;		/* Normal or disabled.  Value cannot be
				 * changed when range is disabled. */
    int nonZero;		/* Should min and max be distinct */
    /*
     * Information used when displaying widget:
     */

    int borderWidth;		/* Width of 3-D border around window. */
    Tk_3DBorder bgBorder;	/* Used for drawing slider and other
				 * background areas. */
    Tk_3DBorder activeBorder;	/* For drawing the slider when active. */
    XColor *troughColorPtr;	/* Color for drawing trough. */
    XColor *rangeColorPtr;	/* Color for drawing trough. */
    GC troughGC;		/* For drawing trough. */
    GC rangeGC;			/* For drawing trough. */
    GC copyGC;			/* Used for copying from pixmap onto screen. */
    XFontStruct *fontPtr;	/* Information about text font, or NULL. */
    XColor *textColorPtr;	/* Color for drawing text. */
    GC textGC;			/* GC for drawing text in normal mode. */
    int relief;			/* Indicates whether window as a whole is
				 * raised, sunken, or flat. */
    int highlightWidth;		/* Width in pixels of highlight to draw
				 * around widget when it has the focus.
				 * <= 0 means don't draw a highlight. */
    XColor *highlightColorPtr;	/* Color for drawing traversal highlight. */
    GC highlightGC;		/* For drawing traversal highlight. */
    int inset;			/* Total width of all borders, including
				 * traversal highlight and 3-D border.
				 * Indicates how much interior stuff must
				 * be offset from outside edges to leave
				 * room for borders. */
    int sliderLength;		/* Length of slider, measured in pixels along
				 * long dimension of range. */
    int showValue;		/* Non-zero means to display the range value
				 * below or to the left of the slider;  zero
				 * means don't display the value. */

    /*
     * Layout information for horizontal ranges, assuming that window
     * gets the size it requested:
     */

    int horizLabelY;		/* Y-coord at which to draw label. */
    int horizValueY;		/* Y-coord at which to draw value text. */
    int horizTroughY;		/* Y-coord of top of slider trough. */
    int horizTickY;		/* Y-coord at which to draw tick text. */
    /*
     * Layout information for vertical ranges, assuming that window
     * gets the size it requested:
     */

    int vertTickRightX;		/* X-location of right side of tick-marks. */
    int vertValueRightX;	/* X-location of right side of value string. */
    int vertTroughX;		/* X-location of range's slider trough. */
    int vertLabelX;		/* X-location of origin of label. */

    /*
     * Miscellaneous information:
     */

    Cursor cursor;		/* Current cursor for window, or None. */
    int flags;			/* Various flags;  see below for
				 * definitions. */
} Range;

/*
 * Flag bits for ranges:
 *
 * REDRAW_SLIDER -		1 means slider (and numerical readout) need
 *				to be redrawn.
 * REDRAW_OTHER -		1 means other stuff besides slider and value
 *				need to be redrawn.
 * REDRAW_ALL -			1 means the entire widget needs to be redrawn.
 * ACTIVE -			1 means the widget is active (the mouse is
 *				in its window).
 * BUTTON_PRESSED -		1 means a button press is in progress, so
 *				slider should appear depressed and should be
 *				draggable.
 * INVOKE_COMMAND -		1 means the range's command needs to be
 *				invoked during the next redisplay (the
 *				value of the range has changed since the
 *				last time the command was invoked).
 * SETTING_VAR -		1 means that the associated variable is
 *				being set by us, so there's no need for
 *				RangeVarProc to do anything.
 * NEVER_SET -			1 means that the range's value has never
 *				been set before (so must invoke -command and
 *				set associated variable even if the value
 *				doesn't appear to have changed).
 * GOT_FOCUS -			1 means that the focus is currently in
 *				this widget.
 */

#define REDRAW_SLIDER		1
#define REDRAW_OTHER		2
#define REDRAW_ALL		3
#define ACTIVE			4
#define BUTTON_PRESSED		8
#define INVOKE_COMMAND		0x10
#define SETTING_VAR		0x20
#define NEVER_SET		0x40
#define GOT_FOCUS		0x80

/*
 * Symbolic values for the active parts of a slider.  These are
 * the values that may be returned by the RangeElement procedure.
 */

#define OTHER		0
#define TROUGH1		1
#define MIN_SLIDER	2
#define TROUGH2		3
#define MAX_SLIDER	4
#define RANGE		5

/*
 * Space to leave between range area and text, and between text and
 * edge of window.
 */

#define SPACING 2

/*
 * How many characters of space to provide when formatting the
 * range's value:
 */

#define PRINT_CHARS 150

/*
 * Information used for argv parsing.
 */

static Tk_ConfigSpec configSpecs[] = {
    {TK_CONFIG_BORDER, "-activebackground", "activeBackground", "Foreground",
	DEF_RANGE_ACTIVE_BG_COLOR, Tk_Offset(Range, activeBorder),
	TK_CONFIG_COLOR_ONLY},
    {TK_CONFIG_BORDER, "-activebackground", "activeBackground", "Foreground",
	DEF_RANGE_ACTIVE_BG_MONO, Tk_Offset(Range, activeBorder),
	TK_CONFIG_MONO_ONLY},
    {TK_CONFIG_BORDER, "-background", "background", "Background",
	DEF_RANGE_BG_COLOR, Tk_Offset(Range, bgBorder),
	TK_CONFIG_COLOR_ONLY},
    {TK_CONFIG_BORDER, "-background", "background", "Background",
	DEF_RANGE_BG_MONO, Tk_Offset(Range, bgBorder),
	TK_CONFIG_MONO_ONLY},
    {TK_CONFIG_DOUBLE, "-bigincrement", "bigIncrement", "BigIncrement",
	DEF_RANGE_BIG_INCREMENT, Tk_Offset(Range, bigIncrement), 0},
    {TK_CONFIG_SYNONYM, "-bd", "borderWidth", (char *) NULL,
	(char *) NULL, 0, 0},
    {TK_CONFIG_SYNONYM, "-bg", "background", (char *) NULL,
	(char *) NULL, 0, 0},
    {TK_CONFIG_PIXELS, "-borderwidth", "borderWidth", "BorderWidth",
	DEF_RANGE_BORDER_WIDTH, Tk_Offset(Range, borderWidth), 0},
    {TK_CONFIG_STRING, "-command", "command", "Command",
	DEF_RANGE_COMMAND, Tk_Offset(Range, command), TK_CONFIG_NULL_OK},
    {TK_CONFIG_ACTIVE_CURSOR, "-cursor", "cursor", "Cursor",
	DEF_RANGE_CURSOR, Tk_Offset(Range, cursor), TK_CONFIG_NULL_OK},
    {TK_CONFIG_INT, "-digits", "digits", "Digits",
	DEF_RANGE_DIGITS, Tk_Offset(Range, digits), 0},
    {TK_CONFIG_SYNONYM, "-fg", "foreground", (char *) NULL,
	(char *) NULL, 0, 0},
    {TK_CONFIG_FONT, "-font", "font", "Font",
	DEF_RANGE_FONT, Tk_Offset(Range, fontPtr),
	0},
    {TK_CONFIG_COLOR, "-foreground", "foreground", "Foreground",
	DEF_RANGE_FG_COLOR, Tk_Offset(Range, textColorPtr),
	TK_CONFIG_COLOR_ONLY},
    {TK_CONFIG_COLOR, "-foreground", "foreground", "Foreground",
	DEF_RANGE_FG_MONO, Tk_Offset(Range, textColorPtr),
	TK_CONFIG_MONO_ONLY},
    {TK_CONFIG_DOUBLE, "-from", "from", "From",
	DEF_RANGE_FROM, Tk_Offset(Range, fromValue), 0},
    {TK_CONFIG_COLOR, "-highlightcolor", "highlightColor", "HighlightColor",
	DEF_RANGE_HIGHLIGHT, Tk_Offset(Range, highlightColorPtr), 0},
    {TK_CONFIG_PIXELS, "-highlightthickness", "highlightThickness",
	"HighlightThickness",
	DEF_RANGE_HIGHLIGHT_WIDTH, Tk_Offset(Range, highlightWidth), 0},
    {TK_CONFIG_STRING, "-label", "label", "Label",
	DEF_RANGE_LABEL, Tk_Offset(Range, label), TK_CONFIG_NULL_OK},
    {TK_CONFIG_PIXELS, "-length", "length", "Length",
	DEF_RANGE_LENGTH, Tk_Offset(Range, length), 0},
    {TK_CONFIG_UID, "-orient", "orient", "Orient",
	DEF_RANGE_ORIENT, Tk_Offset(Range, orientUid), 0},
    {TK_CONFIG_BOOLEAN, "-nonzero", "nonZero", "NonZero",
        DEF_RANGE_NON_ZERO, Tk_Offset(Range, nonZero), 0},
    {TK_CONFIG_RELIEF, "-relief", "relief", "Relief",
	DEF_RANGE_RELIEF, Tk_Offset(Range, relief), 0},
    {TK_CONFIG_INT, "-repeatdelay", "repeatDelay", "RepeatDelay",
	DEF_RANGE_REPEAT_DELAY, Tk_Offset(Range, repeatDelay), 0},
    {TK_CONFIG_INT, "-repeatinterval", "repeatInterval", "RepeatInterval",
	DEF_RANGE_REPEAT_INTERVAL, Tk_Offset(Range, repeatInterval), 0},
    {TK_CONFIG_DOUBLE, "-resolution", "resolution", "Resolution",
	DEF_RANGE_RESOLUTION, Tk_Offset(Range, resolution), 0},
    {TK_CONFIG_BOOLEAN, "-showvalue", "showValue", "ShowValue",
	DEF_RANGE_SHOW_VALUE, Tk_Offset(Range, showValue), 0},
    {TK_CONFIG_PIXELS, "-sliderlength", "sliderLength", "SliderLength",
	DEF_RANGE_SLIDER_LENGTH, Tk_Offset(Range, sliderLength), 0},
    {TK_CONFIG_UID, "-state", "state", "State",
	DEF_RANGE_STATE, Tk_Offset(Range, state), 0},
    {TK_CONFIG_DOUBLE, "-tickinterval", "tickInterval", "TickInterval",
	DEF_RANGE_TICK_INTERVAL, Tk_Offset(Range, tickInterval), 0},
    {TK_CONFIG_DOUBLE, "-to", "to", "To",
	DEF_RANGE_TO, Tk_Offset(Range, toValue), 0},
    {TK_CONFIG_COLOR, "-troughcolor", "troughColor", "Background",
	DEF_RANGE_TROUGH_COLOR, Tk_Offset(Range, troughColorPtr),
	TK_CONFIG_COLOR_ONLY},
    {TK_CONFIG_COLOR, "-troughcolor", "troughColor", "Background",
	DEF_RANGE_TROUGH_MONO, Tk_Offset(Range, troughColorPtr),
	TK_CONFIG_MONO_ONLY},
    {TK_CONFIG_COLOR, "-rangecolor", "rangeColor", "RangeColor",
	DEF_RANGE_RANGE_COLOR, Tk_Offset(Range, rangeColorPtr),
	TK_CONFIG_COLOR_ONLY},
    {TK_CONFIG_COLOR, "-rangecolor", "rangeColor", "RangeColor",
	DEF_RANGE_TROUGH_MONO, Tk_Offset(Range, rangeColorPtr),
	TK_CONFIG_MONO_ONLY},
    {TK_CONFIG_STRING, "-var_min", "var_min", "Var_min",
	DEF_RANGE_VARIABLE, Tk_Offset(Range, min_varName), TK_CONFIG_NULL_OK},
    {TK_CONFIG_STRING, "-var_max", "var_max", "Var_max",
	DEF_RANGE_VARIABLE, Tk_Offset(Range, max_varName), TK_CONFIG_NULL_OK},
    {TK_CONFIG_PIXELS, "-width", "width", "Width",
	DEF_RANGE_WIDTH, Tk_Offset(Range, width), 0},
    {TK_CONFIG_END, (char *) NULL, (char *) NULL, (char *) NULL,
	(char *) NULL, 0, 0}
};

/*
 * Forward declarations for procedures defined later in this file:
 */

static void		ComputeFormat _ANSI_ARGS_((Range *rangePtr));
static void		ComputeRangeGeometry _ANSI_ARGS_((Range *rangePtr));
static int		ConfigureRange _ANSI_ARGS_((Tcl_Interp *interp,
			    Range *rangePtr, int argc, char **argv,
			    int flags));
static void		DestroyRange _ANSI_ARGS_((ClientData clientData));
static void		DisplayHorizontalRange _ANSI_ARGS_((Range *rangePtr,
			    Drawable drawable, XRectangle *drawnAreaPtr));
static void		DisplayHorizontalValue _ANSI_ARGS_((Range *rangePtr,
			    Drawable drawable, double value, int top));
static void		DisplayVerticalRange _ANSI_ARGS_((Range *rangePtr,
			    Drawable drawable, XRectangle *drawnAreaPtr));
static void		DisplayVerticalValue _ANSI_ARGS_((Range *rangePtr,
			    Drawable drawable, double value, int rightEdge));
static void		EventuallyRedrawRange _ANSI_ARGS_((Range *rangePtr,
			    int what));
static double		PixelToValue _ANSI_ARGS_((Range *rangePtr, int x,
			    int y));
static double		RoundToResolution _ANSI_ARGS_((Range *rangePtr,
			    double value));
static int		RangeElement _ANSI_ARGS_((Range *rangePtr, int x,
			    int y));
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
static void		SetRangeMinValue _ANSI_ARGS_((Range *rangePtr,
			    double value, int setVar, int invokeCommand));
static void		SetRangeMaxValue _ANSI_ARGS_((Range *rangePtr,
			    double value, int setVar, int invokeCommand));
static int		ValueToPixel _ANSI_ARGS_((Range *rangePtr, double value));

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
    register Range *rangePtr;
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

    /*
     * Initialize fields that won't be initialized by ConfigureRange,
     * or which ConfigureRange expects to have reasonable values
     * (e.g. resource pointers).
     */

    rangePtr = (Range *) ckalloc(sizeof(Range));
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
    rangePtr->state = tkNormalUid;
    rangePtr->borderWidth = 0;
    rangePtr->bgBorder = NULL;
    rangePtr->activeBorder = NULL;
    rangePtr->troughColorPtr = NULL;
    rangePtr->rangeColorPtr = NULL;
    rangePtr->troughGC = None;
    rangePtr->rangeGC = None;
    rangePtr->copyGC = None;
    rangePtr->fontPtr = NULL;
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
    Tk_CreateEventHandler(rangePtr->tkwin,
	    ExposureMask|StructureNotifyMask|FocusChangeMask,
	    RangeEventProc, (ClientData) rangePtr);
    Tcl_CreateCommand(interp, Tk_PathName(rangePtr->tkwin), RangeWidgetCmd,
	    (ClientData) rangePtr, (void (*)()) NULL);
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
    register Range *rangePtr = (Range *) clientData;
    int result = TCL_OK;
    size_t length;
    int c;

    if (argc < 2) {
	Tcl_AppendResult(interp, "wrong # args: should be \"",
		argv[0], " option ?arg arg ...?\"", (char *) NULL);
	return TCL_ERROR;
    }
    Tk_Preserve((ClientData) rangePtr);
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
	    y = ValueToPixel(rangePtr, value);
	} else {
	    x = ValueToPixel(rangePtr, value);
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
	    y = ValueToPixel(rangePtr, value);
	} else {
	    x = ValueToPixel(rangePtr, value);
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
	value = PixelToValue(rangePtr, x, y);
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
	thing = RangeElement(rangePtr, x,y);
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
	    SetRangeMinValue(rangePtr, value, 1, 1);
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
	    SetRangeMaxValue(rangePtr, value, 1, 1);
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
	    SetRangeMinValue(rangePtr, val1, 1, 1);
	    SetRangeMaxValue(rangePtr, val2, 1, 1);
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
DestroyRange(clientData)
    ClientData clientData;	/* Info about range widget. */
{
    register Range *rangePtr = (Range *) clientData;

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
    register Range *rangePtr;	/* Information about widget;  may or may
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

    rangePtr->fromValue = RoundToResolution(rangePtr, rangePtr->fromValue);
    rangePtr->toValue = RoundToResolution(rangePtr, rangePtr->toValue);
    if (rangePtr->toValue == rangePtr->fromValue)
	rangePtr->toValue+=rangePtr->resolution;
    rangePtr->tickInterval = RoundToResolution(rangePtr,
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
    SetRangeMinValue(rangePtr, rangePtr->min_value, 1, 1);
    SetRangeMaxValue(rangePtr, rangePtr->max_value, 1, 1);

    if (rangePtr->label != NULL) {
	rangePtr->labelLength = strlen(rangePtr->label);
    } else {
	rangePtr->labelLength = 0;
    }

    if ((rangePtr->state != tkNormalUid)
	    && (rangePtr->state != tkDisabledUid)
	    && (rangePtr->state != tkActiveUid)) {
	Tcl_AppendResult(interp, "bad state value \"", rangePtr->state,
		"\":  must be normal, active, or disabled", (char *) NULL);
	rangePtr->state = tkNormalUid;
	return TCL_ERROR;
    }

    Tk_SetBackgroundFromBorder(rangePtr->tkwin, rangePtr->bgBorder);

    gcValues.foreground = rangePtr->troughColorPtr->pixel;
    newGC = Tk_GetGC(rangePtr->tkwin, GCForeground, &gcValues);
    if (rangePtr->troughGC != None) {
	Tk_FreeGC(rangePtr->display, rangePtr->troughGC);
    }
    rangePtr->troughGC = newGC;

    gcValues.foreground = rangePtr->rangeColorPtr->pixel;
    newGC = Tk_GetGC(rangePtr->tkwin, GCForeground, &gcValues);
    if (rangePtr->rangeGC != None) {
	Tk_FreeGC(rangePtr->display, rangePtr->rangeGC);
    }
    rangePtr->rangeGC = newGC;

    if (rangePtr->copyGC == None) {
	gcValues.graphics_exposures = False;
	rangePtr->copyGC = Tk_GetGC(rangePtr->tkwin, GCGraphicsExposures,
	    &gcValues);
    }
    if (rangePtr->highlightWidth < 0) {
	rangePtr->highlightWidth = 0;
    }
    if (rangePtr->highlightWidth > 0) {
	gcValues.foreground = rangePtr->highlightColorPtr->pixel;
	newGC = Tk_GetGC(rangePtr->tkwin, GCForeground, &gcValues);
	if (rangePtr->highlightGC != None) {
	    Tk_FreeGC(rangePtr->display, rangePtr->highlightGC);
	}
	rangePtr->highlightGC = newGC;
    }
    gcValues.font = rangePtr->fontPtr->fid;
    gcValues.foreground = rangePtr->textColorPtr->pixel;
    newGC = Tk_GetGC(rangePtr->tkwin, GCForeground|GCFont, &gcValues);
    if (rangePtr->textGC != None) {
	Tk_FreeGC(rangePtr->display, rangePtr->textGC);
    }
    rangePtr->textGC = newGC;

    rangePtr->inset = rangePtr->highlightWidth + rangePtr->borderWidth;

    /*
     * Recompute display-related information, and let the geometry
     * manager know how much space is needed now.
     */

    ComputeRangeGeometry(rangePtr);

    EventuallyRedrawRange(rangePtr, REDRAW_ALL);
    return TCL_OK;
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
    Range *rangePtr;			/* Information about range widget. */
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
	leastSigDigit = floor(log10(rangePtr->resolution));
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
    register Range *rangePtr;		/* Information about widget. */
{
    XCharStruct bbox;
    char valueString[PRINT_CHARS];
    int dummy, lineHeight, valuePixels, x, y, extraSpace;

    /*
     * Horizontal ranges are simpler than vertical ones because
     * all sizes are the same (the height of a line of text);
     * handle them first and then quit.
     */

    if (!rangePtr->vertical) {
	lineHeight = rangePtr->fontPtr->ascent + rangePtr->fontPtr->descent;
	y = rangePtr->inset;
	extraSpace = 0;
	if (rangePtr->labelLength != 0) {
	    rangePtr->horizLabelY = y + SPACING;
	    y += lineHeight + SPACING;
	    extraSpace = SPACING;
	}
	if (rangePtr->showValue) {
	    rangePtr->horizValueY = y + SPACING;
	    y += lineHeight + SPACING;
	    extraSpace = SPACING;
	} else {
	    rangePtr->horizValueY = y;
	}
	y += extraSpace;
	rangePtr->horizTroughY = y;
	y += rangePtr->width + 2*rangePtr->borderWidth;
	if (rangePtr->tickInterval != 0) {
	    rangePtr->horizTickY = y + SPACING;
	    y += lineHeight + 2*SPACING;
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
    XTextExtents(rangePtr->fontPtr, valueString, (int) strlen(valueString),
	    &dummy, &dummy, &dummy, &bbox);
    valuePixels = bbox.rbearing - bbox.lbearing;
    sprintf(valueString, rangePtr->format, rangePtr->toValue);
    XTextExtents(rangePtr->fontPtr, valueString, (int) strlen(valueString),
	    &dummy, &dummy, &dummy, &bbox);
    if (valuePixels < bbox.rbearing - bbox.lbearing) {
	valuePixels = bbox.rbearing - bbox.lbearing;
    }

    /*
     * Assign x-locations to the elements of the range, working from
     * left to right.
     */

    x = rangePtr->inset;
    if ((rangePtr->tickInterval != 0) && (rangePtr->showValue)) {
	rangePtr->vertTickRightX = x + SPACING + valuePixels;
	rangePtr->vertValueRightX = rangePtr->vertTickRightX + valuePixels
		+ rangePtr->fontPtr->ascent/2;
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
	XTextExtents(rangePtr->fontPtr, rangePtr->label,
		rangePtr->labelLength, &dummy, &dummy, &dummy, &bbox);
	rangePtr->vertLabelX = x + rangePtr->fontPtr->ascent/2 - bbox.lbearing;
	x = rangePtr->vertLabelX + bbox.rbearing
		+ rangePtr->fontPtr->ascent/2;
    }
    Tk_GeometryRequest(rangePtr->tkwin, x + rangePtr->inset,
	    rangePtr->length + 2*rangePtr->inset);
    Tk_SetInternalBorder(rangePtr->tkwin, rangePtr->inset);
}

/*
 *--------------------------------------------------------------
 *
 * DisplayVerticalRange --
 *
 *	This procedure redraws the contents of a vertical range
 *	window.  It is invoked as a do-when-idle handler, so it only
 *	runs when there's nothing else for the application to do.
 *
 * Results:
 *	There is no return value.  If only a part of the range needs
 *	to be redrawn, then drawnAreaPtr is modified to reflect the
 *	area that was actually modified.
 *
 * Side effects:
 *	Information appears on the screen.
 *
 *--------------------------------------------------------------
 */

static void
DisplayVerticalRange(rangePtr, drawable, drawnAreaPtr)
    Range *rangePtr;			/* Widget record for range. */
    Drawable drawable;			/* Where to display range (window
					 * or pixmap). */
    XRectangle *drawnAreaPtr;		/* Initally contains area of window;
					 * if only a part of the range is
					 * redrawn, gets modified to reflect
					 * the part of the window that was
					 * redrawn. */
{
    Tk_Window tkwin = rangePtr->tkwin;
    int x, y1, y2, width, height, shadowWidth, relief;
    double tickValue;
    Tk_3DBorder sliderBorder;
    int rangeFillHeight;
    /*
     * Display the information from left to right across the window.
     */

    if (!(rangePtr->flags & REDRAW_OTHER)) {
	drawnAreaPtr->x = rangePtr->vertTickRightX;
	drawnAreaPtr->y = rangePtr->inset;
	drawnAreaPtr->width = rangePtr->vertTroughX + rangePtr->width
		+ 2*rangePtr->borderWidth - rangePtr->vertTickRightX;
	drawnAreaPtr->height -= 2*rangePtr->inset;
    }
    Tk_Fill3DRectangle(tkwin, drawable, rangePtr->bgBorder,
	    drawnAreaPtr->x, drawnAreaPtr->y, drawnAreaPtr->width,
	    drawnAreaPtr->height, 0, TK_RELIEF_FLAT);
    if (rangePtr->flags & REDRAW_OTHER) {
	/*
	 * Display the tick marks.
	 */

	if (rangePtr->tickInterval != 0) {
	    for (tickValue = rangePtr->fromValue; ;
		    tickValue += rangePtr->tickInterval) {
		/*
		 * The RoundToResolution call gets rid of accumulated
		 * round-off errors, if any.
		 */

		tickValue = RoundToResolution(rangePtr, tickValue);
		if (rangePtr->toValue >= rangePtr->fromValue) {
		    if (tickValue > rangePtr->toValue) {
			break;
		    }
		} else {
		    if (tickValue < rangePtr->toValue) {
			break;
		    }
		}
		DisplayVerticalValue(rangePtr, drawable, tickValue,
			rangePtr->vertTickRightX);
	    }
	}
    }

    /*
     * Display the value, if it is desired.
     */

    if (rangePtr->showValue) {
	DisplayVerticalValue(rangePtr, drawable, rangePtr->min_value,
		rangePtr->vertValueRightX);
	DisplayVerticalValue(rangePtr, drawable, rangePtr->max_value,
		rangePtr->vertValueRightX);
    }

    /*
     * Display the trough and the slider.
     */

    Tk_Draw3DRectangle(tkwin, drawable,
	    rangePtr->bgBorder, rangePtr->vertTroughX, rangePtr->inset,
	    rangePtr->width + 2*rangePtr->borderWidth,
	    Tk_Height(tkwin) - 2*rangePtr->inset, rangePtr->borderWidth,
	    TK_RELIEF_SUNKEN);
    XFillRectangle(rangePtr->display, drawable, rangePtr->troughGC,
	    rangePtr->vertTroughX + rangePtr->borderWidth,
	    rangePtr->inset + rangePtr->borderWidth,
	    (unsigned) rangePtr->width,
	    (unsigned) (Tk_Height(tkwin) - 2*rangePtr->inset
                - 2*rangePtr->borderWidth));
    if (rangePtr->state == tkActiveUid) {
	sliderBorder = rangePtr->activeBorder;
    } else {
	sliderBorder = rangePtr->bgBorder;
    }
    width = rangePtr->width;
    height = rangePtr->sliderLength/2;
    x = rangePtr->vertTroughX + rangePtr->borderWidth;
    y1 = ValueToPixel(rangePtr, rangePtr->min_value) - height;
    y2 = ValueToPixel(rangePtr, rangePtr->max_value);
    shadowWidth = rangePtr->borderWidth/2;
    if (shadowWidth == 0) {
	shadowWidth = 1;
    }
    relief = (rangePtr->flags & BUTTON_PRESSED) ? TK_RELIEF_SUNKEN
	    : TK_RELIEF_RAISED;

    rangeFillHeight = y2 - y1 - height - 2 * shadowWidth;
    if (rangeFillHeight>0) {		/* enough space to draw range fill */
	XFillRectangle(rangePtr->display, drawable, rangePtr->rangeGC,
		       rangePtr->vertTroughX + rangePtr->borderWidth,
		       y1 + height + shadowWidth,
		       (unsigned) rangePtr->width,
		       (unsigned) rangeFillHeight);
    }
    Tk_Draw3DRectangle(tkwin, drawable, sliderBorder,
		       x, y1, width, height, shadowWidth, relief);
    Tk_Draw3DRectangle(tkwin, drawable, sliderBorder,
		       x, y2, width, height, shadowWidth, relief);
    x+=shadowWidth;
    y1+=shadowWidth;
    y2+=shadowWidth;
    width -= 2*shadowWidth;
    height -= 2*shadowWidth;
    Tk_Draw3DRectangle(tkwin, drawable, sliderBorder,
		       x, y1, width, height, shadowWidth, relief);
    Tk_Draw3DRectangle(tkwin, drawable, sliderBorder,
		       x, y2, width, height, shadowWidth, relief);

    /*
     * Draw the label to the right of the range.
     */

    if ((rangePtr->flags & REDRAW_OTHER) && (rangePtr->labelLength != 0)) {
	XDrawString(rangePtr->display, drawable,
	    rangePtr->textGC, rangePtr->vertLabelX,
	    rangePtr->inset + (3*rangePtr->fontPtr->ascent)/2,
	    rangePtr->label, rangePtr->labelLength);
    }
}

/*
 *----------------------------------------------------------------------
 *
 * DisplayVerticalValue --
 *
 *	This procedure is called to display values (range readings)
 *	for vertically-oriented ranges.
 *
 * Results:
 *	None.
 *
 * Side effects:
 *	The numerical value corresponding to value is displayed with
 *	its right edge at "rightEdge", and at a vertical position in
 *	the range that corresponds to "value".
 *
 *----------------------------------------------------------------------
 */

static void
DisplayVerticalValue(rangePtr, drawable, value, rightEdge)
    register Range *rangePtr;	/* Information about widget in which to
				 * display value. */
    Drawable drawable;		/* Pixmap or window in which to draw
				 * the value. */
    double value;		/* Y-coordinate of number to display,
				 * specified in application coords, not
				 * in pixels (we'll compute pixels). */
    int rightEdge;		/* X-coordinate of right edge of text,
				 * specified in pixels. */
{
    register Tk_Window tkwin = rangePtr->tkwin;
    int y, dummy, length;
    char valueString[PRINT_CHARS];
    XCharStruct bbox;

    y = ValueToPixel(rangePtr, value) + rangePtr->fontPtr->ascent/2;
    sprintf(valueString, rangePtr->format, value);
    length = strlen(valueString);
    XTextExtents(rangePtr->fontPtr, valueString, length,
	    &dummy, &dummy, &dummy, &bbox);

    /*
     * Adjust the y-coordinate if necessary to keep the text entirely
     * inside the window.
     */

    if ((y - bbox.ascent) < (rangePtr->inset + SPACING)) {
	y = rangePtr->inset + SPACING + bbox.ascent;
    }
    if ((y + bbox.descent) > (Tk_Height(tkwin) - rangePtr->inset - SPACING)) {
	y = Tk_Height(tkwin) - rangePtr->inset - SPACING - bbox.descent;
    }
    XDrawString(rangePtr->display, drawable, rangePtr->textGC,
	    rightEdge - bbox.rbearing, y, valueString, length);
}

/*
 *--------------------------------------------------------------
 *
 * DisplayHorizontalRange --
 *
 *	This procedure redraws the contents of a horizontal range
 *	window.  It is invoked as a do-when-idle handler, so it only
 *	runs when there's nothing else for the application to do.
 *
 * Results:
 *	There is no return value.  If only a part of the range needs
 *	to be redrawn, then drawnAreaPtr is modified to reflect the
 *	area that was actually modified.
 *
 * Side effects:
 *	Information appears on the screen.
 *
 *--------------------------------------------------------------
 */
static void
DisplayHorizontalRange(rangePtr, drawable, drawnAreaPtr)
    Range *rangePtr;			/* Widget record for range. */
    Drawable drawable;			/* Where to display range (window
					 * or pixmap). */
    XRectangle *drawnAreaPtr;		/* Initally contains area of window;
					 * if only a part of the range is
					 * redrawn, gets modified to reflect
					 * the part of the window that was
					 * redrawn. */
{
    register Tk_Window tkwin = rangePtr->tkwin;
    int x1, x2, y, width, height, shadowWidth, relief;
    double tickValue;
    Tk_3DBorder sliderBorder;
    int rangeFillWidth;

    /*
     * Display the information from bottom to top across the window.
     */

    if (!(rangePtr->flags & REDRAW_OTHER)) {
	drawnAreaPtr->x = rangePtr->inset;
	drawnAreaPtr->y = rangePtr->horizValueY;
	drawnAreaPtr->width -= 2*rangePtr->inset;
	drawnAreaPtr->height = rangePtr->horizTroughY + rangePtr->width
		+ 2*rangePtr->borderWidth - rangePtr->horizValueY;
    }
    Tk_Fill3DRectangle(tkwin, drawable, rangePtr->bgBorder,
	    drawnAreaPtr->x, drawnAreaPtr->y, drawnAreaPtr->width,
	    drawnAreaPtr->height, 0, TK_RELIEF_FLAT);
    if (rangePtr->flags & REDRAW_OTHER) {
	/*
	 * Display the tick marks.
	 */

	if (rangePtr->tickInterval != 0) {
	    for (tickValue = rangePtr->fromValue; ;
		    tickValue += rangePtr->tickInterval) {
		/*
		 * The RoundToResolution call gets rid of accumulated
		 * round-off errors, if any.
		 */

		tickValue = RoundToResolution(rangePtr, tickValue);
		if (rangePtr->toValue > rangePtr->fromValue) {
		    if (tickValue > rangePtr->toValue) {
			break;
		    }
		} else {
		    if (tickValue < rangePtr->toValue) {
			break;
		    }
		}
		DisplayHorizontalValue(rangePtr, drawable, tickValue,
			rangePtr->horizTickY);
	    }
	}
    }

    /*
     * Display the value, if it is desired.
     */

    if (rangePtr->showValue) {
	DisplayHorizontalValue(rangePtr, drawable, rangePtr->min_value,
		rangePtr->horizValueY);
	DisplayHorizontalValue(rangePtr, drawable, rangePtr->max_value,
		rangePtr->horizValueY);
    }

    /*
     * Display the trough and the slider.
     */

    y = rangePtr->horizTroughY;
    Tk_Draw3DRectangle(tkwin, drawable,
	    rangePtr->bgBorder, rangePtr->inset, y,
	    Tk_Width(tkwin) - 2*rangePtr->inset,	       
	    rangePtr->width + 2*rangePtr->borderWidth,
	    rangePtr->borderWidth, TK_RELIEF_SUNKEN);
    XFillRectangle(rangePtr->display, drawable, rangePtr->troughGC,
	    rangePtr->inset + rangePtr->borderWidth,
	    y + rangePtr->borderWidth,
	    (unsigned) (Tk_Width(tkwin) - 2*rangePtr->inset
		- 2*rangePtr->borderWidth),
	    (unsigned) rangePtr->width);
    if (rangePtr->state == tkActiveUid) {
	sliderBorder = rangePtr->activeBorder;
    } else {
	sliderBorder = rangePtr->bgBorder;
    }
    width = rangePtr->sliderLength/2;
    height = rangePtr->width;
    x1 = ValueToPixel(rangePtr, rangePtr->min_value) - width;
    x2 = ValueToPixel(rangePtr, rangePtr->max_value);
    y += rangePtr->borderWidth;
    shadowWidth = rangePtr->borderWidth/2;
    if (shadowWidth == 0) {
	shadowWidth = 1;
    }
    relief = (rangePtr->flags & BUTTON_PRESSED) ? TK_RELIEF_SUNKEN
	    : TK_RELIEF_RAISED;
    rangeFillWidth = x2 - x1 - width - 2*shadowWidth;
    if (rangeFillWidth>0) {		/* enough space to draw range fill */
	XFillRectangle(rangePtr->display, drawable, rangePtr->rangeGC,
		       x1 + width + shadowWidth,
		       y,
		       (unsigned) rangeFillWidth,
		       (unsigned) rangePtr->width);
    }
    Tk_Draw3DRectangle(tkwin, drawable, sliderBorder,
		       x1, y, width, height, shadowWidth, relief);
    Tk_Draw3DRectangle(tkwin, drawable, sliderBorder,
		       x2, y, width, height, shadowWidth, relief);
    x1 += shadowWidth;
    x2 += shadowWidth;
    y += shadowWidth;
    width -= 2*shadowWidth;
    height -= 2*shadowWidth;
    Tk_Fill3DRectangle(tkwin, drawable, sliderBorder,
		       x1, y, width, height, shadowWidth, relief);
    Tk_Fill3DRectangle(tkwin, drawable, sliderBorder,
		       x2, y, width, height, shadowWidth, relief);

    /*
     * Draw the label at the top of the range.
     */

    if ((rangePtr->flags & REDRAW_OTHER) && (rangePtr->labelLength != 0)) {
	XDrawString(rangePtr->display, drawable,
	    rangePtr->textGC, rangePtr->inset + rangePtr->fontPtr->ascent/2,
	    rangePtr->horizLabelY + rangePtr->fontPtr->ascent,
	    rangePtr->label, rangePtr->labelLength);
    }
}

/*
 *----------------------------------------------------------------------
 *
 * DisplayHorizontalValue --
 *
 *	This procedure is called to display values (range readings)
 *	for horizontally-oriented ranges.
 *
 * Results:
 *	None.
 *
 * Side effects:
 *	The numerical value corresponding to value is displayed with
 *	its bottom edge at "bottom", and at a horizontal position in
 *	the range that corresponds to "value".
 *
 *----------------------------------------------------------------------
 */

static void
DisplayHorizontalValue(rangePtr, drawable, value, top)
    register Range *rangePtr;	/* Information about widget in which to
				 * display value. */
    Drawable drawable;		/* Pixmap or window in which to draw
				 * the value. */
    double value;		/* X-coordinate of number to display,
				 * specified in application coords, not
				 * in pixels (we'll compute pixels). */
    int top;			/* Y-coordinate of top edge of text,
				 * specified in pixels. */
{
    register Tk_Window tkwin = rangePtr->tkwin;
    int x, y, dummy, length;
    char valueString[PRINT_CHARS];
    XCharStruct bbox;

    x = ValueToPixel(rangePtr, value);
    y = top + rangePtr->fontPtr->ascent;
    sprintf(valueString, rangePtr->format, value);
    length = strlen(valueString);
    XTextExtents(rangePtr->fontPtr, valueString, length,
	    &dummy, &dummy, &dummy, &bbox);

    /*
     * Adjust the x-coordinate if necessary to keep the text entirely
     * inside the window.
     */

    x -= (bbox.rbearing - bbox.lbearing)/2;
    if ((x + bbox.lbearing) < (rangePtr->inset + SPACING)) {
	x = rangePtr->inset + SPACING - bbox.lbearing;
    }
    if ((x + bbox.rbearing) > (Tk_Width(tkwin) - rangePtr->inset)) {
	x = Tk_Width(tkwin) - rangePtr->inset - SPACING - bbox.rbearing;
    }
    XDrawString(rangePtr->display, drawable, rangePtr->textGC, x, y,
	    valueString, length);
}

/*
 *----------------------------------------------------------------------
 *
 * DisplayRange --
 *
 *	This procedure is invoked as an idle handler to redisplay
 *	the contents of a range widget.
 *
 * Results:
 *	None.
 *
 * Side effects:
 *	The range gets redisplayed.
 *
 *----------------------------------------------------------------------
 */

void
DisplayRange(clientData)
    ClientData clientData;	/* Widget record for range. */
{
    Range *rangePtr = (Range *) clientData;
    Tk_Window tkwin = rangePtr->tkwin;
    Pixmap pixmap;
    int result;
    char string[PRINT_CHARS];
    XRectangle drawnArea;

    if ((rangePtr->tkwin == NULL) || !Tk_IsMapped(rangePtr->tkwin)) {
	goto done;
    }

    /*
     * Invoke the range's command if needed.
     */

    Tk_Preserve((ClientData) rangePtr);
    if ((rangePtr->flags & INVOKE_COMMAND) && (rangePtr->command != NULL)) {
	sprintf(string, rangePtr->format, rangePtr->min_value);
	result = Tcl_VarEval(rangePtr->interp, rangePtr->command,
		" ", string, (char *) NULL);
	if (result != TCL_OK) {
	    Tcl_AddErrorInfo(rangePtr->interp,
		    "\n    (command executed by range)");
	    Tk_BackgroundError(rangePtr->interp);
	}
	sprintf(string, rangePtr->format, rangePtr->max_value);
	result = Tcl_VarEval(rangePtr->interp, rangePtr->command,
		" ", string, (char *) NULL);
	if (result != TCL_OK) {
	    Tcl_AddErrorInfo(rangePtr->interp,
		    "\n    (command executed by range)");
	    Tk_BackgroundError(rangePtr->interp);
	}
    }
    rangePtr->flags &= ~INVOKE_COMMAND;
    if (rangePtr->tkwin == NULL) {
	Tk_Release((ClientData) rangePtr);
	return;
    }
    Tk_Release((ClientData) rangePtr);

    /*
     * In order to avoid screen flashes, this procedure redraws
     * the range in a pixmap, then copies the pixmap to the
     * screen in a single operation.  This means that there's no
     * point in time where the on-sreen image has been cleared.
     */

    pixmap = Tk_GetPixmap(rangePtr->display, Tk_WindowId(tkwin),
	    (unsigned) Tk_Width(tkwin), (unsigned) Tk_Height(tkwin),
	    (unsigned) Tk_Depth(tkwin));
    drawnArea.x = 0;
    drawnArea.y = 0;
    drawnArea.width = Tk_Width(tkwin);
    drawnArea.height = Tk_Height(tkwin);

    /*
     * Much of the redisplay is done totally differently for
     * horizontal and vertical ranges.  Handle the part that's
     * different.
     */

    if (rangePtr->vertical) {
	DisplayVerticalRange(rangePtr, pixmap, &drawnArea);
    } else {
	DisplayHorizontalRange(rangePtr, pixmap, &drawnArea);
    }

    /*
     * Now handle the part of redisplay that is the same for
     * horizontal and vertical ranges:  border and traversal
     * highlight.
     */

    if (rangePtr->flags & REDRAW_OTHER) {
	if (rangePtr->relief != TK_RELIEF_FLAT) {
	    Tk_Draw3DRectangle(tkwin, pixmap,
		    rangePtr->bgBorder, rangePtr->highlightWidth,
		    rangePtr->highlightWidth,
		    Tk_Width(tkwin) - 2*rangePtr->highlightWidth,
		    Tk_Height(tkwin) - 2*rangePtr->highlightWidth,
		    rangePtr->borderWidth, rangePtr->relief);
	}
	if (rangePtr->highlightWidth != 0) {
	    GC gc;
    
	    if (rangePtr->flags & GOT_FOCUS) {
		gc = rangePtr->highlightGC;
	    } else {
		gc = Tk_3DBorderGC(tkwin, rangePtr->bgBorder, TK_3D_FLAT_GC);
	    }
	    Tk_DrawFocusHighlight(tkwin, gc, rangePtr->highlightWidth, pixmap);
	}
    }

    /*
     * Copy the information from the off-screen pixmap onto the screen,
     * then delete the pixmap.
     */

    XCopyArea(rangePtr->display, pixmap, Tk_WindowId(tkwin),
	    rangePtr->copyGC, drawnArea.x, drawnArea.y, drawnArea.width,
	    drawnArea.height, drawnArea.x, drawnArea.y);
    Tk_FreePixmap(rangePtr->display, pixmap);

    done:
    rangePtr->flags &= ~REDRAW_ALL;
}

/*
 *----------------------------------------------------------------------
 *
 * RangeElement --
 *
 *	Determine which part of a range widget lies under a given
 *	point.
 *
 * Results:
 *	The return value is either TROUGH1, SLIDER, TROUGH2, or
 *	OTHER, depending on which of the range's active elements
 *	(if any) is under the point at (x,y).
 *
 * Side effects:
 *	None.
 *
 *----------------------------------------------------------------------
 */

static int
RangeElement(rangePtr, x, y)
    Range *rangePtr;		/* Widget record for range. */
    int x, y;			/* Coordinates within rangePtr's window. */
{
    int sliderMin;
    int sliderMax;
    if (rangePtr->vertical) {
	if ((x < rangePtr->vertTroughX)
		|| (x >= (rangePtr->vertTroughX + 2*rangePtr->borderWidth +
		rangePtr->width))) {
	    return OTHER;
	}
	if ((y < rangePtr->inset)
		|| (y >= (Tk_Height(rangePtr->tkwin) - rangePtr->inset))) {
	    return OTHER;
	}
	sliderMin = ValueToPixel(rangePtr, rangePtr->min_value)-
	    rangePtr->sliderLength/2;
	sliderMax = ValueToPixel(rangePtr, rangePtr->max_value)+
	    rangePtr->sliderLength/2;
	
	if (y < sliderMin) {
	    return TROUGH1;
	}
	if (y > sliderMax) {
	    return TROUGH2;
	}
	sliderMin += rangePtr->sliderLength/2;
	sliderMax -= rangePtr->sliderLength/2;
	if (y < sliderMin) {
	    return MIN_SLIDER;
	}
	if (y > sliderMax) {
	    return MAX_SLIDER;
	}
	return RANGE;
    }

    if ((y < rangePtr->horizTroughY)
	    || (y >= (rangePtr->horizTroughY + 2*rangePtr->borderWidth +
	    rangePtr->width))) {
	return OTHER;
    }
    if ((x < rangePtr->inset)
	    || (x >= Tk_Width(rangePtr->tkwin) - rangePtr->inset)) {
	return OTHER;
    }
    sliderMin = ValueToPixel(rangePtr, rangePtr->min_value)-
	rangePtr->sliderLength/2;
    sliderMax = ValueToPixel(rangePtr, rangePtr->max_value)+
	rangePtr->sliderLength/2;

    if (x < sliderMin) {
	return TROUGH1;
    }
    if (x > sliderMax) {
	return TROUGH2;
    }
    sliderMin += rangePtr->sliderLength/2;
    sliderMax -= rangePtr->sliderLength/2;
    if (x < sliderMin) {
	return MIN_SLIDER;
    }
    if (x > sliderMax) {
	return MAX_SLIDER;
    }
    return RANGE;
}

/*
 *----------------------------------------------------------------------
 *
 * PixelToValue --
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

static double
PixelToValue(rangePtr, x, y)
    register Range *rangePtr;		/* Information about widget. */
    int x, y;				/* Coordinates of point within
					 * window. */
{
    double value, pixelRange;

    if (rangePtr->vertical) {
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

	return rangePtr->min_value;
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
    return RoundToResolution(rangePtr, value);
}

/*
 *----------------------------------------------------------------------
 *
 * ValueToPixel --
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

static int
ValueToPixel(rangePtr, value)
    register Range *rangePtr;		/* Information about widget. */
    double value;			/* Reading of the widget. */
{
    int y, pixelRange;
    double valueRange;

    valueRange = rangePtr->toValue - rangePtr->fromValue;
    pixelRange = (rangePtr->vertical ? Tk_Height(rangePtr->tkwin)
	    : Tk_Width(rangePtr->tkwin)) - rangePtr->sliderLength
	    - 2*rangePtr->inset - 2*rangePtr->borderWidth;
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
    Range *rangePtr = (Range *) clientData;

    if ((eventPtr->type == Expose) && (eventPtr->xexpose.count == 0)) {
	EventuallyRedrawRange(rangePtr, REDRAW_ALL);
    } else if (eventPtr->type == DestroyNotify) {
	Tcl_DeleteCommand(rangePtr->interp, Tk_PathName(rangePtr->tkwin));
	rangePtr->tkwin = NULL;
	if (rangePtr->flags & REDRAW_ALL) {
	    Tk_CancelIdleCall(DisplayRange, (ClientData) rangePtr);
	}
	Tk_EventuallyFree((ClientData) rangePtr, DestroyRange);
    } else if (eventPtr->type == ConfigureNotify) {
	ComputeRangeGeometry(rangePtr);
    } else if (eventPtr->type == FocusIn) {
	if (eventPtr->xfocus.detail != NotifyPointer) {
	    rangePtr->flags |= GOT_FOCUS;
	    if (rangePtr->highlightWidth > 0) {
		EventuallyRedrawRange(rangePtr, REDRAW_ALL);
	    }
	}
    } else if (eventPtr->type == FocusOut) {
	if (eventPtr->xfocus.detail != NotifyPointer) {
	    rangePtr->flags &= ~GOT_FOCUS;
	    if (rangePtr->highlightWidth > 0) {
		EventuallyRedrawRange(rangePtr, REDRAW_ALL);
	    }
	}
    }
}

/*
 *--------------------------------------------------------------
 *
 * SetRangeMinValue --
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

static void
SetRangeMinValue(rangePtr, value, setVar, invokeCommand)
    register Range *rangePtr;	/* Info about widget. */
    double value;		/* New value for range.  Gets adjusted
				 * if it's off the range. */
    int setVar;			/* Non-zero means reflect new value through
				 * to associated variable, if any. */
    int invokeCommand;		/* Non-zero means invoked -command option
				 * to notify of new value, 0 means don't. */
{
    char string[PRINT_CHARS];
    value = RoundToResolution(rangePtr, value);
    if (value<rangePtr->fromValue) value=rangePtr->fromValue;
    if (value>=rangePtr->toValue) {
	if (rangePtr->nonZero) 
	    value=rangePtr->toValue-rangePtr->resolution;
	else if (value>rangePtr->toValue)
	    value=rangePtr->toValue;
    }
    if (value>=rangePtr->max_value) {
	if (rangePtr->nonZero) /* we know we're not at the toValue here */
	    SetRangeMaxValue(rangePtr, value+rangePtr->resolution, setVar,
			     invokeCommand);
	else if (value>rangePtr->max_value)
	    SetRangeMaxValue(rangePtr, value, setVar, invokeCommand);
    }
    if (rangePtr->flags & NEVER_SET) {
	rangePtr->flags &= ~NEVER_SET;
    } else if (rangePtr->min_value == value) {
	return;
    }
    rangePtr->min_value = value;
    if (invokeCommand) {
	rangePtr->flags |= INVOKE_COMMAND;
    }
    EventuallyRedrawRange(rangePtr, REDRAW_SLIDER);

    if (setVar && (rangePtr->min_varName != NULL)) {
	sprintf(string, rangePtr->format, rangePtr->min_value);
	rangePtr->flags |= SETTING_VAR;
	Tcl_SetVar(rangePtr->interp, rangePtr->min_varName, string,
	       TCL_GLOBAL_ONLY);
	rangePtr->flags &= ~SETTING_VAR;
    }
}

/*
 *--------------------------------------------------------------
 *
 * SetRangeMaxValue --
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

static void
SetRangeMaxValue(rangePtr, value, setVar, invokeCommand)
    register Range *rangePtr;	/* Info about widget. */
    double value;		/* New value for range.  Gets adjusted
				 * if it's off the range. */
    int setVar;			/* Non-zero means reflect new value through
				 * to associated variable, if any. */
    int invokeCommand;		/* Non-zero means invoked -command option
				 * to notify of new value, 0 means don't. */
{
    char string[PRINT_CHARS];
    value = RoundToResolution(rangePtr, value);
    if (value<=rangePtr->fromValue) {
	if (rangePtr->nonZero)
	    value=rangePtr->fromValue+rangePtr->resolution;
	else if (value<rangePtr->fromValue)
	    value=rangePtr->fromValue;
    }
    if (value>rangePtr->toValue) value=rangePtr->toValue;
    if (value<=rangePtr->min_value) {
	if (rangePtr->nonZero) /* we know we're not at the fromValue here */
	    SetRangeMinValue(rangePtr, value-rangePtr->resolution, setVar, 
			     invokeCommand);
	else if (value<rangePtr->min_value)
	    SetRangeMinValue(rangePtr, value, setVar, invokeCommand);
    }
    if (rangePtr->flags & NEVER_SET) {
	rangePtr->flags &= ~NEVER_SET;
    } else if (rangePtr->max_value == value) {
	return;
    }
    rangePtr->max_value = value;
    if (invokeCommand) {
	rangePtr->flags |= INVOKE_COMMAND;
    }
    EventuallyRedrawRange(rangePtr, REDRAW_SLIDER);

    if (setVar && (rangePtr->max_varName != NULL)) {
	sprintf(string, rangePtr->format, rangePtr->max_value);
	rangePtr->flags |= SETTING_VAR;
	Tcl_SetVar(rangePtr->interp, rangePtr->max_varName, string,
	       TCL_GLOBAL_ONLY);
	rangePtr->flags &= ~SETTING_VAR;
    }
}

/*
 *--------------------------------------------------------------
 *
 * EventuallyRedrawRange --
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

static void
EventuallyRedrawRange(rangePtr, what)
    register Range *rangePtr;	/* Information about widget. */
    int what;			/* What to redraw:  REDRAW_SLIDER
				 * or REDRAW_ALL. */
{
    if ((what == 0) || (rangePtr->tkwin == NULL)
	    || !Tk_IsMapped(rangePtr->tkwin)) {
	return;
    }
    if ((rangePtr->flags & REDRAW_ALL) == 0) {
	Tk_DoWhenIdle(DisplayRange, (ClientData) rangePtr);
    }
    rangePtr->flags |= what;
}

/*
 *--------------------------------------------------------------
 *
 * RoundToResolution --
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

static double
RoundToResolution(rangePtr, value)
    Range *rangePtr;		/* Information about range widget. */
    double value;		/* Value to round. */
{
    double rem;

    rem = fmod(value, rangePtr->resolution);
    if (rem < 0) {
	rem = rangePtr->resolution + rem;
    }
    value -= rem;
    if (rem >= rangePtr->resolution/2) {
	value += rangePtr->resolution;
    }
    return value;
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
    register Range *rangePtr = (Range *) clientData;
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
	    SetRangeMinValue(rangePtr, rangePtr->min_value, 1, 0);
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
    stringValue = Tcl_GetVar2(interp, name1, name2, flags & TCL_GLOBAL_ONLY);
    if (stringValue != NULL) {
	value = strtod(stringValue, &end);
	if ((end == stringValue) || (*end != 0)) {
	    result = "can't assign non-numeric value to range variable";
	} else {
	    rangePtr->min_value = value;
	}

	/*
	 * This code is a bit tricky because it sets the range's value before
	 * calling SetRangeMinValue.  This way, SetRangeMinValue won't bother to
	 * set the variable again or to invoke the -command.  However, it
	 * also won't redisplay the range, so we have to ask for that
	 * explicitly.
	 */

	SetRangeMinValue(rangePtr, rangePtr->min_value, 1, 0);
	EventuallyRedrawRange(rangePtr, REDRAW_SLIDER);
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
    register Range *rangePtr = (Range *) clientData;
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
	    SetRangeMaxValue(rangePtr, rangePtr->max_value, 1, 0);
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
    stringValue = Tcl_GetVar2(interp, name1, name2, flags & TCL_GLOBAL_ONLY);
    if (stringValue != NULL) {
	value = strtod(stringValue, &end);
	if ((end == stringValue) || (*end != 0)) {
	    result = "can't assign non-numeric value to range variable";
	} else {
	    rangePtr->max_value = value;
	}

	/*
	 * This code is a bit tricky because it sets the range's value before
	 * calling SetRangeMaxValue.  This way, SetRangeMaxValue won't bother to
	 * set the variable again or to invoke the -command.  However, it
	 * also won't redisplay the range, so we have to ask for that
	 * explicitly.
	 */

	SetRangeMaxValue(rangePtr, rangePtr->max_value, 1, 0);
	EventuallyRedrawRange(rangePtr, REDRAW_SLIDER);
    }

    return result;
}

