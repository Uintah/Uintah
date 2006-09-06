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
 * tkRange.h --
 *
 *	Declarations of types and functions used to implement
 *	the range widget.
 *
 * Copyright (c) 1996 by Sun Microsystems, Inc.
 *
 * See the file "license.terms" for information on usage and redistribution
 * of this file, and for a DISCLAIMER OF ALL WARRANTIES.
 *
 * Originally based on the tkScale widget, and then totally modified by...
 * David Weinstein 
 * January 1999
 * Copyright SCI
 */

#ifndef _TKRANGE
#define _TKRANGE

#ifndef _TK
#  include <tk.h>
#endif


# undef TCL_STORAGE_CLASS
# define TCL_STORAGE_CLASS DLLEXPORT

/*
 * Legal values for the "orient" field of TkRange records.
 */

enum orient {
    ORIENT_HORIZONTAL, ORIENT_VERTICAL
};

/*
 * Legal values for the "state" field of TkRange records.
 */

enum state {
    STATE_ACTIVE, STATE_DISABLED, STATE_NORMAL
};

/*
 * A data structure of the following type is kept for each range
 * widget managed by this file:
 */

typedef struct TkRange {
    Tk_Window tkwin;		/* Window that embodies the range.  NULL
				 * means that the window has been destroyed
				 * but the data structures haven't yet been
				 * cleaned up.*/
    Display *display;		/* Display containing widget.  Used, among
				 * other things, so that resources can be
				 * freed even after tkwin has gone away. */
    Tcl_Interp *interp;		/* Interpreter associated with range. */
    Tcl_Command widgetCmd;	/* Token for range's widget command. */
    Tk_OptionTable optionTable;	/* Table that defines configuration options
				 * available for this widget. */
    enum orient orient;		/* Orientation for window (vertical or
				 * horizontal). */
    int width;			/* Desired narrow dimension of range,
				 * in pixels. */
    int length;			/* Desired long dimension of range,
				 * in pixels. */
    double minvalue;		/* Current value of range. */
    double maxvalue;		/* Current value of range. */
    Tcl_Obj *minVarNamePtr;	/* Name of variable or NULL.
				 * If non-NULL, range's value tracks
				 * the contents of this variable and
				 * vice versa. */
    Tcl_Obj *maxVarNamePtr;	/* Name of variable or NULL.
				 * If non-NULL, range's value tracks
				 * the contents of this variable and
				 * vice versa. */
    double fromValue;		/* Value corresponding to left or top of
				 * range. */
    double toValue;		/* Value corresponding to right or bottom
				 * of range. */
    double tickInterval;	/* Distance between tick marks;
				 * 0 means don't display any tick marks. */
    double resolution;		/* If > 0, all values are rounded to an
				 * even multiple of this value. */
    int digits;			/* Number of significant digits to print
				 * in values.  0 means we get to choose the
				 * number based on resolution and/or the
				 * range of the range. */
    char format[10];		/* Sprintf conversion specifier computed from
				 * digits and other information. */
    double bigIncrement;	/* Amount to use for large increments to
				 * range value.	 (0 means we pick a value). */
    char *command;		/* Command prefix to use when invoking Tcl
				 * commands because the range value changed.
				 * NULL means don't invoke commands. */
    int repeatDelay;		/* How long to wait before auto-repeating
				 * on scrolling actions (in ms). */
    int repeatInterval;		/* Interval between autorepeats (in ms). */
    char *label;		/* Label to display above or to right of
				 * range;  NULL means don't display a label. */
    int labelLength;		/* Number of non-NULL chars. in label. */
    enum state state;		/* Values are active, normal, or disabled.
				 * Value of range cannot be changed when 
				 * disabled. */
    int nonZero;                /* Should min and max be distinct */

    /*
     * Information used when displaying widget:
     */

    int borderWidth;		/* Width of 3-D border around window. */
    Tk_3DBorder bgBorder;	/* Used for drawing slider and other
				 * background areas. */
    Tk_3DBorder activeBorder;	/* For drawing the slider when active. */
    int sliderRelief;		/* Is slider to be drawn raised, sunken, 
				 * etc. */
    XColor *troughColorPtr;	/* Color for drawing trough. */
    XColor *rangeColorPtr;	/* Color for drawing trough. */
    GC troughGC;		/* For drawing trough. */
    GC copyGC;			/* Used for copying from pixmap onto screen. */
    Tk_Font tkfont;		/* Information about text font, or NULL. */
    XColor *textColorPtr;	/* Color for drawing text. */
    GC textGC;			/* GC for drawing text in normal mode. */
    GC rangeGC;			/* GC for drawing text in normal mode. */
    int relief;			/* Indicates whether window as a whole is
				 * raised, sunken, or flat. */
    int highlightWidth;		/* Width in pixels of highlight to draw
				 * around widget when it has the focus.
				 * <= 0 means don't draw a highlight. */
    Tk_3DBorder highlightBorder;/* Value of -highlightbackground option:
				 * specifies background with which to draw 3-D
				 * default ring and focus highlight area when
				 * highlight is off. */
    XColor *highlightColorPtr;	/* Color for drawing traversal highlight. */
    int inset;			/* Total width of all borders, including
				 * traversal highlight and 3-D border.
				 * Indicates how much interior stuff must
				 * be offset from outside edges to leave
				 * room for borders. */
    int sliderLength;		/* Length of slider, measured in pixels along
				 * long dimension of range. */
    int showValue;		/* Non-zero means to display the range value
				 * below or to the left of the slider;	zero
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

    int fontHeight;		/* Height of range font. */
    Tk_Cursor cursor;		/* Current cursor for window, or None. */
    Tcl_Obj *takeFocusPtr;	/* Value of -takefocus option;	not used in
				 * the C code, but used by keyboard traversal
				 * scripts.  May be NULL. */
    int flags;			/* Various flags;  see below for
				 * definitions. */
} TkRange;

/*
 * Flag bits for ranges:
 *
 * REDRAW_SLIDER -		1 means slider (and numerical readout) need
 *				to be redrawn.
 * REDRAW_OTHER -		1 means other stuff besides slider and value
 *				need to be redrawn.
 * REDRAW_ALL -			1 means the entire widget needs to be redrawn.
 * REDRAW_PENDING -		1 means any sort of redraw is pending
 * ACTIVE -			1 means the widget is active (the mouse is
 *				in its window).
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
 * RANGE_DELETED -		1 means the range widget is being deleted
 */

#define REDRAW_SLIDER		(1<<0)
#define REDRAW_OTHER		(1<<1)
#define REDRAW_ALL		(REDRAW_OTHER|REDRAW_SLIDER)
#define REDRAW_PENDING		(1<<2)
#define ACTIVE			(1<<3)
#define INVOKE_COMMAND		(1<<4)
#define SETTING_VAR		(1<<5)
#define NEVER_SET		(1<<6)
#define GOT_FOCUS		(1<<7)
#define RANGE_DELETED		(1<<8)

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
 * Declaration of procedures used in the implementation of the range
 * widget. 
 */

EXTERN void		TkEventuallyRedrawRange _ANSI_ARGS_((TkRange *rangePtr,
			    int what));
EXTERN double		TkRangeRoundToResolution _ANSI_ARGS_((TkRange *rangePtr,
							 double value));
EXTERN TkRange *	TkpCreateRange _ANSI_ARGS_((Tk_Window tkwin));
EXTERN void		TkpDestroyRange _ANSI_ARGS_((TkRange *rangePtr));
EXTERN void		TkpDisplayRange _ANSI_ARGS_((ClientData clientData));
EXTERN int		TkpRangeElement _ANSI_ARGS_((TkRange *rangePtr,
			     int x, int y));
EXTERN void		TkRangeSetMinValue _ANSI_ARGS_((TkRange *rangePtr,
			    double value, int setVar, int invokeCommand));
EXTERN void		TkRangeSetMaxValue _ANSI_ARGS_((TkRange *rangePtr,
			    double value, int setVar, int invokeCommand));
EXTERN double		TkRangePixelToValue _ANSI_ARGS_((TkRange *rangePtr, 
			    int x, int y));
EXTERN int		TkRangeValueToPixel _ANSI_ARGS_((TkRange *rangePtr,
			    double value));

# undef TCL_STORAGE_CLASS
# define TCL_STORAGE_CLASS DLLIMPORT

/* These are in tk/unix/tkUnixDefault.h for all other widgets. */
#define DEF_RANGE_ACTIVE_BG_COLOR	ACTIVE_BG
#define DEF_RANGE_ACTIVE_BG_MONO	BLACK
#define DEF_RANGE_BG_COLOR		NORMAL_BG
#define DEF_RANGE_BG_MONO		WHITE
#define DEF_RANGE_BIG_INCREMENT		"0"
#define DEF_RANGE_BORDER_WIDTH		"2"
#define DEF_RANGE_COMMAND		""
#define DEF_RANGE_CURSOR		""
#define DEF_RANGE_DIGITS		"0"
#define DEF_RANGE_FONT			"Helvetica -12 bold"
#define DEF_RANGE_FG_COLOR		BLACK
#define DEF_RANGE_FG_MONO		BLACK
#define DEF_RANGE_FROM			"0"
#define DEF_RANGE_HIGHLIGHT_BG_COLOR	DEF_RANGE_BG_COLOR
#define DEF_RANGE_HIGHLIGHT_BG_MONO	DEF_RANGE_BG_MONO
#define DEF_RANGE_HIGHLIGHT		BLACK
#define DEF_RANGE_HIGHLIGHT_WIDTH	"1"
#define DEF_RANGE_LABEL			""
#define DEF_RANGE_LENGTH		"100"
#define DEF_RANGE_ORIENT		"vertical"
#define DEF_RANGE_RELIEF		"flat"
#define DEF_RANGE_REPEAT_DELAY	        "300"
#define DEF_RANGE_REPEAT_INTERVAL	"100"
#define DEF_RANGE_RESOLUTION		"1"
#define DEF_RANGE_TROUGH_COLOR		TROUGH
#define DEF_RANGE_TROUGH_MONO		WHITE
#define DEF_RANGE_RANGE_COLOR		"#CC0033"
#define DEF_RANGE_RANGE_MONO		WHITE
#define DEF_RANGE_SHOW_VALUE		"1"
#define DEF_RANGE_NON_ZERO		"0"
#define DEF_RANGE_SLIDER_LENGTH		"30"
#define DEF_RANGE_SLIDER_RELIEF		"raised"
#define DEF_RANGE_STATE			"normal"
#define DEF_RANGE_TAKE_FOCUS		(char *) NULL
#define DEF_RANGE_TICK_INTERVAL		"0"
#define DEF_RANGE_TO			"100"
#define DEF_RANGE_VARIABLE		""
#define DEF_RANGE_WIDTH			"15"

#endif /* _TKRANGE */
