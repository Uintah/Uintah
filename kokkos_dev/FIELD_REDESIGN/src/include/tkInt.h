/*
 * tkInt.h --
 *
 *	Declarations for things used internally by the Tk
 *	procedures but not exported outside the module.
 *
 * Copyright (c) 1990-1994 The Regents of the University of California.
 * Copyright (c) 1994-1997 Sun Microsystems, Inc.
 * Copyright (c) 1998 by Scriptics Corporation.
 *
 * See the file "license.terms" for information on usage and redistribution
 * of this file, and for a DISCLAIMER OF ALL WARRANTIES.
 *
 * RCS: $Id$ 
 */

#ifndef _TKINT
#define _TKINT

#ifndef _TK
#include "tk.h"
#endif
#ifndef _TCL
#include "tcl.h"
#endif
#ifndef _TKPORT
#include <tkPort.h>
#endif

#ifdef BUILD_tk
# undef TCL_STORAGE_CLASS
# define TCL_STORAGE_CLASS DLLEXPORT
#endif

/*
 * Opaque type declarations:
 */

typedef struct TkColormap TkColormap;
typedef struct TkGrabEvent TkGrabEvent;
typedef struct Tk_PostscriptInfo Tk_PostscriptInfo;
typedef struct TkpCursor_ *TkpCursor;
typedef struct TkRegion_ *TkRegion;
typedef struct TkStressedCmap TkStressedCmap;
typedef struct TkBindInfo_ *TkBindInfo;

/*
 * Procedure types.
 */

typedef int (TkBindEvalProc) _ANSI_ARGS_((ClientData clientData,
	Tcl_Interp *interp, XEvent *eventPtr, Tk_Window tkwin,
	KeySym keySym));
typedef void (TkBindFreeProc) _ANSI_ARGS_((ClientData clientData));
typedef Window (TkClassCreateProc) _ANSI_ARGS_((Tk_Window tkwin,
	Window parent, ClientData instanceData));
typedef void (TkClassGeometryProc) _ANSI_ARGS_((ClientData instanceData));
typedef void (TkClassModalProc) _ANSI_ARGS_((Tk_Window tkwin,
	XEvent *eventPtr));


/*
 * Widget class procedures used to implement platform specific widget
 * behavior.
 */

typedef struct TkClassProcs {
    TkClassCreateProc *createProc;
				/* Procedure to invoke when the
                                   platform-dependent window needs to be
                                   created. */
    TkClassGeometryProc *geometryProc;
				/* Procedure to invoke when the geometry of a
				   window needs to be recalculated as a result
				   of some change in the system. */
    TkClassModalProc *modalProc;
				/* Procedure to invoke after all bindings on a
				   widget have been triggered in order to
				   handle a modal loop. */
} TkClassProcs;

/*
 * One of the following structures is maintained for each cursor in
 * use in the system.  This structure is used by tkCursor.c and the
 * various system specific cursor files.
 */

typedef struct TkCursor {
    Tk_Cursor cursor;		/* System specific identifier for cursor. */
    int refCount;		/* Number of active uses of cursor. */
    Tcl_HashTable *otherTable;	/* Second table (other than idTable) used
				 * to index this entry. */
    Tcl_HashEntry *hashPtr;	/* Entry in otherTable for this structure
				 * (needed when deleting). */
} TkCursor;

/*
 * One of the following structures is maintained for each display
 * containing a window managed by Tk:
 */

typedef struct TkDisplay {
    Display *display;		/* Xlib's info about display. */
    struct TkDisplay *nextPtr;	/* Next in list of all displays. */
    char *name;			/* Name of display (with any screen
				 * identifier removed).  Malloc-ed. */
    Time lastEventTime;		/* Time of last event received for this
				 * display. */

    /*
     * Information used primarily by tkBind.c:
     */

    int bindInfoStale;		/* Non-zero means the variables in this
				 * part of the structure are potentially
				 * incorrect and should be recomputed. */
    unsigned int modeModMask;	/* Has one bit set to indicate the modifier
				 * corresponding to "mode shift".  If no
				 * such modifier, than this is zero. */
    unsigned int metaModMask;	/* Has one bit set to indicate the modifier
				 * corresponding to the "Meta" key.  If no
				 * such modifier, then this is zero. */
    unsigned int altModMask;	/* Has one bit set to indicate the modifier
				 * corresponding to the "Meta" key.  If no
				 * such modifier, then this is zero. */
    enum {LU_IGNORE, LU_CAPS, LU_SHIFT} lockUsage;
				/* Indicates how to interpret lock modifier. */
    int numModKeyCodes;		/* Number of entries in modKeyCodes array
				 * below. */
    KeyCode *modKeyCodes;	/* Pointer to an array giving keycodes for
				 * all of the keys that have modifiers
				 * associated with them.  Malloc'ed, but
				 * may be NULL. */

    /*
     * Information used by tkError.c only:
     */

    struct TkErrorHandler *errorPtr;
				/* First in list of error handlers
				 * for this display.  NULL means
				 * no handlers exist at present. */
    int deleteCount;		/* Counts # of handlers deleted since
				 * last time inactive handlers were
				 * garbage-collected.  When this number
				 * gets big, handlers get cleaned up. */

    /*
     * Information used by tkSend.c only:
     */

    Tk_Window commTkwin;	/* Window used for communication
				 * between interpreters during "send"
				 * commands.  NULL means send info hasn't
				 * been initialized yet. */
    Atom commProperty;		/* X's name for comm property. */
    Atom registryProperty;	/* X's name for property containing
				 * registry of interpreter names. */
    Atom appNameProperty;	/* X's name for property used to hold the
				 * application name on each comm window. */

    /*
     * Information used by tkSelect.c and tkClipboard.c only:
     */

    struct TkSelectionInfo *selectionInfoPtr;
				/* First in list of selection information
				 * records.  Each entry contains information
				 * about the current owner of a particular
				 * selection on this display. */
    Atom multipleAtom;		/* Atom for MULTIPLE.  None means
				 * selection stuff isn't initialized. */
    Atom incrAtom;		/* Atom for INCR. */
    Atom targetsAtom;		/* Atom for TARGETS. */
    Atom timestampAtom;		/* Atom for TIMESTAMP. */
    Atom textAtom;		/* Atom for TEXT. */
    Atom compoundTextAtom;	/* Atom for COMPOUND_TEXT. */
    Atom applicationAtom;	/* Atom for TK_APPLICATION. */
    Atom windowAtom;		/* Atom for TK_WINDOW. */
    Atom clipboardAtom;		/* Atom for CLIPBOARD. */

    Tk_Window clipWindow;	/* Window used for clipboard ownership and to
				 * retrieve selections between processes. NULL
				 * means clipboard info hasn't been
				 * initialized. */
    int clipboardActive;	/* 1 means we currently own the clipboard
				 * selection, 0 means we don't. */
    struct TkMainInfo *clipboardAppPtr;
				/* Last application that owned clipboard. */
    struct TkClipboardTarget *clipTargetPtr;
				/* First in list of clipboard type information
				 * records.  Each entry contains information
				 * about the buffers for a given selection
				 * target. */

    /*
     * Information used by tkAtom.c only:
     */

    int atomInit;		/* 0 means stuff below hasn't been
				 * initialized yet. */
    Tcl_HashTable nameTable;	/* Maps from names to Atom's. */
    Tcl_HashTable atomTable;	/* Maps from Atom's back to names. */

    /*
     * Information used by tkCursor.c only:
     */

    Font cursorFont;		/* Font to use for standard cursors.
				 * None means font not loaded yet. */

    /*
     * Information used by tkGrab.c only:
     */

    struct TkWindow *grabWinPtr;
				/* Window in which the pointer is currently
				 * grabbed, or NULL if none. */
    struct TkWindow *eventualGrabWinPtr;
				/* Value that grabWinPtr will have once the
				 * grab event queue (below) has been
				 * completely emptied. */
    struct TkWindow *buttonWinPtr;
				/* Window in which first mouse button was
				 * pressed while grab was in effect, or NULL
				 * if no such press in effect. */
    struct TkWindow *serverWinPtr;
				/* If no application contains the pointer then
				 * this is NULL.  Otherwise it contains the
				 * last window for which we've gotten an
				 * Enter or Leave event from the server (i.e.
				 * the last window known to have contained
				 * the pointer).  Doesn't reflect events
				 * that were synthesized in tkGrab.c. */
    TkGrabEvent *firstGrabEventPtr;
				/* First in list of enter/leave events
				 * synthesized by grab code.  These events
				 * must be processed in order before any other
				 * events are processed.  NULL means no such
				 * events. */
    TkGrabEvent *lastGrabEventPtr;
				/* Last in list of synthesized events, or NULL
				 * if list is empty. */
    int grabFlags;		/* Miscellaneous flag values.  See definitions
				 * in tkGrab.c. */

    /*
     * Information used by tkXId.c only:
     */

    struct TkIdStack *idStackPtr;
				/* First in list of chunks of free resource
				 * identifiers, or NULL if there are no free
				 * resources. */
    XID (*defaultAllocProc) _ANSI_ARGS_((Display *display));
				/* Default resource allocator for display. */
    struct TkIdStack *windowStackPtr;
				/* First in list of chunks of window
				 * identifers that can't be reused right
				 * now. */
    int idCleanupScheduled;	/* 1 means a call to WindowIdCleanup has
				 * already been scheduled, 0 means it
				 * hasn't. */

    /*
     * Information maintained by tkWindow.c for use later on by tkXId.c:
     */


    int destroyCount;		/* Number of Tk_DestroyWindow operations
				 * in progress. */
    unsigned long lastDestroyRequest;
				/* Id of most recent XDestroyWindow request;
				 * can re-use ids in windowStackPtr when
				 * server has seen this request and event
				 * queue is empty. */

    /*
     * Information used by tkVisual.c only:
     */

    TkColormap *cmapPtr;	/* First in list of all non-default colormaps
				 * allocated for this display. */

    /*
     * Information used by tkFocus.c only:
     */

    struct TkWindow *implicitWinPtr;
				/* If the focus arrived at a toplevel window
				 * implicitly via an Enter event (rather
				 * than via a FocusIn event), this points
				 * to the toplevel window.  Otherwise it is
				 * NULL. */
    struct TkWindow *focusPtr;	/* Points to the window on this display that
				 * should be receiving keyboard events.  When
				 * multiple applications on the display have
				 * the focus, this will refer to the
				 * innermost window in the innermost
				 * application.  This information isn't used
				 * under Unix or Windows, but it's needed on
				 * the Macintosh. */

    /*
     * Used by tkColor.c only:
     */

    TkStressedCmap *stressPtr;	/* First in list of colormaps that have
				 * filled up, so we have to pick an
				 * approximate color. */

    /*
     * Used by tkEvent.c only:
     */

    struct TkWindowEvent *delayedMotionPtr;
				/* Points to a malloc-ed motion event
				 * whose processing has been delayed in
				 * the hopes that another motion event
				 * will come along right away and we can
				 * merge the two of them together.  NULL
				 * means that there is no delayed motion
				 * event. */

    /*
     * Miscellaneous information:
     */

#ifdef TK_USE_INPUT_METHODS
    XIM inputMethod;		/* Input method for this display */
#endif /* TK_USE_INPUT_METHODS */
    Tcl_HashTable winTable;	/* Maps from X window ids to TkWindow ptrs. */

    int refCount;		/* Reference count of how many Tk applications
                                 * are using this display. Used to clean up
                                 * the display when we no longer have any
                                 * Tk applications using it.
                                 */
} TkDisplay;

/*
 * One of the following structures exists for each error handler
 * created by a call to Tk_CreateErrorHandler.  The structure
 * is managed by tkError.c.
 */

typedef struct TkErrorHandler {
    TkDisplay *dispPtr;		/* Display to which handler applies. */
    unsigned long firstRequest;	/* Only errors with serial numbers
				 * >= to this are considered. */
    unsigned long lastRequest;	/* Only errors with serial numbers
				 * <= to this are considered.  This
				 * field is filled in when XUnhandle
				 * is called.  -1 means XUnhandle
				 * hasn't been called yet. */
    int error;			/* Consider only errors with this
				 * error_code (-1 means consider
				 * all errors). */
    int request;		/* Consider only errors with this
				 * major request code (-1 means
				 * consider all major codes). */
    int minorCode;		/* Consider only errors with this
				 * minor request code (-1 means
				 * consider all minor codes). */
    Tk_ErrorProc *errorProc;	/* Procedure to invoke when a matching
				 * error occurs.  NULL means just ignore
				 * errors. */
    ClientData clientData;	/* Arbitrary value to pass to
				 * errorProc. */
    struct TkErrorHandler *nextPtr;
				/* Pointer to next older handler for
				 * this display, or NULL for end of
				 * list. */
} TkErrorHandler;

/*
 * One of the following structures exists for each event handler
 * created by calling Tk_CreateEventHandler.  This information
 * is used by tkEvent.c only.
 */

typedef struct TkEventHandler {
    unsigned long mask;		/* Events for which to invoke
				 * proc. */
    Tk_EventProc *proc;		/* Procedure to invoke when an event
				 * in mask occurs. */
    ClientData clientData;	/* Argument to pass to proc. */
    struct TkEventHandler *nextPtr;
				/* Next in list of handlers
				 * associated with window (NULL means
				 * end of list). */
} TkEventHandler;

/*
 * Tk keeps one of the following data structures for each main
 * window (created by a call to Tk_CreateMainWindow).  It stores
 * information that is shared by all of the windows associated
 * with a particular main window.
 */

typedef struct TkMainInfo {
    int refCount;		/* Number of windows whose "mainPtr" fields
				 * point here.  When this becomes zero, can
				 * free up the structure (the reference
				 * count is zero because windows can get
				 * deleted in almost any order;  the main
				 * window isn't necessarily the last one
				 * deleted). */
    struct TkWindow *winPtr;	/* Pointer to main window. */
    Tcl_Interp *interp;		/* Interpreter associated with application. */
    Tcl_HashTable nameTable;	/* Hash table mapping path names to TkWindow
				 * structs for all windows related to this
				 * main window.  Managed by tkWindow.c. */
    Tk_BindingTable bindingTable;
				/* Used in conjunction with "bind" command
				 * to bind events to Tcl commands. */
    TkBindInfo bindInfo;	/* Information used by tkBind.c on a per
				 * interpreter basis. */
    struct TkFontInfo *fontInfoPtr;
				/* Hold named font tables.  Used only by
				 * tkFont.c. */

    /*
     * Information used only by tkFocus.c and tk*Embed.c:
     */

    struct TkToplevelFocusInfo *tlFocusPtr;
				/* First in list of records containing focus
				 * information for each top-level in the
				 * application.  Used only by tkFocus.c. */
    struct TkDisplayFocusInfo *displayFocusPtr;
				/* First in list of records containing focus
				 * information for each display that this
				 * application has ever used.  Used only
				 * by tkFocus.c. */

    struct ElArray *optionRootPtr;
				/* Top level of option hierarchy for this
				 * main window.  NULL means uninitialized.
				 * Managed by tkOption.c. */
    Tcl_HashTable imageTable;	/* Maps from image names to Tk_ImageMaster
				 * structures.  Managed by tkImage.c. */
    int strictMotif;		/* This is linked to the tk_strictMotif
				 * global variable. */
    struct TkMainInfo *nextPtr;	/* Next in list of all main windows managed by
				 * this process. */
} TkMainInfo;

/*
 * Tk keeps the following data structure for each of it's builtin
 * bitmaps.  This structure is only used by tkBitmap.c and other
 * platform specific bitmap files.
 */

typedef struct {
    char *source;		/* Bits for bitmap. */
    int width, height;		/* Dimensions of bitmap. */
    int native;			/* 0 means generic (X style) bitmap,
    				 * 1 means native style bitmap. */
} TkPredefBitmap;

/*
 * Tk keeps one of the following structures for each window.
 * Some of the information (like size and location) is a shadow
 * of information managed by the X server, and some is special
 * information used here, such as event and geometry management
 * information.  This information is (mostly) managed by tkWindow.c.
 * WARNING: the declaration below must be kept consistent with the
 * Tk_FakeWin structure in tk.h.  If you change one, be sure to
 * change the other!!
 */

typedef struct TkWindow {

    /*
     * Structural information:
     */

    Display *display;		/* Display containing window. */
    TkDisplay *dispPtr;		/* Tk's information about display
				 * for window. */
    int screenNum;		/* Index of screen for window, among all
				 * those for dispPtr. */
    Visual *visual;		/* Visual to use for window.  If not default,
				 * MUST be set before X window is created. */
    int depth;			/* Number of bits/pixel. */
    Window window;		/* X's id for window.   NULL means window
				 * hasn't actually been created yet, or it's
				 * been deleted. */
    struct TkWindow *childList;	/* First in list of child windows,
				 * or NULL if no children.  List is in
				 * stacking order, lowest window first.*/
    struct TkWindow *lastChildPtr;
				/* Last in list of child windows (highest
				 * in stacking order), or NULL if no
				 * children. */
    struct TkWindow *parentPtr;	/* Pointer to parent window (logical
				 * parent, not necessarily X parent).  NULL
				 * means either this is the main window, or
				 * the window's parent has already been
				 * deleted. */
    struct TkWindow *nextPtr;	/* Next higher sibling (in stacking order)
				 * in list of children with same parent.  NULL
				 * means end of list. */
    TkMainInfo *mainPtr;	/* Information shared by all windows
				 * associated with a particular main
				 * window.  NULL means this window is
				 * a rogue that isn't associated with
				 * any application (at present, this
				 * only happens for the dummy windows
				 * used for "send" communication).  */

    /*
     * Name and type information for the window:
     */

    char *pathName;		/* Path name of window (concatenation
				 * of all names between this window and
				 * its top-level ancestor).  This is a
				 * pointer into an entry in
				 * mainPtr->nameTable.  NULL means that
				 * the window hasn't been completely
				 * created yet. */
    Tk_Uid nameUid;		/* Name of the window within its parent
				 * (unique within the parent). */
    Tk_Uid classUid;		/* Class of the window.  NULL means window
				 * hasn't been given a class yet. */

    /*
     * Geometry and other attributes of window.  This information
     * may not be updated on the server immediately;  stuff that
     * hasn't been reflected in the server yet is called "dirty".
     * At present, information can be dirty only if the window
     * hasn't yet been created.
     */

    XWindowChanges changes;	/* Geometry and other info about
				 * window. */
    unsigned int dirtyChanges;	/* Bits indicate fields of "changes"
				 * that are dirty. */
    XSetWindowAttributes atts;	/* Current attributes of window. */
    unsigned long dirtyAtts;	/* Bits indicate fields of "atts"
				 * that are dirty. */

    unsigned int flags;		/* Various flag values:  these are all
				 * defined in tk.h (confusing, but they're
				 * needed there for some query macros). */

    /*
     * Information kept by the event manager (tkEvent.c):
     */

    TkEventHandler *handlerList;/* First in list of event handlers
				 * declared for this window, or
				 * NULL if none. */
#ifdef TK_USE_INPUT_METHODS
    XIC inputContext;		/* Input context (for input methods). */
#endif /* TK_USE_INPUT_METHODS */

    /*
     * Information used for event bindings (see "bind" and "bindtags"
     * commands in tkCmds.c):
     */

    ClientData *tagPtr;		/* Points to array of tags used for bindings
				 * on this window.  Each tag is a Tk_Uid.
				 * Malloc'ed.  NULL means no tags. */
    int numTags;		/* Number of tags at *tagPtr. */

    /*
     * Information used by tkOption.c to manage options for the
     * window.
     */

    int optionLevel;		/* -1 means no option information is
				 * currently cached for this window.
				 * Otherwise this gives the level in
				 * the option stack at which info is
				 * cached. */
    /*
     * Information used by tkSelect.c to manage the selection.
     */

    struct TkSelHandler *selHandlerList;
				/* First in list of handlers for
				 * returning the selection in various
				 * forms. */

    /*
     * Information used by tkGeometry.c for geometry management.
     */

    Tk_GeomMgr *geomMgrPtr;	/* Information about geometry manager for
				 * this window. */
    ClientData geomData;	/* Argument for geometry manager procedures. */
    int reqWidth, reqHeight;	/* Arguments from last call to
				 * Tk_GeometryRequest, or 0's if
				 * Tk_GeometryRequest hasn't been
				 * called. */
    int internalBorderWidth;	/* Width of internal border of window
				 * (0 means no internal border).  Geometry
				 * managers should not normally place children
				 * on top of the border. */

    /*
     * Information maintained by tkWm.c for window manager communication.
     */

    struct TkWmInfo *wmInfoPtr;	/* For top-level windows (and also
				 * for special Unix menubar and wrapper
				 * windows), points to structure with
				 * wm-related info (see tkWm.c).  For
				 * other windows, this is NULL. */

    /*
     * Information used by widget classes.
     */

    TkClassProcs *classProcsPtr;
    ClientData instanceData;

    /*
     * Platform specific information private to each port.
     */

    struct TkWindowPrivate *privatePtr;
} TkWindow;

/*
 * The following structure is used as a two way map between integers
 * and strings, usually to map between an internal C representation
 * and the strings used in Tcl.
 */

typedef struct TkStateMap {
    int numKey;			/* Integer representation of a value. */
    char *strKey;		/* String representation of a value. */
} TkStateMap;

/*
 * This structure is used by the Mac and Window porting layers as
 * the internal representation of a clip_mask in a GC.
 */

typedef struct TkpClipMask {
    int type;			/* One of TKP_CLIP_PIXMAP or TKP_CLIP_REGION */
    union {
	Pixmap pixmap;
	TkRegion region;
    } value;
} TkpClipMask;

#define TKP_CLIP_PIXMAP 0
#define TKP_CLIP_REGION 1

/*
 * Pointer to first entry in list of all displays currently known.
 */

extern TkDisplay *tkDisplayList;

/*
 * Return values from TkGrabState:
 */

#define TK_GRAB_NONE		0
#define TK_GRAB_IN_TREE		1
#define TK_GRAB_ANCESTOR	2
#define TK_GRAB_EXCLUDED	3

/*
 * The macro below is used to modify a "char" value (e.g. by casting
 * it to an unsigned character) so that it can be used safely with
 * macros such as isspace.
 */

#define UCHAR(c) ((unsigned char) (c))

/*
 * The following symbol is used in the mode field of FocusIn events
 * generated by an embedded application to request the input focus from
 * its container.
 */

#define EMBEDDED_APP_WANTS_FOCUS (NotifyNormal + 20)

/*
 * Miscellaneous variables shared among Tk modules but not exported
 * to the outside world:
 */

extern Tk_Uid			tkActiveUid;
extern Tk_ImageType		tkBitmapImageType;
extern Tk_Uid			tkDisabledUid;
extern Tk_PhotoImageFormat	tkImgFmtGIF;
extern void			(*tkHandleEventProc) _ANSI_ARGS_((
    				    XEvent* eventPtr));
extern Tk_PhotoImageFormat	tkImgFmtPPM;
extern TkMainInfo		*tkMainWindowList;
extern Tk_Uid			tkNormalUid;
extern Tk_ImageType		tkPhotoImageType;
extern Tcl_HashTable		tkPredefBitmapTable;
extern int			tkSendSerial;

/*
 * Internal procedures shared among Tk modules but not exported
 * to the outside world:
 */

EXTERN char *		TkAlignImageData _ANSI_ARGS_((XImage *image,
			    int alignment, int bitOrder));
EXTERN TkWindow *	TkAllocWindow _ANSI_ARGS_((TkDisplay *dispPtr,
			    int screenNum, TkWindow *parentPtr));
EXTERN void		TkBezierPoints _ANSI_ARGS_((double control[],
			    int numSteps, double *coordPtr));
EXTERN void		TkBezierScreenPoints _ANSI_ARGS_((Tk_Canvas canvas,
			    double control[], int numSteps,
			    XPoint *xPointPtr));
EXTERN void		TkBindDeadWindow _ANSI_ARGS_((TkWindow *winPtr));
EXTERN void		TkBindEventProc _ANSI_ARGS_((TkWindow *winPtr,
			    XEvent *eventPtr));
EXTERN void		TkBindFree _ANSI_ARGS_((TkMainInfo *mainPtr));
EXTERN void		TkBindInit _ANSI_ARGS_((TkMainInfo *mainPtr));
EXTERN void		TkChangeEventWindow _ANSI_ARGS_((XEvent *eventPtr,
			    TkWindow *winPtr));
#ifndef TkClipBox
EXTERN void		TkClipBox _ANSI_ARGS_((TkRegion rgn,
			    XRectangle* rect_return));
#endif
EXTERN int		TkClipInit _ANSI_ARGS_((Tcl_Interp *interp,
			    TkDisplay *dispPtr));
EXTERN void		TkComputeAnchor _ANSI_ARGS_((Tk_Anchor anchor,
			    Tk_Window tkwin, int padX, int padY,
			    int innerWidth, int innerHeight, int *xPtr,
			    int *yPtr));
EXTERN int		TkCopyAndGlobalEval _ANSI_ARGS_((Tcl_Interp *interp,
			    char *script));
EXTERN unsigned long	TkCreateBindingProcedure _ANSI_ARGS_((
			    Tcl_Interp *interp, Tk_BindingTable bindingTable,
			    ClientData object, char *eventString,
			    TkBindEvalProc *evalProc, TkBindFreeProc *freeProc,
			    ClientData clientData));
EXTERN TkCursor *	TkCreateCursorFromData _ANSI_ARGS_((Tk_Window tkwin,
			    char *source, char *mask, int width, int height,
			    int xHot, int yHot, XColor fg, XColor bg));
EXTERN int		TkCreateFrame _ANSI_ARGS_((ClientData clientData,
			    Tcl_Interp *interp, int argc, char **argv,
			    int toplevel, char *appName));
EXTERN Tk_Window	TkCreateMainWindow _ANSI_ARGS_((Tcl_Interp *interp,
			    char *screenName, char *baseName));
#ifndef TkCreateRegion
EXTERN TkRegion		TkCreateRegion _ANSI_ARGS_((void));
#endif
EXTERN Time		TkCurrentTime _ANSI_ARGS_((TkDisplay *dispPtr));
EXTERN int		TkDeadAppCmd _ANSI_ARGS_((ClientData clientData,
			    Tcl_Interp *interp, int argc, char **argv));
EXTERN void		TkDeleteAllImages _ANSI_ARGS_((TkMainInfo *mainPtr));
#ifndef TkDestroyRegion
EXTERN void		TkDestroyRegion _ANSI_ARGS_((TkRegion rgn));
#endif
EXTERN void		TkDoConfigureNotify _ANSI_ARGS_((TkWindow *winPtr));
EXTERN void		TkDrawInsetFocusHighlight _ANSI_ARGS_((
			    Tk_Window tkwin, GC gc, int width,
			    Drawable drawable, int padding));
EXTERN void		TkEventDeadWindow _ANSI_ARGS_((TkWindow *winPtr));
EXTERN void		TkFillPolygon _ANSI_ARGS_((Tk_Canvas canvas,
			    double *coordPtr, int numPoints, Display *display,
			    Drawable drawable, GC gc, GC outlineGC));
EXTERN int		TkFindStateNum _ANSI_ARGS_((Tcl_Interp *interp,
			    CONST char *option, CONST TkStateMap *mapPtr,
			    CONST char *strKey));
EXTERN char *		TkFindStateString _ANSI_ARGS_((
			    CONST TkStateMap *mapPtr, int numKey));
EXTERN void		TkFocusDeadWindow _ANSI_ARGS_((TkWindow *winPtr));
EXTERN int		TkFocusFilterEvent _ANSI_ARGS_((TkWindow *winPtr,
			    XEvent *eventPtr));
EXTERN TkWindow *	TkFocusKeyEvent _ANSI_ARGS_((TkWindow *winPtr,
			    XEvent *eventPtr));
EXTERN void		TkFontPkgInit _ANSI_ARGS_((TkMainInfo *mainPtr));
EXTERN void		TkFontPkgFree _ANSI_ARGS_((TkMainInfo *mainPtr));
EXTERN void		TkFreeBindingTags _ANSI_ARGS_((TkWindow *winPtr));
EXTERN void		TkFreeCursor _ANSI_ARGS_((TkCursor *cursorPtr));
EXTERN void		TkFreeWindowId _ANSI_ARGS_((TkDisplay *dispPtr,
			    Window w));
EXTERN void		TkGenerateActivateEvents _ANSI_ARGS_((
			    TkWindow *winPtr, int active));
EXTERN char *		TkGetBitmapData _ANSI_ARGS_((Tcl_Interp *interp,
			    char *string, char *fileName, int *widthPtr,
			    int *heightPtr, int *hotXPtr, int *hotYPtr));
EXTERN void		TkGetButtPoints _ANSI_ARGS_((double p1[], double p2[],
			    double width, int project, double m1[],
			    double m2[]));
EXTERN TkCursor *	TkGetCursorByName _ANSI_ARGS_((Tcl_Interp *interp,
			    Tk_Window tkwin, Tk_Uid string));
EXTERN char *		TkGetDefaultScreenName _ANSI_ARGS_((Tcl_Interp *interp,
			    char *screenName));
EXTERN TkDisplay *	TkGetDisplay _ANSI_ARGS_((Display *display));
EXTERN int		TkGetDisplayOf _ANSI_ARGS_((Tcl_Interp *interp,
			    int objc, Tcl_Obj *CONST objv[],
			    Tk_Window *tkwinPtr));
EXTERN TkWindow *	TkGetFocusWin _ANSI_ARGS_((TkWindow *winPtr));
EXTERN int		TkGetInterpNames _ANSI_ARGS_((Tcl_Interp *interp,
			    Tk_Window tkwin));
EXTERN int		TkGetMiterPoints _ANSI_ARGS_((double p1[], double p2[],
			    double p3[], double width, double m1[],
			    double m2[]));
EXTERN void		TkGetPointerCoords _ANSI_ARGS_((Tk_Window tkwin,
			    int *xPtr, int *yPtr));
EXTERN int		TkGetProlog _ANSI_ARGS_((Tcl_Interp *interp));
EXTERN void		TkGetServerInfo _ANSI_ARGS_((Tcl_Interp *interp,
			    Tk_Window tkwin));
EXTERN void		TkGrabDeadWindow _ANSI_ARGS_((TkWindow *winPtr));
EXTERN int		TkGrabState _ANSI_ARGS_((TkWindow *winPtr));
EXTERN void		TkIncludePoint _ANSI_ARGS_((Tk_Item *itemPtr,
			    double *pointPtr));
EXTERN void		TkInitXId _ANSI_ARGS_((TkDisplay *dispPtr));
EXTERN void		TkInOutEvents _ANSI_ARGS_((XEvent *eventPtr,
			    TkWindow *sourcePtr, TkWindow *destPtr,
			    int leaveType, int enterType,
			    Tcl_QueuePosition position));
EXTERN void		TkInstallFrameMenu _ANSI_ARGS_((Tk_Window tkwin));
#ifndef TkIntersectRegion
EXTERN void		TkIntersectRegion _ANSI_ARGS_((TkRegion sra,
			    TkRegion srcb, TkRegion dr_return));
#endif
EXTERN char *		TkKeysymToString _ANSI_ARGS_((KeySym keysym));
EXTERN int		TkLineToArea _ANSI_ARGS_((double end1Ptr[2],
			    double end2Ptr[2], double rectPtr[4]));
EXTERN double		TkLineToPoint _ANSI_ARGS_((double end1Ptr[2],
			    double end2Ptr[2], double pointPtr[2]));
EXTERN int		TkMakeBezierCurve _ANSI_ARGS_((Tk_Canvas canvas,
			    double *pointPtr, int numPoints, int numSteps,
			    XPoint xPoints[], double dblPoints[]));
EXTERN void		TkMakeBezierPostscript _ANSI_ARGS_((Tcl_Interp *interp,
			    Tk_Canvas canvas, double *pointPtr,
			    int numPoints));
EXTERN void		TkOptionClassChanged _ANSI_ARGS_((TkWindow *winPtr));
EXTERN void		TkOptionDeadWindow _ANSI_ARGS_((TkWindow *winPtr));
EXTERN int		TkOvalToArea _ANSI_ARGS_((double *ovalPtr,
			    double *rectPtr));
EXTERN double		TkOvalToPoint _ANSI_ARGS_((double ovalPtr[4],
			    double width, int filled, double pointPtr[2]));
EXTERN int		TkpChangeFocus _ANSI_ARGS_((TkWindow *winPtr,
			    int force));
EXTERN void		TkpCloseDisplay _ANSI_ARGS_((TkDisplay *dispPtr));
EXTERN void		TkpClaimFocus _ANSI_ARGS_((TkWindow *topLevelPtr,
			    int force));
#ifndef TkpCmapStressed
EXTERN int		TkpCmapStressed _ANSI_ARGS_((Tk_Window tkwin,
			    Colormap colormap));
#endif
#ifndef TkpCreateNativeBitmap
EXTERN Pixmap		TkpCreateNativeBitmap _ANSI_ARGS_((Display *display,
			    char * source));
#endif
#ifndef TkpDefineNativeBitmaps
EXTERN void		TkpDefineNativeBitmaps _ANSI_ARGS_((void));
#endif
EXTERN void		TkpDisplayWarning _ANSI_ARGS_((char *msg,
			    char *title));
EXTERN void		TkpGetAppName _ANSI_ARGS_((Tcl_Interp *interp,
			    Tcl_DString *name));
EXTERN unsigned long	TkpGetMS _ANSI_ARGS_((void));
#ifndef TkpGetNativeAppBitmap
EXTERN Pixmap		TkpGetNativeAppBitmap _ANSI_ARGS_((Display *display,
			    char *name, int *width, int *height));
#endif
EXTERN TkWindow *	TkpGetOtherWindow _ANSI_ARGS_((TkWindow *winPtr));
EXTERN TkWindow *	TkpGetWrapperWindow _ANSI_ARGS_((TkWindow *winPtr));
EXTERN int		TkpInit _ANSI_ARGS_((Tcl_Interp *interp));
EXTERN void		TkpInitializeMenuBindings _ANSI_ARGS_((
			    Tcl_Interp *interp, Tk_BindingTable bindingTable));
EXTERN void		TkpMakeContainer _ANSI_ARGS_((Tk_Window tkwin));
EXTERN void		TkpMakeMenuWindow _ANSI_ARGS_((Tk_Window tkwin,
			    int transient));
EXTERN Window		TkpMakeWindow _ANSI_ARGS_((TkWindow *winPtr,
			    Window parent));
EXTERN void		TkpMenuNotifyToplevelCreate _ANSI_ARGS_((
			    Tcl_Interp *, char *menuName));
EXTERN TkDisplay *	TkpOpenDisplay _ANSI_ARGS_((char *display_name));
EXTERN void		TkPointerDeadWindow _ANSI_ARGS_((TkWindow *winPtr));
EXTERN int		TkPointerEvent _ANSI_ARGS_((XEvent *eventPtr,
			    TkWindow *winPtr));
EXTERN int		TkPolygonToArea _ANSI_ARGS_((double *polyPtr,
			    int numPoints, double *rectPtr));
EXTERN double		TkPolygonToPoint _ANSI_ARGS_((double *polyPtr,
			    int numPoints, double *pointPtr));
EXTERN int		TkPositionInTree _ANSI_ARGS_((TkWindow *winPtr,
			    TkWindow *treePtr));
#ifndef TkpPrintWindowId
EXTERN void		TkpPrintWindowId _ANSI_ARGS_((char *buf,
			    Window window));
#endif
EXTERN void		TkpRedirectKeyEvent _ANSI_ARGS_((TkWindow *winPtr,
			    XEvent *eventPtr));
#ifndef TkpScanWindowId
EXTERN int		TkpScanWindowId _ANSI_ARGS_((Tcl_Interp *interp,
			    char *string, int *idPtr));
#endif
EXTERN void		TkpSetCapture _ANSI_ARGS_((TkWindow *winPtr));
EXTERN void		TkpSetCursor _ANSI_ARGS_((TkpCursor cursor));
EXTERN void		TkpSetMainMenubar _ANSI_ARGS_((Tcl_Interp *interp,
			    Tk_Window tkwin, char *menuName));
#ifndef TkpSync
EXTERN void		TkpSync _ANSI_ARGS_((Display *display));
#endif
EXTERN int		TkpTestembedCmd _ANSI_ARGS_((ClientData clientData,
			    Tcl_Interp *interp, int argc, char **argv));
EXTERN int		TkpUseWindow _ANSI_ARGS_((Tcl_Interp *interp,
			    Tk_Window tkwin, char *string));
#ifndef TkPutImage
EXTERN void		TkPutImage _ANSI_ARGS_((unsigned long *colors,
			    int ncolors, Display* display, Drawable d,
			    GC gc, XImage* image, int src_x, int src_y,
			    int dest_x, int dest_y, unsigned int width,
			    unsigned int height));
#endif
EXTERN int		TkpWindowWasRecentlyDeleted _ANSI_ARGS_((Window win,
			    TkDisplay *dispPtr));
EXTERN void		TkpWmSetState _ANSI_ARGS_((TkWindow *winPtr,
			    int state));
EXTERN void		TkQueueEventForAllChildren _ANSI_ARGS_((
			    TkWindow *winPtr, XEvent *eventPtr));
EXTERN int		TkReadBitmapFile _ANSI_ARGS_((Display* display,
			    Drawable d, CONST char* filename,
			    unsigned int* width_return,
			    unsigned int* height_return,
			    Pixmap* bitmap_return,
			    int* x_hot_return, int* y_hot_return));
#ifndef TkRectInRegion
EXTERN int		TkRectInRegion _ANSI_ARGS_((TkRegion rgn,
			    int x, int y, unsigned int width,
			    unsigned int height));
#endif
EXTERN int		TkScrollWindow _ANSI_ARGS_((Tk_Window tkwin, GC gc,
			    int x, int y, int width, int height, int dx,
			    int dy, TkRegion damageRgn));
EXTERN void		TkSelDeadWindow _ANSI_ARGS_((TkWindow *winPtr));
EXTERN void		TkSelEventProc _ANSI_ARGS_((Tk_Window tkwin,
			    XEvent *eventPtr));
EXTERN void		TkSelInit _ANSI_ARGS_((Tk_Window tkwin));
EXTERN void		TkSelPropProc _ANSI_ARGS_((XEvent *eventPtr));
EXTERN void		TkSetClassProcs _ANSI_ARGS_((Tk_Window tkwin,
			    TkClassProcs *procs, ClientData instanceData));
#ifndef TkSetPixmapColormap
EXTERN void		TkSetPixmapColormap _ANSI_ARGS_((Pixmap pixmap,
			    Colormap colormap));
#endif
#ifndef TkSetRegion
EXTERN void		TkSetRegion _ANSI_ARGS_((Display* display, GC gc,
			    TkRegion rgn));
#endif
EXTERN void		TkSetWindowMenuBar _ANSI_ARGS_((Tcl_Interp *interp,
			    Tk_Window tkwin, char *oldMenuName, 
			    char *menuName));
EXTERN KeySym		TkStringToKeysym _ANSI_ARGS_((char *name));
EXTERN int		TkThickPolyLineToArea _ANSI_ARGS_((double *coordPtr,
			    int numPoints, double width, int capStyle,
			    int joinStyle, double *rectPtr));
#ifndef TkUnionRectWithRegion
EXTERN void		TkUnionRectWithRegion _ANSI_ARGS_((XRectangle* rect,
			    TkRegion src, TkRegion dr_return));
#endif
EXTERN void		TkWmAddToColormapWindows _ANSI_ARGS_((
			    TkWindow *winPtr));
EXTERN void		TkWmDeadWindow _ANSI_ARGS_((TkWindow *winPtr));
EXTERN TkWindow *	TkWmFocusToplevel _ANSI_ARGS_((TkWindow *winPtr));
EXTERN void		TkWmMapWindow _ANSI_ARGS_((TkWindow *winPtr));
EXTERN void		TkWmNewWindow _ANSI_ARGS_((TkWindow *winPtr));
EXTERN void		TkWmProtocolEventProc _ANSI_ARGS_((TkWindow *winPtr,
			    XEvent *evenvPtr));
EXTERN void		TkWmRemoveFromColormapWindows _ANSI_ARGS_((
			    TkWindow *winPtr));
EXTERN void		TkWmRestackToplevel _ANSI_ARGS_((TkWindow *winPtr,
			    int aboveBelow, TkWindow *otherPtr));
EXTERN void		TkWmSetClass _ANSI_ARGS_((TkWindow *winPtr));
EXTERN void		TkWmUnmapWindow _ANSI_ARGS_((TkWindow *winPtr));

/* 
 * Unsupported commands.
 */
EXTERN int		TkUnsupported1Cmd _ANSI_ARGS_((ClientData clientData,
			    Tcl_Interp *interp, int argc, char **argv));

# undef TCL_STORAGE_CLASS
# define TCL_STORAGE_CLASS DLLIMPORT

#endif  /* _TKINT */
