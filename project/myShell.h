/*
* $XConsortium: Shell.h,v 1.23 91/05/04 20:59:47 rws Exp $
*/
/***********************************************************
Copyright 1987, 1988 by Digital Equipment Corporation, Maynard, Massachusetts,
and the Massachusetts Institute of Technology, Cambridge, Massachusetts.

                        All Rights Reserved

Permission to use, copy, modify, and distribute this software and its 
documentation for any purpose and without fee is hereby granted, 
provided that the above copyright notice appear in all copies and that
both that copyright notice and this permission notice appear in 
supporting documentation, and that the names of Digital or MIT not be
used in advertising or publicity pertaining to distribution of the
software without specific, written prior permission.  

DIGITAL DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING
ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT SHALL
DIGITAL BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR
ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS
SOFTWARE.

******************************************************************/

#ifndef _XtShell_h
#define _XtShell_h

/***********************************************************************
 *
 * Shell Widget
 *
 ***********************************************************************/
/*
 * Shell-specific resources names, classes, and a representation type.
 */

/* The string definitions are automatically generated. */
/* Do not edit. */

#ifdef XTSTRINGDEFINES
#define XtNiconName "iconName"
#define XtCIconName "IconName"
#define XtNiconPixmap "iconPixmap"
#define XtCIconPixmap "IconPixmap"
#define XtNiconWindow "iconWindow"
#define XtCIconWindow "IconWindow"
#define XtNiconMask "iconMask"
#define XtCIconMask "IconMask"
#define XtNwindowGroup "windowGroup"
#define XtCWindowGroup "WindowGroup"
#define XtNvisual "visual"
#define XtCVisual "Visual"
#define XtNtitleEncoding "titleEncoding"
#define XtCTitleEncoding "TitleEncoding"
#define XtNsaveUnder "saveUnder"
#define XtCSaveUnder "SaveUnder"
#define XtNtransient "transient"
#define XtCTransient "Transient"
#define XtNoverrideRedirect "overrideRedirect"
#define XtCOverrideRedirect "OverrideRedirect"
#define XtNtransientFor "transientFor"
#define XtCTransientFor "TransientFor"
#define XtNiconNameEncoding "iconNameEncoding"
#define XtCIconNameEncoding "IconNameEncoding"
#define XtNallowShellResize "allowShellResize"
#define XtCAllowShellResize "AllowShellResize"
#define XtNcreatePopupChildProc "createPopupChildProc"
#define XtCCreatePopupChildProc "CreatePopupChildProc"
#define XtNtitle "title"
#define XtCTitle "Title"
#define XtRAtom "Atom"
#define XtNargc "argc"
#define XtCArgc "Argc"
#define XtNargv "argv"
#define XtCArgv "Argv"
#define XtNiconX "iconX"
#define XtCIconX "IconX"
#define XtNiconY "iconY"
#define XtCIconY "IconY"
#define XtNinput "input"
#define XtCInput "Input"
#define XtNiconic "iconic"
#define XtCIconic "Iconic"
#define XtNinitialState "initialState"
#define XtCInitialState "InitialState"
#define XtNgeometry "geometry"
#define XtCGeometry "Geometry"
#define XtNbaseWidth "baseWidth"
#define XtCBaseWidth "BaseWidth"
#define XtNbaseHeight "baseHeight"
#define XtCBaseHeight "BaseHeight"
#define XtNwinGravity "winGravity"
#define XtCWinGravity "WinGravity"
#define XtNminWidth "minWidth"
#define XtCMinWidth "MinWidth"
#define XtNminHeight "minHeight"
#define XtCMinHeight "MinHeight"
#define XtNmaxWidth "maxWidth"
#define XtCMaxWidth "MaxWidth"
#define XtNmaxHeight "maxHeight"
#define XtCMaxHeight "MaxHeight"
#define XtNwidthInc "widthInc"
#define XtCWidthInc "WidthInc"
#define XtNheightInc "heightInc"
#define XtCHeightInc "HeightInc"
#define XtNminAspectY "minAspectY"
#define XtCMinAspectY "MinAspectY"
#define XtNmaxAspectY "maxAspectY"
#define XtCMaxAspectY "MaxAspectY"
#define XtNminAspectX "minAspectX"
#define XtCMinAspectX "MinAspectX"
#define XtNmaxAspectX "maxAspectX"
#define XtCMaxAspectX "MaxAspectX"
#define XtNwmTimeout "wmTimeout"
#define XtCWmTimeout "WmTimeout"
#define XtNwaitForWm "waitforwm"
#define XtCWaitForWm "Waitforwm"
#else
#if __STDC__
#define _XtConst_ const
#else
#define _XtConst_ /**/
#endif
extern _XtConst_ char XtShellStrings[];
#ifndef XtNiconName
#define XtNiconName ((const char*)&XtShellStrings[0])
#endif
#ifndef XtCIconName
#define XtCIconName ((const char*)&XtShellStrings[9])
#endif
#ifndef XtNiconPixmap
#define XtNiconPixmap ((const char*)&XtShellStrings[18])
#endif
#ifndef XtCIconPixmap
#define XtCIconPixmap ((const char*)&XtShellStrings[29])
#endif
#ifndef XtNiconWindow
#define XtNiconWindow ((const char*)&XtShellStrings[40])
#endif
#ifndef XtCIconWindow
#define XtCIconWindow ((const char*)&XtShellStrings[51])
#endif
#ifndef XtNiconMask
#define XtNiconMask ((const char*)&XtShellStrings[62])
#endif
#ifndef XtCIconMask
#define XtCIconMask ((const char*)&XtShellStrings[71])
#endif
#ifndef XtNwindowGroup
#define XtNwindowGroup ((const char*)&XtShellStrings[80])
#endif
#ifndef XtCWindowGroup
#define XtCWindowGroup ((const char*)&XtShellStrings[92])
#endif
#ifndef XtNvisual
#define XtNvisual ((const char*)&XtShellStrings[104])
#endif
#ifndef XtCVisual
#define XtCVisual ((const char*)&XtShellStrings[111])
#endif
#ifndef XtNtitleEncoding
#define XtNtitleEncoding ((const char*)&XtShellStrings[118])
#endif
#ifndef XtCTitleEncoding
#define XtCTitleEncoding ((const char*)&XtShellStrings[132])
#endif
#ifndef XtNsaveUnder
#define XtNsaveUnder ((const char*)&XtShellStrings[146])
#endif
#ifndef XtCSaveUnder
#define XtCSaveUnder ((const char*)&XtShellStrings[156])
#endif
#ifndef XtNtransient
#define XtNtransient ((const char*)&XtShellStrings[166])
#endif
#ifndef XtCTransient
#define XtCTransient ((const char*)&XtShellStrings[176])
#endif
#ifndef XtNoverrideRedirect
#define XtNoverrideRedirect ((const char*)&XtShellStrings[186])
#endif
#ifndef XtCOverrideRedirect
#define XtCOverrideRedirect ((const char*)&XtShellStrings[203])
#endif
#ifndef XtNtransientFor
#define XtNtransientFor ((const char*)&XtShellStrings[220])
#endif
#ifndef XtCTransientFor
#define XtCTransientFor ((const char*)&XtShellStrings[233])
#endif
#ifndef XtNiconNameEncoding
#define XtNiconNameEncoding ((const char*)&XtShellStrings[246])
#endif
#ifndef XtCIconNameEncoding
#define XtCIconNameEncoding ((const char*)&XtShellStrings[263])
#endif
#ifndef XtNallowShellResize
#define XtNallowShellResize ((const char*)&XtShellStrings[280])
#endif
#ifndef XtCAllowShellResize
#define XtCAllowShellResize ((const char*)&XtShellStrings[297])
#endif
#ifndef XtNcreatePopupChildProc
#define XtNcreatePopupChildProc ((const char*)&XtShellStrings[314])
#endif
#ifndef XtCCreatePopupChildProc
#define XtCCreatePopupChildProc ((const char*)&XtShellStrings[335])
#endif
#ifndef XtNtitle
#define XtNtitle ((const char*)&XtShellStrings[356])
#endif
#ifndef XtCTitle
#define XtCTitle ((const char*)&XtShellStrings[362])
#endif
#ifndef XtRAtom
#define XtRAtom ((const char*)&XtShellStrings[368])
#endif
#ifndef XtNargc
#define XtNargc ((const char*)&XtShellStrings[373])
#endif
#ifndef XtCArgc
#define XtCArgc ((const char*)&XtShellStrings[378])
#endif
#ifndef XtNargv
#define XtNargv ((const char*)&XtShellStrings[383])
#endif
#ifndef XtCArgv
#define XtCArgv ((const char*)&XtShellStrings[388])
#endif
#ifndef XtNiconX
#define XtNiconX ((const char*)&XtShellStrings[393])
#endif
#ifndef XtCIconX
#define XtCIconX ((const char*)&XtShellStrings[399])
#endif
#ifndef XtNiconY
#define XtNiconY ((const char*)&XtShellStrings[405])
#endif
#ifndef XtCIconY
#define XtCIconY ((const char*)&XtShellStrings[411])
#endif
#ifndef XtNinput
#define XtNinput ((const char*)&XtShellStrings[417])
#endif
#ifndef XtCInput
#define XtCInput ((const char*)&XtShellStrings[423])
#endif
#ifndef XtNiconic
#define XtNiconic ((const char*)&XtShellStrings[429])
#endif
#ifndef XtCIconic
#define XtCIconic ((const char*)&XtShellStrings[436])
#endif
#ifndef XtNinitialState
#define XtNinitialState ((const char*)&XtShellStrings[443])
#endif
#ifndef XtCInitialState
#define XtCInitialState ((const char*)&XtShellStrings[456])
#endif
#ifndef XtNgeometry
#define XtNgeometry ((const char*)&XtShellStrings[469])
#endif
#ifndef XtCGeometry
#define XtCGeometry ((const char*)&XtShellStrings[478])
#endif
#ifndef XtNbaseWidth
#define XtNbaseWidth ((const char*)&XtShellStrings[487])
#endif
#ifndef XtCBaseWidth
#define XtCBaseWidth ((const char*)&XtShellStrings[497])
#endif
#ifndef XtNbaseHeight
#define XtNbaseHeight ((const char*)&XtShellStrings[507])
#endif
#ifndef XtCBaseHeight
#define XtCBaseHeight ((const char*)&XtShellStrings[518])
#endif
#ifndef XtNwinGravity
#define XtNwinGravity ((const char*)&XtShellStrings[529])
#endif
#ifndef XtCWinGravity
#define XtCWinGravity ((const char*)&XtShellStrings[540])
#endif
#ifndef XtNminWidth
#define XtNminWidth ((const char*)&XtShellStrings[551])
#endif
#ifndef XtCMinWidth
#define XtCMinWidth ((const char*)&XtShellStrings[560])
#endif
#ifndef XtNminHeight
#define XtNminHeight ((const char*)&XtShellStrings[569])
#endif
#ifndef XtCMinHeight
#define XtCMinHeight ((const char*)&XtShellStrings[579])
#endif
#ifndef XtNmaxWidth
#define XtNmaxWidth ((const char*)&XtShellStrings[589])
#endif
#ifndef XtCMaxWidth
#define XtCMaxWidth ((const char*)&XtShellStrings[598])
#endif
#ifndef XtNmaxHeight
#define XtNmaxHeight ((const char*)&XtShellStrings[607])
#endif
#ifndef XtCMaxHeight
#define XtCMaxHeight ((const char*)&XtShellStrings[617])
#endif
#ifndef XtNwidthInc
#define XtNwidthInc ((const char*)&XtShellStrings[627])
#endif
#ifndef XtCWidthInc
#define XtCWidthInc ((const char*)&XtShellStrings[636])
#endif
#ifndef XtNheightInc
#define XtNheightInc ((const char*)&XtShellStrings[645])
#endif
#ifndef XtCHeightInc
#define XtCHeightInc ((const char*)&XtShellStrings[655])
#endif
#ifndef XtNminAspectY
#define XtNminAspectY ((const char*)&XtShellStrings[665])
#endif
#ifndef XtCMinAspectY
#define XtCMinAspectY ((const char*)&XtShellStrings[676])
#endif
#ifndef XtNmaxAspectY
#define XtNmaxAspectY ((const char*)&XtShellStrings[687])
#endif
#ifndef XtCMaxAspectY
#define XtCMaxAspectY ((const char*)&XtShellStrings[698])
#endif
#ifndef XtNminAspectX
#define XtNminAspectX ((const char*)&XtShellStrings[709])
#endif
#ifndef XtCMinAspectX
#define XtCMinAspectX ((const char*)&XtShellStrings[720])
#endif
#ifndef XtNmaxAspectX
#define XtNmaxAspectX ((const char*)&XtShellStrings[731])
#endif
#ifndef XtCMaxAspectX
#define XtCMaxAspectX ((const char*)&XtShellStrings[742])
#endif
#ifndef XtNwmTimeout
#define XtNwmTimeout ((const char*)&XtShellStrings[753])
#endif
#ifndef XtCWmTimeout
#define XtCWmTimeout ((const char*)&XtShellStrings[763])
#endif
#ifndef XtNwaitForWm
#define XtNwaitForWm ((const char*)&XtShellStrings[773])
#endif
#ifndef XtCWaitForWm
#define XtCWaitForWm ((const char*)&XtShellStrings[783])
#endif
#undef _XtConst_
#endif

/* Class record constants */

typedef struct _ShellClassRec *ShellWidgetClass;
typedef struct _OverrideShellClassRec *OverrideShellWidgetClass;
typedef struct _WMShellClassRec *WMShellWidgetClass;
typedef struct _TransientShellClassRec *TransientShellWidgetClass;
typedef struct _TopLevelShellClassRec *TopLevelShellWidgetClass;
typedef struct _ApplicationShellClassRec *ApplicationShellWidgetClass;

#ifndef SHELL 
externalref WidgetClass shellWidgetClass;
externalref WidgetClass overrideShellWidgetClass;
externalref WidgetClass wmShellWidgetClass;
externalref WidgetClass transientShellWidgetClass;
externalref WidgetClass topLevelShellWidgetClass;
externalref WidgetClass applicationShellWidgetClass;
#endif

#endif /* _XtShell_h */
/* DON'T ADD STUFF AFTER THIS #endif */
