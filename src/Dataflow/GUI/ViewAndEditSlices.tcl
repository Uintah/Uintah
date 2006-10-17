#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
#


    
itcl_class SCIRun_Render_ViewAndEditSlices {
    inherit Module

    constructor {config} {
	set name ViewAndEditSlices
    }

    destructor {
	set children [winfo children .]
	set pos [lsearch $children .ui[modname]*]
	while { $pos != -1 } {
	    destroy [lindex $children $pos]
	    set children [winfo children .]
	    set pos [lsearch $children .ui[modname]*]
	}
    }
	    


    method generate_window_num {} {
	set num 1
	set w .ui[modname]_$num
	set children [winfo children .]
	while { [lsearch $children $w] != -1 } {
	    set w .ui[modname]_[incr num]
	}
	return $num
    }

    # writeStateToScript
    # Called from genSubnetScript, it will append the TCL
    # commands needed to initialize this module's variables
    # after it is created.  This is located here in the Module class
    # so sub-classes (like SCIRun_Render_Viewer) can specialize
    # the variables they write out
    #
    # 'scriptVar' is the name of the TCL variable one level
    # up that we will append our commands to 
    # 'prefix' is the prefix the variables
    # 'tab' is the indent string to make it look pretty
    method writeStateToScript { scriptVar prefix { tab "" }} {
	upvar 1 $scriptVar script2
	set num 0
	foreach w [winfo children .] {
	    if { [string first .ui[modname] $w] == 0 } {
		append script2 "\n${tab}${prefix} ui"
	    }
	}
	Module::writeStateToScript script2 $prefix $tab
    }

    method gl_frame { w } {
	set id [lindex [split $w .] end]
	$this-c setgl $w $id 0
	$this-c add_viewport $w $id-viewport0
	bind $w <Expose>	"$this-c redraw $w"
	bind $w <Configure>	"$this-c resize $w"
	bind $w <Destroy>	"$this-c destroygl $w"
	# the focus belowis to generate keypress events 
	bind $w <Enter>		"focus %W; 
                                 $this-c event enter   %W %b %s %X %Y %x %y %t"
	bind $w <Leave>		"$this-c event leave   %W %b %s %X %Y %x %y %t"
	bind $w <Motion>	"$this-c event motion  %W %b %s %X %Y %x %y %t"
	bind $w <ButtonPress>	"$this-c event button  %W %b %s %X %Y %x %y %t"
	bind $w <ButtonRelease>	"$this-c event release %W %b %s %X %Y %x %y %t"
	bind $w <KeyPress>	"$this-c event keydown %W %K %s %X %Y %x %y %t"
	bind $w <KeyRelease>	"$this-c event keyup   %W %K %s %X %Y %x %y %t"
	return $w
    }

    method ui { } {
	set children [winfo children .]
	set pos 0
	set create 1
	while { $pos != -1 } {
	    set children [lrange $children $pos end]
	    set pos [lsearch  $children .ui[modname]*]
	    if { $pos != -1 } {
		SciRaise [lindex $children $pos]
		incr pos
		set create 0
	    }
	}
	if { $create } create_ui
    }

    method create_ui { } {
	set num [generate_window_num]
	set title "Window $num"
	set w .ui[modname]
	toplevel $w
	wm title $w "[modname] $title"
	wm protocol $w WM_DELETE_WINDOW "wm withdraw $w"
        four_view $w
    }

    method four_view { w } {
	iwidgets::panedwindow $w.topbot -orient horizontal -thickness 0 \
	    -sashwidth 5000 -sashindent 0 -sashborderwidth 2 -sashheight 6 \
	    -sashcursor sb_v_double_arrow -width 800 -height 800
	pack $w.topbot -expand 1 -fill both -padx 0 -ipadx 0 -pady 0 -ipady 0
	
	$w.topbot add top -margin 0 -minimum 0
	$w.topbot add bottom  -margin 0 -minimum 0

	set top [$w.topbot childsite top]
	set bot [$w.topbot childsite bottom]
	
	Linkedpane $top.lr -orient vertical -thickness 0 \
	    -sashheight 5000 -sashwidth 6 -sashindent 0 -sashborderwidth 2 \
	    -sashcursor sb_h_double_arrow

	$top.lr add left -margin 3 -minimum 0
	$top.lr add right -margin 3 -minimum 0
	set topl [$top.lr childsite left]
	set topr [$top.lr childsite right]

	Linkedpane $bot.lr  -orient vertical -thickness 0 \
	    -sashheight 5000 -sashwidth 6 -sashindent 0 -sashborderwidth 2 \
	    -sashcursor sb_h_double_arrow

	$bot.lr set_link $top.lr
	$top.lr set_link $bot.lr

	$bot.lr add left -margin 3 -minimum 0
	$bot.lr add right -margin 3 -minimum 0
	set botl [$bot.lr childsite left]
	set botr [$bot.lr childsite right]

	pack $top.lr -expand 1 -fill both -padx 0 -ipadx 0 -pady 0 -ipady 0
	pack $bot.lr -expand 1 -fill both -padx 0 -ipadx 0 -pady 0 -ipady 0

 	pack [gl_frame $topl.gl1] -expand 1 -fill both -padx 0 -ipadx 0 -pady 0 -ipady 0
 	pack [gl_frame $topr.gl2] -expand 1 -fill both -padx 0 -ipadx 0 -pady 0 -ipady 0
 	pack [gl_frame $botl.gl3] -expand 1 -fill both -padx 0 -ipadx 0 -pady 0 -ipady 0

        set viewer [info commands SCIRun_Render_Viewer_0]
        if { ![llength $viewer] } return;

 	set eviewer [$viewer ui_embedded]
        frame $botr.frame -relief flat -borderwidth 15
 	pack $botr.frame -expand 1 -fill both -padx 0 -ipadx 0 -pady 0 -ipady 0
        frame $botr.frame.sunken -relief sunken -borderwidth 2
 	pack $botr.frame.sunken -expand 1 -fill both -padx 0 -ipadx 0 -pady 0 -ipady 0

 	$eviewer setWindow $botr.frame.sunken.gl 300 300

 	pack $botr.frame.sunken.gl -expand 1 -fill both -padx 0 -ipadx 0 -pady 0 -ipady 0
        bind $w <Control-v> "$eviewer-c autoview"


    }
}
    
