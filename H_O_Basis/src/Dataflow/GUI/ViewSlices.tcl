#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  License for the specific language governing rights and limitations under
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


itk::usual Linkedpane {
    keep -background -cursor -sashcursor
}



# Linked Pane is just a iwidgets::Panedwindow that is linked to another one while moving
class Linkedpane {
    inherit ::iwidgets::Panedwindow
    private variable other ""
    constructor { args } { 
        eval ::iwidgets::Panedwindow::constructor $args
    }
    public method set_link { w } { set other $w }

    protected method _startGrip {where num { from_link 0 } } {
	if { !$from_link && [winfo exists $other] } { 
	    $other _startGrip $where $num 1
	}
        ::iwidgets::Panedwindow::_startGrip $where $num
    }

    protected method _endGrip {where num {from_link 0}} {
	if { !$from_link && [winfo exists $other] } { 
	    $other _endGrip $where $num 1
	}
	::iwidgets::Panedwindow::_endGrip $where $num
    }

    protected method _moveSash {where num {from_link 0}} {
	if { !$from_link && [winfo exists $other] } { 
	    $other _moveSash $where $num 1
	}
	::iwidgets::Panedwindow::_moveSash $where $num
    }
}
	
    
itcl_class SCIRun_Render_ViewSlices {
    inherit Module
    protected vp_tabs ""

    constructor {config} {
	set name ViewSlices
	uplevel \#0 trace variable $this-min w \"$this update_clut_range\"
	uplevel \#0 trace variable $this-max w \"$this update_clut_range\"
	
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
	    

    method update_clut_range {args} {
	upvar \#0 [modname]-min min [modname]-max max
	set ww [expr $max-$min]
	set wl [expr $min+$ww/2]
	set rez [expr $ww/1000]
	foreach tab $vp_tabs {
	    $tab.clutww.scale configure -from 0 -to $ww -resolution $rez
	    $tab.clutwl.scale configure -from $min -to $max -resolution $rez
	    $tab.clutww.scale set $ww
	    $tab.clutwl.scale set $wl
	}
    }
	

    method labeledSlider { frame text var from to res {width 13}} {
	frame $frame
	pack $frame -side top -expand 1 -fill x
	label $frame.label -text $text -anchor w -width $width
	pack $frame.label -side left -expand 0 -fill none
	scale $frame.scale -orient horizontal -variable $var \
	    -from $from -to $to -resolution $res -showvalue 0 
	entry $frame.entry -text $var -width 4 -justify right
	pack $frame.entry -side right -expand 0 -fill x
	pack $frame.scale -side right -expand 1 -fill x
	pack $frame -side top -expand 0 -fill x
	return $frame.scale
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
	bind $w <Enter>		"focus $w; $this-c enter $w"
	bind $w <Leave>		"$this-c leave %W; $this-c redrawall"
	bind $w <Motion>	"$this-c motion   $w %x %y %s %t %X %Y"
	bind $w <KeyPress>	"$this-c keypress $w %k %K %t"
	bind $w <ButtonPress>	"$this-c button   $w %b %s %X %Y"
	bind $w <ButtonRelease> "$this-c release  $w %b %s %X %Y"
	return $w
    }

    method add_tab { w title } {
	set f [$w.cp.tabs add -label $title].f
	frame $f -bd 2 -relief sunken
	pack $f -expand 1 -fill both
	$w.cp.tabs select end
	return $f
    }
	

    method add_nrrd_tab { w num } {
	set w [add_tab $w "Nrrd$num"]
	foreach c {x y z} {
	    checkbutton $w.flip$c -text "Nrrd$num Flip [string toupper $c]" \
		-variable [modname]-nrrd$num-flip_$c \
		-command "$this-c redrawall"
	    pack $w.flip$c -side top
	}

 	foreach c {yz xz xy} {
	    checkbutton $w.transpose$c \
		-text "Nrrd$num Transpose [string toupper $c]" \
		-variable [modname]-nrrd$num-transpose_$c \
		-command "$this-c redrawall"
	    pack $w.transpose$c -side top
	}
    }

    method add_viewport_tab { w name prefix gl } {
	set prefix [modname]-$prefix
	set f [add_tab $w "$name"]
	lappend vp_tabs $f
	labeledSlider $f.slice Slice: $prefix-slice 0 255 1
	$f.slice.scale configure -command \
	    "$this-c rebind $gl"

	labeledSlider $f.zoom "Zoom %:" $prefix-zoom 1 2000 3
	$f.zoom.scale configure -command \
	    "$this-c redraw $gl"

	labeledSlider $f.clutww "Window Width:" [modname]-clut_ww 1 2000 3
	$f.clutww.scale configure -command \
	    "$this-c rebind $gl"

	labeledSlider $f.clutwl "Window Level:" [modname]-clut_wl 1 2000 3
	$f.clutwl.scale configure -command \
	    "$this-c rebind $gl"

	labeledSlider $f.fusion "Image Fusion:" $prefix-fusion 0 1 0.001
	$f.fusion.scale configure -command \
	    "$this-c redraw $gl"
	update_clut_range

	frame $f.f -bd 0
	checkbutton $f.f.guidelines -text "Show Guidelines" \
	    -variable $prefix-show_guidelines \
	    -command "$this-c redraw $gl"

	checkbutton $f.f.mip -text "MIP" \
	    -variable $prefix-mode \
	    -onvalue 1 \
	    -offvalue 0 \
	    -command "$this-c rebind $gl"
	pack $f.f.mip $f.f.guidelines -side left -anchor w

	bind $gl <ButtonPress> "+$w.cp.tabs select \"$name\""
	pack $f.f -side top -anchor w
    }	
	
	
    method control_panel { w } {
	frame $w -bd 1 -relief groove
	iwidgets::tabnotebook $w.tabs -raiseselect true \
	    -tabpos s -backdrop gray -equaltabs -0 -bevelamount 5 \
	    -borderwidth 0
	pack $w.tabs -expand 1 -fill both
    }



    method show_control_panel { w } {
	pack forget $w.e $w.cp $w.f
	pack $w.cp -side bottom -fill both -expand 0
	pack $w.e -side bottom -fill x
	pack $w.f -expand 1 -fill both

	$w.e configure -command "$this hide_control_panel $w" \
	    -cursor based_arrow_up
    }

    method hide_control_panel { w } {
	pack forget $w.e $w.cp $w.f
	pack $w.e -side bottom -fill x
	pack $w.f -expand 1 -fill both
	$w.e configure -command "$this show_control_panel $w" \
	    -cursor based_arrow_down
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
	set w .ui[modname]_$num
	toplevel $w
	wm title $w "[modname] $title"
	wm protocol $w WM_DELETE_WINDOW "wm withdraw $w"

	frame $w.f -bd 0 -background red
	set img [image create photo -width 1 -height 1]
	button $w.e -height 4 -bd 2 -relief raised -image $img \
	    -cursor based_arrow_down
	button $w.b -text "New Slice Viewer Window" -command "$this create_ui"
	pack $w.b -anchor w
	pack $w.e -side bottom -fill x
	pack $w.f -expand 1 -fill both

	control_panel $w.cp
	show_control_panel $w

	set winname [join [string tolower $title] ""]
	gl_frame $w.f.$winname
	pack $w.f.$winname -expand 1 -fill both

	add_viewport_tab $w $title $winname-viewport0 $w.f.$winname
    }

    method four_view { w main } {
	iwidgets::panedwindow $w.topbot -orient horizontal -thickness 0 \
	    -sashwidth 5000 -sashindent 0 -sashborderwidth 2 -sashheight 6 \
	    -sashcursor sb_v_double_arrow -width 500 -height 500
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

	
	pack [gl_frame $topl "Top Left" $main] -expand 1 -fill both -padx 0 -ipadx 0 -pady 0 -ipady 0
	pack [gl_frame $topr "Top Right" $main] -expand 1 -fill both -padx 0 -ipadx 0 -pady 0 -ipady 0
	pack [gl_frame $botl "Bottom Left" $main] -expand 1 -fill both -padx 0 -ipadx 0 -pady 0 -ipady 0
	pack [gl_frame $botr "Bottom Right" $main] -expand 1 -fill both


    }

    method raise_color {button color module_command} {
        global $color
        set windowname .ui[modname]_color
        if {[winfo exists $windowname]} {
	    destroy $windowname
	}
	# makeColorPicker now creates the $window.color toplevel.
	makeColorPicker $windowname $color \
	    "$this set_color $button $color $module_command" \
	    "destroy $windowname"
    }
    
    method set_color { button color { module_command "" } } {
	upvar \#0 $color-r r $color-g g $color-b b
	# format the r,g,b colors into a hexadecimal string representation
	set colstr [format \#%04x%04x%04x [expr int($r * 65535)] \
			[expr int($g * 65535)] [expr int($b * 65535)]]
	$button config -background $colstr -activebackground $colstr
	if { [string length $module_command] } {
	    $this-c $module_command
	}
    }

}
    
