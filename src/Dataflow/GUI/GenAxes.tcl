#
#  The contents of this file are subject to the University of Utah Public
#  License (the "License"); you may not use this file except in compliance
#  with the License.
#  
#  Software distributed under the License is distributed on an "AS IS"
#  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
#  License for the specific language governing rights and limitations under
#  the License.
#  
#  The Original Source Code is SCIRun, released March 12, 2001.
#  
#  The Original Source Code was developed by the University of Utah.
#  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
#  University of Utah. All Rights Reserved.
#
#
#  GenAxes.tcl
#
#  Written by:
#   McKay Davis
#   Department of Computer Science
#   University of Utah
#   November 10 2003
#
#  Copyright (C) 2003 SCI Group
#



proc setGlobal { name val } {
    upvar \#0 $name var
    set var $val
}

proc updatedPercentage { first second absolute relative name1 name2 op } {
    upvar $first min
    upvar $second max
    upvar $absolute abs
    upvar $name1 cent
    set abs [expr ($relative?$min:0) + ($cent/100.0) * ($max - $min)]
}


proc updatedPercentage2 { divisions name1 name2 op } {
    if [string equal $name1 div] return
    upvar $divisions div
    upvar $name1 cent
    set div [expr 100.0 / $cent]    
}


proc updatedAbsolute { first second percent relative name1 name2 op } {
    upvar $first min
    upvar $second max
    upvar $name1 abs
    upvar $percent cent
    set cent [expr 100.0*($abs - ($relative?$min:0))/($max - $min)]
}
    

proc updatedRangeFirst { second absolute percent relative first name2 op } {
    upvar $first min
    upvar $second max
    upvar $absolute abs
    upvar $percent cent
    set abs [expr ($relative?$min:0) + ($cent/100.0) * ($max - $min)]
#    set cent [expr 100.0*($abs - ($relative?$min:0))/($max - $min)]
}

proc updatedRangeSecond { first absolute percent relative second name2 op } {
    upvar $first min
    upvar $second max
    upvar $absolute abs
    upvar $percent cent
    set abs [expr ($relative?$min:0) + ($cent/100.0) * ($max - $min)]
#    set cent [expr 100.0*($abs - ($relative?$min:0))/($max - $min)]
}



proc setPercentage { var which val } {
    global $var-$which-percent $var-$which-absoltue 
    global $var-range-first $var-range-second
    set $var-$which-precent $val
    set $var-$which-absolute \
	[expr [set $var-range-first] + ($val/100.0) * \
	     ([set $var-range-second] - [set $var-range-first])]
}

proc addLabeledFrame { w text } {
    set frame $w.[string tolower [join $text ""]]
    iwidgets::labeledframe $frame -labelpos nw -labeltext $text
    pack $frame -side top -fill x -expand 0 -pady 0 -ipady 0
    set frame [$frame childsite]
#    frame $frame
#    pack $frame -expand 1 -fill both
    return $frame
}

proc displayRadios { w var text } {
    set w $w.[lindex [split $var -] end]
    frame $w
    label $w.label -width 10 -anchor w -text $text:
    radiobutton $w.on -text On -value 1 -variable $var
    radiobutton $w.off -text Off -value 0 -variable $var
    radiobutton $w.auto -text Auto -value 2 -variable $var
    pack $w.label $w.off $w.on $w.auto -side left -expand 1 -fill x
    pack $w -side top -expand 0 -fill x
}

proc labeledSlider { w text var from to res {width 13}} {
    set frame $w.[lindex [split $var -] end]
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

proc setVars { tovar name1 name2 op} {
    upvar \#0 $name1 var
    setGlobal $tovar $var
    return 1
}


itcl_class SCIRun_Visualization_GenAxes {
    inherit Module
    constructor {config} {
        set name GenAxes
        set_defaults
    }

    method set_defaults {} {
	setGlobal $this-precision  "3"
	setGlobal $this-squash  "0.7"
	setGlobal $this-valuerez  "72"
	setGlobal $this-labelrez "72"
	createDefaultPlaneAxisVariables $this-Plane-01-0-Axis
	createDefaultPlaneAxisVariables $this-Plane-01-1-Axis
	createDefaultPlaneAxisVariables $this-Plane-02-0-Axis
	createDefaultPlaneAxisVariables $this-Plane-02-2-Axis
	createDefaultPlaneAxisVariables $this-Plane-12-1-Axis
	createDefaultPlaneAxisVariables $this-Plane-12-2-Axis
	
    }

    method axisName { axes } {
	set $this-gridnames "{} Minor-"
	set axis [lindex $axes end]
	set ret [lindex [set $this-gridnames] [expr $axis/3]]
	foreach axis $axes {
	    switch [expr $axis%3] {
		0 { set ret "${ret}X" }
		1 { set ret "${ret}Y" }
		2 { set ret "${ret}Z" }
	    }
	}
	return $ret
    }


    method createDefaultPlaneAxisVariables { varPrefix } {

	setGlobal $varPrefix-divisions		"10"
	setGlobal $varPrefix-percent		"10"
	setGlobal $varPrefix-absolute		""
	setGlobal $varPrefix-offset		"1"

	setGlobal $varPrefix-range-first	"0.0"
	setGlobal $varPrefix-range-second	"1.0"
	setGlobal $varPrefix-min-percent	"0"
	setGlobal $varPrefix-max-percent	"100"
	setGlobal $varPrefix-min-absolute	"0.0"
	setGlobal $varPrefix-max-absolute	"1.0"


	setGlobal $varPrefix-minplane		"2"
	setGlobal $varPrefix-maxplane		"2"
	setGlobal $varPrefix-lines		"2"
	setGlobal $varPrefix-minticks		"2"
	setGlobal $varPrefix-maxticks		"2"
	setGlobal $varPrefix-minlabel		"2"
	setGlobal $varPrefix-maxlabel		"2"
	setGlobal $varPrefix-minvalue		"2"
	setGlobal $varPrefix-maxvalue		"2"

	setGlobal $varPrefix-width		"1"
	setGlobal $varPrefix-tickangle		"0"
	setGlobal $varPrefix-ticktilt 		"0"
	setGlobal $varPrefix-labelangle		"0"
	setGlobal $varPrefix-labelheight	"6"
	setGlobal $varPrefix-ticksize		"5"
	setGlobal $varPrefix-valuesquash	"1.0"	
	setGlobal $varPrefix-valuesize		"3"	
    }

    method linkVars { allaxes args } {
	set fromPre "[modname]-all-[join [join $allaxes ""] ""]"
	createDefaultPlaneAxisVariables $fromPre
	foreach varName $args {
	    foreach axes $allaxes  {
		foreach axis $axes {
		    set planeName [join "Plane-$axes" ""]
		    set axisName [join "$axis-Axis" ""]
		    set toVar "$this-$planeName-$axisName-$varName"
		    set fromVar "$fromPre-$varName"
		    global $toVar $fromVar
		    set command "trace variable $fromVar w \"setVars $toVar\""
		    uplevel \#0 $command
		}
	    }
	}
	return $fromPre
    }


    method build_all_tab { tabs } {

	set fromPre [linkVars {"0 1" "0 2" "1 2"} lines divisions percent absolute offset range-frist range-second min-percent max-percent min-absolute max-absolute minplane maxplane lines minticks maxticks minlabel maxlabel minvalue maxvalue width tickangle ticktilt labelangle lableheight ticksize valuesquash valuesize]

	set tab [$tabs add -label "All"]
	set tabframe $tab.frame
	frame $tabframe
	pack $tabframe -expand 1 -fill both
	
	build_plane_dir_ui $tabframe $fromPre "0 1 2" "0 1 2"

    }



    method build_plane_dir_ui { f varPrefix axes axis } {
	setGlobal $varPrefix-text \
	    "[axisName $axes] Plane [axisName $axis] Axis"
	
	trace variable $varPrefix-percent w \
	    "updatedPercentage $varPrefix-range-first $varPrefix-range-second $varPrefix-absolute 0"
	
	
	trace variable $varPrefix-percent w \
	    "updatedPercentage2 $varPrefix-divisions"
	
	trace variable $varPrefix-divisions w \
	    "updatedPercentage2 $varPrefix-percent"
	
	
	trace variable $varPrefix-absolute w \
	    "updatedAbsolute $varPrefix-range-first $varPrefix-range-second $varPrefix-percent 0"
	
	trace variable $varPrefix-range-first w \
	    "updatedRangeFirst $varPrefix-range-second $varPrefix-absolute $varPrefix-percent 0"
	
	trace variable $varPrefix-range-second w \
	    "updatedRangeSecond $varPrefix-range-first $varPrefix-absolute $varPrefix-percent 0"
	
	
	set title "[axisName $axis] Axis Intervals"
	set Frame [addLabeledFrame $f $title]
	set Frame1 $Frame
	
	labeledSlider $Frame "Total \#:" $varPrefix-divisions 0 100 0.5 7
	
	set frame $Frame.end
	frame $frame
	pack $frame -side top -expand 1 -fill x
	label $frame.label -anchor w -text "Interval: " -width 7
	pack $frame.label -side left -expand 0 -fill none
	
	scale $frame.scale -orient horizontal -from 0 \
	    -to 100 -showvalue 0 -resolution 0.001 \
	    -variable $varPrefix-percent
	
	entry $frame.percent -width 4 -justify right \
	    -text $varPrefix-percent
	label $frame.label2 -text "% = " -width 4
	
	entry $frame.absolute -width 7 -justify right \
	    -text $varPrefix-absolute
	pack $frame.absolute $frame.label2 $frame.percent \
	    -side right -expand 0
	pack $frame.scale  -side left -expand 1 -fill x
	
	set frame $Frame.offset
	frame $frame
	checkbutton $frame.offset -variable $varPrefix-offset \
	    -text "Offset Intervals from Range Minimum" 
	pack $frame.offset -side left -expand 0 -fill none
	pack $frame -side top -expand 1 -fill x
	
	
	#######################################
	
	trace variable $varPrefix-min-percent w "updatedPercentage $varPrefix-range-first $varPrefix-range-second $varPrefix-min-absolute 1"
	
	trace variable $varPrefix-max-percent w "updatedPercentage $varPrefix-range-first $varPrefix-range-second $varPrefix-max-absolute 1"
	
	trace variable $varPrefix-min-absolute w "updatedAbsolute $varPrefix-range-first $varPrefix-range-second $varPrefix-min-percent 1"
	
	trace variable $varPrefix-max-absolute w "updatedAbsolute $varPrefix-range-first $varPrefix-range-second $varPrefix-max-percent 1"
	
	set Frame [addLabeledFrame $f "[axisName $axis] Variable Range"]
	set Frame2 $Frame
	
	set frame $Frame.begin
	frame $frame
	pack $frame -expand 1 -fill x
	label $frame.label -text "Min: " -anchor w -width 4
	pack $frame.label -side left -expand 0 -fill none
	
	scale $frame.scale -orient horizontal -from -100 \
	    -to 100 -showvalue 0 -resolution 1 \
	    -variable $varPrefix-min-percent
	
	entry $frame.percent -width 4 -text $varPrefix-min-percent \
	    -justify right
	label $frame.label2 -text "% = " -width 4
	
	entry $frame.absolute -width 7 -text $varPrefix-min-absolute \
	    -justify right
	pack $frame.absolute $frame.label2 $frame.percent \
	    -side right -expand 0
	pack $frame.scale  -side left -expand 1 -fill x
	
	set frame $Frame.end
	frame $frame
	pack $frame -expand 1 -fill x
	label $frame.label -text "Max: " -anchor w -width 4
	pack $frame.label -side left -expand 0 -fill none
	
	scale $frame.scale -orient horizontal -from 0 \
	    -to 200 -showvalue 0 -resolution 1 \
	    -variable $varPrefix-max-percent
	
	entry $frame.percent -width 4 -text $varPrefix-max-percent \
	    -justify right
	label $frame.label2 -text "% = " -width 4
	
	entry $frame.absolute -width 7 -text $varPrefix-max-absolute \
	    -justify right
	pack $frame.absolute $frame.label2 $frame.percent \
	    -side right -expand 0
	pack $frame.scale  -side left -expand 1 -fill x
	
	
	#######################################
	update idletasks
	
	set Frame [addLabeledFrame $f "Display Options - (Click to Hide)"]
	update idletasks
	set Hidden [join [lrange [split $Frame .] 0 end-1] .]
	button $f.but -text "Show Display Options" -command "showOptions $this"
	bind $Hidden.label <ButtonRelease> "hideOptions $this"
	global $this-displayFrames $this-displayButtons
	lappend $this-displayFrames $Hidden
	lappend $this-displayButtons $f.but

	pack forget $Hidden; 
	pack $f.but -side top -expand 0 -pady 0 -ipady 0;

	set frame $Frame.text
	frame $frame
	pack $frame -side top -expand 1 -fill x
	label $frame.label -anchor w -text "Label Text: " -width 13
	pack $frame.label -side left -expand 0
	entry $frame.entry -text $varPrefix-text -justify right
	pack $frame.entry -side left -expand 1 -fill x
	
	
	#	    displayRadios $Frame $varPrefix-minplane "Min Plane"
	#	    displayRadios $Frame $varPrefix-maxplane "Max Plane"
	
	displayRadios $Frame $varPrefix-lines "Lines"
	displayRadios $Frame $varPrefix-minticks "Min Ticks"
	displayRadios $Frame $varPrefix-maxticks "Max Ticks"
	displayRadios $Frame $varPrefix-minlabel "Min Label"
	displayRadios $Frame $varPrefix-maxlabel "Max Label"
	displayRadios $Frame $varPrefix-minvalue "Min Values"
	displayRadios $Frame $varPrefix-maxvalue "Max Values"
	
	
	labeledSlider $Frame "Line Width:" $varPrefix-width 0 10 0.1
	labeledSlider $Frame "Tick Angle:" $varPrefix-tickangle 0 360 1
	labeledSlider $Frame "Tick Tilt:" $varPrefix-ticktilt -90 90 1
	labeledSlider $Frame "Tick Size:" $varPrefix-ticksize 0 25 0.5
	labeledSlider $Frame "Label Angle:" $varPrefix-labelangle 0 360 1
	labeledSlider $Frame "Label Size:" $varPrefix-labelheight 0 50 0.1
	labeledSlider $Frame "Values Size:" $varPrefix-valuesize 0 30 0.1
	labeledSlider $Frame "Values Squash:" $varPrefix-valuesquash 0 2 0.1
	
	update idletasks
	
    }
    

    method build_plane_tab { tabs axes { text "" }} {

	set tab [$tabs add -label "[axisName $axes] Plane"]
	set tabframe $tab.frame
	frame $tabframe
	pack $tabframe -expand 1 -fill both
	iwidgets::tabnotebook $tabframe.tabs -width 330 -raiseselect true \
	    -tabpos n -backdrop gray -equaltabs 0 -bevelamount 5
	pack $tabframe.tabs -expand 1 -fill both

	set fromPre [linkVars [list $axes] lines divisions percent absolute offset range-frist range-second min-percent max-percent min-absolute max-absolute minplane maxplane lines minticks maxticks minlabel maxlabel minvalue maxvalue width tickangle ticktilt labelangle lableheight ticksize valuesquash valuesize]

	set frame [$tabframe.tabs add -label "Both"].frame
	frame $frame
	pack $frame -expand 1 -fill both

	build_plane_dir_ui $frame $fromPre $axes $axes

	foreach axis $axes {	   
	    set tab [$tabframe.tabs add -label "[axisName $axis] Axis $text"]
	    $tabframe.tabs view 0
	    set f $tab.frame
	    frame $f
	    pack $f -expand 1 -fill both
	    bind $f <ButtonRelease> "$this-c needexecute"

	    set planeName [join "Plane-$axes" ""]
	    set axisName [join "$axis-Axis" ""]
	    set varPrefix "$this-$planeName-$axisName"
	    
	    build_plane_dir_ui $f $varPrefix $axes $axis
	    
	}
    }

    method build_options_tab { tabs } {
	set options [$tabs add -label "Fonts"]
	pack $options -side top -expand 1 -fill both

	set valueframe $options.valuefont
	frame $valueframe -borderwidth 2 -relief groove
	pack $valueframe -side top -expand 0 -fill x

	set frame2 $valueframe.valuefont
	frame $frame2 -borderwidth 2
	pack $frame2 -side top -expand 0 -fill x

	label $frame2.label -text "Value Font: " -anchor w
	pack $frame2.label -side left -expand 0 -fill none
	menubutton $frame2.menu -indicatoron 1 -menu $frame2.menu.m \
	    -text "No Fonts in SCIRun/src/Fonts"
	pack $frame2.menu -side right -expand 1 -fill x
	menu $frame2.menu.m -tearoff 0
	$frame2.menu config -takefocus 1 -highlightthickness 2 \
	    -relief raised -bd 2 -anchor w

	labeledSlider $valueframe "Value Precision:" $this-precision 1 12 1 14
	labeledSlider $valueframe "Value Squash:" $this-squash 0 2 .1 14
	set rez [labeledSlider $valueframe "Value Resolution:" $this-valuerez 1 500 1 14]
	bind $rez <ButtonRelease> "$this-c valueFontChanged"

	set labelframe $options.labelfont
	frame $labelframe -borderwidth 2 -relief groove
	pack $labelframe -side top -expand 0 -fill x
	
	set frame $labelframe.labelfont
	frame $frame -borderwidth 2
	pack $frame -side top -expand 0 -fill x

	label $frame.label -text "Label Font: " -anchor w
	pack $frame.label -side left -expand 0 -fill none
	menubutton $frame.menu -indicatoron 1 -menu $frame.menu.m \
	    -text "No Fonts in SCIRun/src/Fonts"
	pack $frame.menu -side right -expand 1 -fill x
	menu $frame.menu.m -tearoff 0
	$frame.menu config -takefocus 1 -highlightthickness 2 \
	    -relief raised -bd 2 -anchor w

	set rez [labeledSlider $labelframe "Label Resolution:" $this-labelrez 1 500 1 14]
	bind $rez <Button> "$this-c labelFontChanged"

	global SCIRUN_SRCDIR
	set dir [file join $SCIRUN_SRCDIR Fonts]
	set files [glob -nocomplain -dir $dir *.ttf]
	set def 0
	set i 0
	foreach font [lsort $files] {
	    set filename [lindex [file split $font] end]
	    set filename [split $filename .]
	    set filename [lrange $filename 0 end-1]
	    set filename [join $filename .]
	    if [string equal SCIRun $filename] { set def $i }
	    $frame.menu.m add command -label $filename \
		-command "$frame.menu configure -text \"$filename\"; \
			  setGlobal $this-labelfont \"$font\"; \
                          $this-c labelFontChanged"

	    $frame2.menu.m add command -label $filename \
		-command "$frame2.menu configure -text \"$filename\"; \
		          setGlobal $this-valuefont \"$font\"; \
                          $this-c valueFontChanged"

	    incr i
	}

	if $i {
	    $frame.menu.m invoke $def
	    $frame2.menu.m invoke $def
	}

    }




    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }
        toplevel $w
	moveToCursor $w

 	iwidgets::tabnotebook $w.tabs -height 600 -raiseselect true -tabpos n \
	    -backdrop gray -equaltabs 0 -bevelamount 5 -borderwidth 0
	pack $w.tabs -expand 1 -fill both


	build_all_tab $w.tabs
	build_plane_tab $w.tabs "0 1"
	build_plane_tab $w.tabs "0 2"
	build_plane_tab $w.tabs "1 2"
	build_options_tab $w.tabs

	$w.tabs view 0

	makeSciButtonPanel $w $w $this

    }
}


proc showOptions {this} {
    global $this-displayFrames $this-displayButtons
    foreach button [set $this-displayButtons] {
	pack forget $button
    }

    foreach frame [set $this-displayFrames] {
	pack $frame -side top -fill x -expand 0 -pady 0 -ipady 0
    }
    
#    set w .ui[$this modname]
#    set x [lindex [split [wm geometry $w] x] 0]
#    wm geometry $w ${x}x800
}


proc hideOptions {this} {
    global $this-displayFrames $this-displayButtons
    foreach frame [set $this-displayFrames] {
	pack forget $frame
    }

    foreach button [set $this-displayButtons] {
	pack $button -side top -expand 0 -pady 0 -ipady 0
    }
#    set w .ui[$this modname]
#    set x [lindex [split [wm geometry $w] x] 0]
#    wm geometry $w ${x}x400

}


