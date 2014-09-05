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


catch {rename Isosurface ""}

package require Iwidgets 3.0   

itcl_class SCIRun_Visualization_Isosurface {
    inherit Module
    
    constructor {config} {
	set name Isosurface
	set_defaults
    }
    
    method set_defaults {} {
	global $this-isoval-min 
	global $this-isoval-max 
	global $this-isoval
	global $this-isoval-typed
	global $this-isoval-quantity
	global $this-active-isoval-selection-tab
	global $this-continuous
	global $this-extract-from-new-field
	global $this-algorithm
	global $this-type
	global $this-gen
	global $this-build_trisurf
	global $this-np
	global $this-active_tab
	global $this-update_type
	global $this-color-r
	global $this-color-g
	global $this-color-b

	set $this-isoval-min 0
	set $this-isoval-max 4095
	set $this-isoval 0
	set $this-isoval-typed 0
	set $this-isoval-quantity 1
	set $this-active-isoval-selection-tab 0
	set $this-continuous 0
	set $this-extract-from-new-field 1
	set $this-algorithm 0
	set $this-type ""
	set $this-gen 0
	set $this-build_trisurf 0
	set $this-np 1
	set $this-active_tab "MC"
	set $this-update_type "on release"
	set $this-color-r 0.4
	set $this-color-g 0.2
	set $this-color-b 0.9
	trace variable $this-active_tab w "$this switch_to_active_tab"
	trace variable $this-update_type w "$this set_update_type"

	# SAGE vars
	global $this-visibility $this-value $this-scan
	global $this-bbox
	global $this-cutoff_depth 
	global $this-reduce
	global $this-all
	global $this-rebuild
	global $this-min_size
	global $this-poll

	set $this-visiblilty 0
	set $this-value 1
	set $this-scan 1
	set $this-bbox 1
	set $this-reduce 1
	set $this-all 0
	set $this-rebuild 0
	set $this-min_size 1
	set $this-poll 0

    }

    method raiseColor {swatch color} {
	 global $color
	 set window .ui[modname]
	 if {[winfo exists $window.color]} {
	     raise $window.color
	     return;
	 } else {
	     toplevel $window.color
	     makeColorPicker $window.color $color \
		     "$this setColor $swatch $color" \
		     "destroy $window.color"
	 }
    }

    method setColor {swatch color} {
	 global $color
	 global $color-r
	 global $color-g
	 global $color-b
	 set ir [expr int([set $color-r] * 65535)]
	 set ig [expr int([set $color-g] * 65535)]
	 set ib [expr int([set $color-b] * 65535)]

	 set window .ui[modname]
	 $swatch config -background [format #%04x%04x%04x $ir $ig $ib]
	 destroy $window.color
    }

    method addColorSelection {frame color} {
	 #add node color picking 
	 global $color
	 global $color-r
	 global $color-g
	 global $color-b
	 set ir [expr int([set $color-r] * 65535)]
	 set ig [expr int([set $color-g] * 65535)]
	 set ib [expr int([set $color-b] * 65535)]
	 
	 frame $frame.colorFrame
	 frame $frame.colorFrame.swatch -relief ridge -borderwidth \
		 4 -height 0.8c -width 1.0c \
		 -background [format #%04x%04x%04x $ir $ig $ib]
	 
	 set cmmd "$this raiseColor $frame.colorFrame.swatch $color"
	 button $frame.colorFrame.set_color \
		 -text "Change Color" -command $cmmd
	 
	 #pack the node color frame
	 pack $frame.colorFrame.set_color $frame.colorFrame.swatch -side left
	 pack $frame.colorFrame -side left

    }

    method switch_to_active_tab {name1 name2 op} {
	#puts stdout "switching"
	set window .ui[modname]
	if {[winfo exists $window]} {
	    set mf [$window.f.meth childsite]
	    $mf.tabs view [set $this-active_tab]
	}
    }

    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	
	toplevel $w
	frame $w.f 
	pack $w.f -padx 2 -pady 2 -expand 1 -fill x
	set n "$this-c needexecute "

	set oldmeth [set $this-active-isoval-selection-tab]

	# Iso Value Selection Methods
	iwidgets::labeledframe $w.f.iso -labelpos nw -labeltext "Isovalue Selection Methods"
	set isf [$w.f.iso childsite]

	iwidgets::tabnotebook $isf.tabs -raiseselect true
	pack $isf.tabs -side top -fill x -expand 1
	pack $w.f.iso -side top -fill x -expand 1

	# Iso Value using slider

	set sel [$isf.tabs add -label "Slider" -command "set $this-active-isoval-selection-tab 0"]

	scaleEntry2 $sel.isoval \
	    [set $this-isoval-min] [set $this-isoval-max] \
	    5c $this-isoval $this-isoval2

#	scale $sel.isoval -label "Isovalue:" \
		-variable $this-isoval \
		-from [set $this-isoval-min] -to [set $this-isoval-max] \
		-length 5c \
		-showvalue true \
		-orient horizontal  \
		-digits 5 \
		-resolution 0.001 \
	        -command "$this change-isoval"

	
	button $sel.extract -text "Extract" -command "$this-c needexecute"

	pack $sel.isoval  -fill x
	pack $sel.extract -side top -expand 1

	# Iso Value using text-entry
	
#	set sel [$isf.tabs add -label "Text" -command "set $this-active-isoval-selection-tab 1"]
	
#	frame $sel.f
#	label $sel.f.l -text "Type isovalue:"
#	entry $sel.f.e -width 20 -text $this-isoval-typed
#	bind $sel.f.e <Return> "$this-c needexecute"
#	pack $sel.f.l $sel.f.e -side left -fill x -expand 1
#	button $sel.extract -text "Extract" -command "$this-c needexecute"
#	pack $sel.f $sel.extract -side top -expand 1

	# Iso Value using quantity
	
	set sel [$isf.tabs add -label "Quantity" -command "set $this-active-isoval-selection-tab 1"]
	
	frame $sel.f
	label $sel.f.l -text "Number of evenly-spaced isovals:"
	entry $sel.f.e -width 20 -text $this-isoval-quantity
	bind $sel.f.e <Return> "$this-c needexecute"
	pack $sel.f.l $sel.f.e -side left -fill x -expand 1
	button $sel.extract -text "Extract" -command "$this-c needexecute"
	pack $sel.f $sel.extract -side top -expand 1

	# Pack the Iso Value Selection Tabs

	$isf.tabs view $oldmeth
	$isf.tabs configure -tabpos "n"

	pack $isf.tabs -side top
	pack $w.f.iso -side top

	#  Info
	
	global $this-type
	global $this-gen

	iwidgets::labeledframe $w.f.info -labelpos nw -labeltext "Info"
	set info [$w.f.info childsite]
	
	label $info.type_label -text "File Type: " 
	label $info.type -text [set $this-type]
	label $info.gen_label -text "Generation: "
	label $info.gen -text [set $this-gen]

	pack $info.type_label $info.type $info.gen_label $info.gen -side left
	pack $w.f.info -side top -fill x -expand 1

	#  Options

	iwidgets::labeledframe $w.f.opt -labelpos nw -labeltext "Options"
	set opt [$w.f.opt childsite]
	
	iwidgets::optionmenu $opt.update -labeltext "Update:" \
		-labelpos w -command "$this update-type $opt.update"
	$opt.update insert end "on release" Manual Auto
	$opt.update select [set $this-update_type]

	global $this-update
	set $this-update $opt.update

	global $this-build_trisurf
	checkbutton $opt.buildsurf -text "Build Output Field" \
		-variable $this-build_trisurf

	checkbutton $opt.aefnf -text "Auto Extract from New Field" \
		-relief flat -variable $this-extract-from-new-field 

	pack $opt.update $opt.aefnf $opt.buildsurf -side top -anchor w
	pack $w.f.opt -side top -fill x -expand 1

	# Color

	iwidgets::labeledframe $w.f.color -labelpos nw -labeltext "Default Color"
	set color [$w.f.color childsite]
	addColorSelection $color $this-color
	pack $w.f.color -side top -fill x -expand 1


	#  Methods
	iwidgets::labeledframe $w.f.meth -labelpos nw -labeltext "Methods"
	set mf [$w.f.meth childsite]
	
	iwidgets::tabnotebook  $mf.tabs -raiseselect true 
	#-fill both
	pack $mf.tabs -side top

	#  Method:

	set alg [$mf.tabs add -label "MC" -command "$this select-alg 0"]
	
        scale $alg.np -label "Threads:" \
		-variable $this-np \
		-from 1 -to 32 \
		-showvalue true \
		-orient horizontal
	
        pack $alg.np -side left -fill x

	set alg [$mf.tabs add -label "NOISE"  -command "$this select-alg 1"]

	$mf.tabs view [set $this-active_tab]
	$mf.tabs configure -tabpos "n"

	
	pack $mf.tabs -side top -fill x -expand 1
	pack $w.f.meth -side top -fill x -expand 1
    }

    method change-isoval { n } {
	global $this-continuous

	if { [set $this-continuous] == 1.0 } {
	    eval "$this-c needexecute"
	}

	if { [set $this-update_type] == "Auto" } {
	    set $this-continuous 1
	} else {
	    set $this-continuous 0
	}
    }
    
    method set-isoval {} {
	global $this-update

	set type [[set $this-update] get]
	if { $type == "on release" } {
	    eval "$this-c needexecute"
	}
    }
    
    method orient { tab page { val 4 }} {
	global $page
	global $tab
	
	$tab.tabs configure -tabpos [$page.orient get]
    }

    method select-alg { alg } {
	global $this-algorithm
	global $this-active_tab

	if { $alg == 0 } {
	    set $this-active_tab "MC"
	} else {
	    set $this-active_tab "NOISE"
	}
	if { [set $this-algorithm] != $alg } {
	    set $this-algorithm $alg
	    if { [set $this-continuous] == 1.0 } {
		eval "$this-c needexecute"
	    }
	}
    }

    method set_update_type { name1 name2 op } {
	set window .ui[modname]
	if {[winfo exists $window]} {
	    set opt [$window.f.opt childsite]
	    $opt.update select [set $this-update_type]
	}
    }

    method update-type { w } {
	global $w
	global $this-continuous
	global $this-update_type

	set $this-update_type [$w get]
	if { [set $this-update_type] == "Auto" } {
	    set $this-continuous 1
	} else {
	    set $this-continuous 0
	}
    }

    method set_info { type generation } {
	global $this-type
	global $this-gen

	set $this-type $type
	set $this-gen $generation

	set w .ui[modname]    
	if [ expr [winfo exists $w] ] {
	    set info [$w.f.info childsite]
	    
	    $info.type configure -text $type 
	    $info.gen  configure -text $generation
	}
    }

    method set_minmax {min max} {
	set w .ui[modname]

	global $this-isoval-min $this-isoval-max
	set $this-isoval-min $min
	set $this-isoval-max $max
	if [ expr [winfo exists $w] ] {
	    $w.f.iso.childsite.tabs.canvas.notebook.cs.page1.cs.isoval.s \
		configure -from $min -to $max
	    
	    bind $w.f.iso.childsite.tabs.canvas.notebook.cs.page1.cs.isoval.e \
		<Return> "$this manualSliderEntry $min $max $this-isoval $this-isoval2"
	}
    }

    method scaleEntry2 { win start stop length var1 var2 } {
	frame $win 
#	pack $win -side top -padx 5

	scale $win.s -from $start -to $stop -length $length \
	    -variable $var1 -orient horizontal -showvalue false \
	    -command "$this updateSliderEntry $var1 $var2" \
	    -resolution 0.001

	entry $win.e -width 10 -text $var2

	bind $win.s <ButtonRelease> "$this set-isoval"

	bind $win.e <Return> "$this manualSliderEntry $start $stop $var1 $var2"

	pack $win.s -side left
	pack $win.e -side bottom -padx 5
    }

    method updateSliderEntry {var1 var2 someUknownVar} {
	set $var2 [set $var1]

	change-isoval [set $var1]
    }

    method manualSliderEntry { start stop var1 var2 } {

	puts stdout $start
	puts stdout $stop
	puts stdout $var1
	puts stdout $var2

	if { [set $var2] < $start } {
	    set $var2 $start
	}
	
	if { [set $var2] > $stop } {
	    set $var2 $stop }
	
	set $var1 [set $var2]

	puts stdout [set $var1]
	puts stdout [set $var2]

	change-isoval [set $var1]
    }
}
