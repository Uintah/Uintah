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
	global $this-quantity-range
	global $this-quantity-min
	global $this-quantity-max
	global $this-isoval-list
	global $this-active-isoval-selection-tab
	global $this-continuous
	global $this-extract-from-new-field
	global $this-algorithm
	global $this-build_trisurf
	global $this-np
	global $this-active_tab
	global $this-update_type
	global $this-color-r
	global $this-color-g
	global $this-color-b

	set $this-isoval-min 0
	set $this-isoval-max 99
	set $this-isoval 0
	set $this-isoval-typed 0
	set $this-isoval-quantity 1
	set $this-quantity-range "field"
	set $this-quantity-min 0
	set $this-quantity-max 100
	set $this-isoval-list "0.0 1.0 2.0 3.0"
	set $this-active-isoval-selection-tab 0
	set $this-continuous 0
	set $this-extract-from-new-field 1
	set $this-algorithm 0
	set $this-build_trisurf 1
	set $this-np 1
	set $this-update_type "on release"
	set $this-color-r 0.4
	set $this-color-g 0.2
	set $this-color-b 0.9

	trace variable $this-active_tab w "$this switch_to_active_tab"
	trace variable $this-update_type w "$this set_update_type"
	trace variable $this-isoval-max w "$this set_minmax_callback"

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
	     wm deiconify $window.color
	     raise $window.color
	     return
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
		 -text "Default Color" -command $cmmd
	 
	 #pack the node color frame
	 pack $frame.colorFrame.set_color $frame.colorFrame.swatch -side left
	 pack $frame.colorFrame -side left -padx 3 -pady 3

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
	    wm deiconify $w
	    raise $w
	    return
	}
	
	toplevel $w
	frame $w.f 
	pack $w.f -padx 2 -pady 2 -expand 1 -fill x
	set n "$this-c needexecute "

	set oldmeth [set $this-active-isoval-selection-tab]

	# Iso Value Selection Methods
	iwidgets::labeledframe $w.f.iso -labelpos nw -labeltext "Isovalue Selection"
	set isf [$w.f.iso childsite]

	iwidgets::tabnotebook $isf.tabs -raiseselect true -height 130
	pack $isf.tabs -side top -fill x -expand 1
	pack $w.f.iso -side top -fill x -expand 1

	# Iso Value using slider

	set sel [$isf.tabs add -label "Slider" -command "set $this-active-isoval-selection-tab 0"]

	scaleEntry2 $sel.isoval \
	    [set $this-isoval-min] [set $this-isoval-max] \
	     4c $this-isoval $this-isoval2

	pack $sel.isoval  -fill x

	# Iso Value using quantity
	
	set sel [$isf.tabs add -label "Quantity" -command "set $this-active-isoval-selection-tab 1"]
	
	frame $sel.f
	label $sel.f.l -text "Number of evenly-spaced isovals:"
	entry $sel.f.e -width 20 -text $this-isoval-quantity
	bind $sel.f.e <Return> "$this-c needexecute"
	pack $sel.f.l $sel.f.e -side left -fill x -expand 1
	frame $sel.m
	radiobutton $sel.m.c -text "ColorMap MinMax" \
		-variable $this-quantity-range -value "colormap" \
		-command "$this-c needexecute"
	radiobutton $sel.m.f -text "Field MinMax" \
		-variable $this-quantity-range -value "field" \
		-command "$this-c needexecute"
	radiobutton $sel.m.m -text "Manual" \
		-variable $this-quantity-range -value "manual" \
		-command "$this-c needexecute"

	frame $sel.m.t 
	label $sel.m.t.minl -text "Min"
	entry $sel.m.t.mine -width 6 -text $this-quantity-min
	label $sel.m.t.maxl -text "Max"
	entry $sel.m.t.maxe -width 6 -text $this-quantity-max
	bind $sel.m.t.mine <Return> "$this-c needexecute"
	bind $sel.m.t.maxe <Return> "$this-c needexecute"
	pack $sel.m.t.minl $sel.m.t.mine $sel.m.t.maxl $sel.m.t.maxe \
		-side left -fill x -expand 1

	pack $sel.m.c $sel.m.f -side top -anchor w
	pack $sel.m.m $sel.m.t -side left -anchor w

	pack $sel.f $sel.m -side top -expand 1 -fill x -pady 5

	# Iso Value using list
	
	set isolist [$isf.tabs add -label "List" -command "set $this-active-isoval-selection-tab 2"]
	
	frame $isolist.f
	label $isolist.f.l -text "List of Isovals:"
	entry $isolist.f.e -width 40 -text $this-isoval-list
	bind $isolist.f.e <Return> "$this-c needexecute"
	pack $isolist.f.l $isolist.f.e -side left -fill both -expand 1
	pack $isolist.f -fill x

	# Pack the Iso Value Selection Tabs

	$isf.tabs view $oldmeth
	$isf.tabs configure -tabpos "n"

	pack $isf.tabs -side top
	pack $w.f.iso -side top

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

	addColorSelection $opt $this-color

	pack $w.f.opt -side top -fill x -expand 1

	#  Methods
	iwidgets::labeledframe $w.f.meth -labelpos nw \
	    -labeltext "Computation Method"
	set mf [$w.f.meth childsite]

	frame $mf.mc
	radiobutton $mf.mc.r -text "Marching Cubes" \
	    -variable $this-algorithm -value 0 -command "$this select-alg"

	label $mf.mc.lthreads -text "Threads:"
	entry $mf.mc.ethreads -textvar $this-np -width 3

	pack $mf.mc.r -side left
	pack $mf.mc.ethreads $mf.mc.lthreads -side right -padx 5

	bind $mf.mc.ethreads <Return> "$this select-alg"

	radiobutton $mf.noise -text "NOISE" \
	    -variable $this-algorithm -value 1 -command "$this select-alg"

	pack $mf.mc -side top -anchor w -expand y -fill x
	pack $mf.noise -side top -anchor w

	pack $w.f.meth -side top -fill x -expand 1

	makeSciButtonPanel $w $w $this
	moveToCursor $w
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

    method select-alg {} {
	if { [set $this-update] != "Manual" } {
	    eval "$this-c needexecute"
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

    method set_minmax_callback {varname varele varop} {
	set w .ui[modname]
	global $this-isoval-min $this-isoval-max
	set min [set $this-isoval-min]
	set max [set $this-isoval-max]
	if [ expr [winfo exists $w] ] {
          $w.f.iso.childsite.tabs.canvas.notebook.cs.page1.cs.isoval.l.s \
		  configure -from $min -to $max
          $w.f.iso.childsite.tabs.canvas.notebook.cs.page1.cs.isoval.l.s \
	          configure -resolution [expr ($max - $min)/10000.]
	  $w.f.iso.childsite.tabs.canvas.notebook.cs.page1.cs.isoval.l.s \
		  configure -tickinterval [expr ($max - $min)/3.001]
	  bind $w.f.iso.childsite.tabs.canvas.notebook.cs.page1.cs.isoval.r.e \
		  <Return> "$this manualSliderEntry $min $max $this-isoval $this-isoval2"
	}

	# have to set this value here -- it needs to be set after
	#  the first call to set_minmax, otherwise a net that has 
	#  update_type set to "Auto" will execute immediately... rather
	#  than waiting for a user-generated execute event
	# (the configure commands just above are what cause this)
	global $this-continuous
	global $this-update_type
	if { [set $this-update_type] == "Auto" } {
	    set $this-continuous 1
	} else {
	    set $this-continuous 0
	}
    }

    method scaleEntry2 { win start stop length var1 var2 } {
	frame $win 
#	pack $win -side top -padx 5

	frame $win.l
	frame $win.r
	scale $win.l.s -from $start -to $stop -length $length \
	    -tickinterval [expr ($stop - $start)/3.001] \
	    -variable $var1 -orient horizontal -showvalue false \
	    -command "$this updateSliderEntry $var1 $var2" \
	    -resolution [expr ($stop - $start)/10000.]

	entry $win.r.e -width 7 -text $var2

	bind $win.l.s <ButtonRelease> "$this set-isoval"

	bind $win.r.e <Return> "$this manualSliderEntry $start $stop $var1 $var2"

	pack $win.l.s -side top -expand 1 -fill x -padx 5
	pack $win.r.e -side top -padx 5 -pady 3
	pack $win.l -side left -expand 1 -fill x
	pack $win.r -side right -fill y
    }

    method updateSliderEntry {var1 var2 someUknownVar} {
	set $var2 [set $var1]
	set n [set $var1]

	global $this-continuous

	if { [set $this-continuous] == 1.0 } {
	    eval "$this-c needexecute"
	}
    }
    
    method manualSliderEntry { start stop var1 var2 } {
	if { [set $var2] < $start } {
	    set $var2 $start
	}
	if { [set $var2] > $stop } {
	    set $var2 $stop 
	}
	set $var1 [set $var2]
	eval "$this-c needexecute"
    }
}
