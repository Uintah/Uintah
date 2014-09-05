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
	global $this-quantity-list
	global $this-quantity-range
	global $this-quantity-clusive
	global $this-quantity-min
	global $this-quantity-max
	global $this-isoval-list
	global $this-matrix-list
	global $this-active-isoval-selection-tab
	global $this-continuous
	global $this-extract-from-new-field
	global $this-algorithm
	global $this-build_trisurf
	global $this-build_geom
	global $this-np
	global $this-active_tab
	global $this-update_type
	global $this-color-r
	global $this-color-g
	global $this-color-b

	set $this-isoval-min 0.0
	set $this-isoval-max 99.0
	set $this-isoval 0.0
	set $this-isoval-typed 0
	set $this-isoval-quantity 1
	set $this-quantity-range "field"
	set $this-quantity-clusive "exclusive"
	set $this-quantity-min 0
	set $this-quantity-max 100
	set $this-quantity-list ""
	set $this-isoval-list "0.0 1.0 2.0 3.0"
	set $this-matrix-list "No matrix present - execution needed."
	set $this-active-isoval-selection-tab 0
	set $this-continuous 0
	set $this-extract-from-new-field 1
	set $this-algorithm 0
	set $this-build_trisurf 1
	set $this-build_geom 1
	set $this-np 1
	set $this-update_type "On Release"
	set $this-color-r 0.4
	set $this-color-g 0.2
	set $this-color-b 0.9

	trace variable $this-active_tab w "$this switch_to_active_tab"
	trace variable $this-update_type w "$this update_type_callback"
	trace variable $this-isoval-max w "$this update_minmax_callback"

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
	     SciRaise $window.color
	     return
	 } else {
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
         $this-c needexecute
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
	set window .ui[modname]
	if {[winfo exists $window]} {
	    set mf [$window.f.meth childsite]
	    $mf.tabs view [set $this-active_tab]
	}
    }

    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
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
	global Color
	iwidgets::tabnotebook $isf.tabs -raiseselect true -height 200 \
	    -backdrop $Color(Basecolor)
	pack $isf.tabs -side top -fill x -expand 1
	pack $w.f.iso -side top -fill x -expand 1


	###### Iso Value using slider
	set isoslider [$isf.tabs add -label "Slider" \
		     -command "set $this-active-isoval-selection-tab 0"]

	scaleEntry2 $isoslider.isoval \
	    [set $this-isoval-min] [set $this-isoval-max] \
	     4c $this-isoval $this-isoval-typed

	pack $isoslider.isoval  -fill x


	###### Iso Value using quantity	
	set isoquant [$isf.tabs add -label "Quantity" \
		     -command "set $this-active-isoval-selection-tab 1"]
	
	###### Save the isoval-quantity since the iwidget resets it
	global $this-isoval-quantity
	set quantity [set $this-isoval-quantity]
	iwidgets::spinint $isoquant.q -labeltext "Number of evenly-spaced isovals: " \
	    -range {0 100} -step 1 \
	    -textvariable $this-isoval-quantity \
	    -width 10 -fixed 10 -justify right
	
	$isoquant.q delete 0 end
	$isoquant.q insert 0 $quantity

	frame $isoquant.f
	label $isoquant.f.l -text "List of Isovals:"
	entry $isoquant.f.e -width 40 -text $this-quantity-list -state disabled
	pack $isoquant.f.l $isoquant.f.e -side left -fill both -expand 1

	frame $isoquant.m
	radiobutton $isoquant.m.c -text "ColorMap MinMax" \
		-variable $this-quantity-range -value "colormap" \
		-command "$this-c needexecute"
	radiobutton $isoquant.m.f -text "Field MinMax" \
		-variable $this-quantity-range -value "field" \
		-command "$this-c needexecute"
	radiobutton $isoquant.m.m -text "Manual" \
		-variable $this-quantity-range -value "manual" \
		-command "$this-c needexecute"

	frame $isoquant.m.t 
	label $isoquant.m.t.minl -text "Min"
	entry $isoquant.m.t.mine -width 6 -text $this-quantity-min
	label $isoquant.m.t.maxl -text "Max"
	entry $isoquant.m.t.maxe -width 6 -text $this-quantity-max
	bind $isoquant.m.t.mine <Return> "$this-c needexecute"
	bind $isoquant.m.t.maxe <Return> "$this-c needexecute"
	pack $isoquant.m.t.minl $isoquant.m.t.mine $isoquant.m.t.maxl $isoquant.m.t.maxe \
		-side left -fill x -expand 1

	pack $isoquant.m.c $isoquant.m.f -side top -anchor w
	pack $isoquant.m.m $isoquant.m.t -side left -anchor w

	frame $isoquant.t
	radiobutton $isoquant.t.e -text "Exclusive" \
		-variable $this-quantity-clusive -value "exclusive" \
		-command "$this-c needexecute"
	radiobutton $isoquant.t.i -text "Inclusive" \
		-variable $this-quantity-clusive -value "inclusive" \
		-command "$this-c needexecute"

	pack $isoquant.t.e $isoquant.t.i -side left -anchor w

	pack $isoquant.q $isoquant.m $isoquant.t -side top -expand 1 -fill x -pady 5

	pack $isoquant.f -fill x

	###### Iso Value using list
	set isolist [$isf.tabs add -label "List" \
			 -command "set $this-active-isoval-selection-tab 2"]

	
	frame $isolist.f
	label $isolist.f.l -text "List of Isovals:"
	entry $isolist.f.e -width 40 -text $this-isoval-list
	bind $isolist.f.e <Return> "$this-c needexecute"
	pack $isolist.f.l $isolist.f.e -side left -fill both -expand 1
	pack $isolist.f -fill x


	###### Iso Value using matrix
	set isomatrix [$isf.tabs add -label "Matrix" \
			   -command "set $this-active-isoval-selection-tab 3"]

	frame $isomatrix.f
	label $isomatrix.f.l -text "List of Isovals:"
	entry $isomatrix.f.e -width 40 -text $this-matrix-list -state disabled
	pack $isomatrix.f.l $isomatrix.f.e -side left -fill both -expand 1
	pack $isomatrix.f -fill x


	# Pack the Iso Value Selection Tabs

	$isf.tabs view $oldmeth
	$isf.tabs configure -tabpos "n"

	pack $isf.tabs -side top
	pack $w.f.iso -side top

	#  Options

	iwidgets::labeledframe $w.f.opt -labelpos nw -labeltext "Options"
	set opt [$w.f.opt childsite]
	
	iwidgets::optionmenu $opt.update -labeltext "Update:" \
		-labelpos w -command "$this set_update_type $opt.update"
	$opt.update insert end "On Release" Manual Auto
	$opt.update select [set $this-update_type]

	global $this-update
	set $this-update $opt.update

	global $this-build_trisurf
	checkbutton $opt.buildsurf -text "Build Output Field" \
		-variable $this-build_trisurf

	global $this-build_geom
	checkbutton $opt.buildgeom -text "Build Output Geometry" \
		-variable $this-build_geom

	checkbutton $opt.aefnf -text "Auto Extract from New Field" \
		-relief flat -variable $this-extract-from-new-field 


	pack $opt.update $opt.aefnf $opt.buildsurf $opt.buildgeom \
	    -side top -anchor w

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
	if { $type == "On Release" } {
	    eval "$this-c needexecute"
	}
    }
    
    method set-isoquant-list { vals } {
	global $this-quantity-list
	
	set $this-quantity-list $vals
    }
    
    method set-isomatrix-list { vals } {
	global $this-matrix-list
	
	set $this-matrix-list $vals
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

    method update_type_callback { name1 name2 op } {
	set window .ui[modname]
	if {[winfo exists $window]} {
	    set opt [$window.f.opt childsite]
	    $opt.update select [set $this-update_type]
	}
    }

    method set_update_type { w } {
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

    method update_minmax_callback { name1 name2 op } {
	global $this-isoval-min
	global $this-isoval-max

	set_min_max [set $this-isoval-min] [set $this-isoval-max]
    }

    method set_min_max { min max } {
	set w .ui[modname]
	global $this-isoval-min
	global $this-isoval-max

	set $this-isoval-min $min
	set $this-isoval-max $max

	if [ expr [winfo exists $w] ] {
	    set lg [expr floor( log10($max-$min) ) ]
	    set range [expr pow(10.0, $lg )]

	    set scale 1.0

	    if { $lg > 5.0 } {
		set scale [expr pow(10.0, $lg-5 )]
	    }

	    set win $w.f.iso.childsite.tabs.canvas.notebook.cs.page1.cs.isoval

	    $win.l.s configure -from $min -to $max
	    $win.l.s configure -resolution [expr $range/(1.0e4*$scale)]
	    $win.l.s configure -tickinterval [expr ($max - $min)]

	    bind $win.r.e <Return> "$this manualSliderEntryReturn \
             $min $max $this-isoval $this-isoval-typed"
	    bind $win.r.e <KeyRelease> "$this manualSliderEntry \
             $min $max $this-isoval $this-isoval-typed"
	}
    }

    method scaleEntry2 { win start stop length var_slider var_typed } {
	frame $win 

	frame $win.l
	frame $win.r
	
	set lg [expr floor( log10($stop-$start) ) ]
	set range [expr pow(10.0, $lg )]

	set scale 1.0

 	if { $lg > 5.0 } {
	    set scale [expr pow(10.0, $lg-5 )]
	}

	scale $win.l.s \
	    -from $start -to $stop \
	    -length $length \
	    -variable $var_slider -orient horizontal -showvalue false \
	    -command "$this updateSliderEntry $var_slider $var_typed" \
	    -resolution [expr $range/(1.0e4*$scale)] \
	    -tickinterval [expr ($stop - $start)]

	entry $win.r.e -width 7 -text $var_typed

	bind $win.l.s <ButtonRelease> "$this set-isoval"

	bind $win.r.e <Return> "$this manualSliderEntryReturn \
             $start $stop $var_slider $var_typed"
	bind $win.r.e <KeyRelease> "$this manualSliderEntry \
             $start $stop $var_slider $var_typed"

	pack $win.l.s -side top -expand 1 -fill x -padx 5
	pack $win.r.e -side top -padx 5 -pady 3
	pack $win.l -side left -expand 1 -fill x
	pack $win.r -side right -fill y
    }

    method updateSliderEntry {var_slider var_typed someUknownVar} {
	global $this-continuous
        global $var_slider
        global $var_typed
	set $var_typed [set $var_slider]

	if { [set $this-continuous] == 1.0 } {
	    eval "$this-c needexecute"
	} elseif { [set $this-update_type] == "Auto" } {
	    set $this-continuous 1
	}
    }

    method manualSliderEntryReturn { start stop var_slider var_typed } {
	# Since the user has typed in a value and hit return, we know
	# they are done and if their value is not valid or within range,
	# we can change it to be either the old value, or the min or max
	# depending on what is appropriate.
  	if { ![string is double [set $var_typed]] } {
  	    set $var_typed [set $var_slider] 
  	}

	if {[set $var_typed] < $start} {
	    set $var_typed $start
	} elseif {[set $var_typed] > $stop} {
	    set $var_typed $stop
	}

	# Force the update to be manual
	global $this-continuous
	set continuous [set $this-continuous]
	
	set $this-continuous 0
	
	set $var_slider [set $var_typed]
	
	set $this-continuous $continuous
    }
    
    method manualSliderEntry { start stop var_slider var_typed } {
	# Evaluate as the user types in an isoval but never change the value
	# they are typing in because they might not be done. Only update the
	# actual isoval when user has typed in a double and it is within range.
	
 	set var_new [set $var_slider]

 	# only update the value if it evaluates to a double 
	# and is within range
 	if {[string is double [set $var_typed]] && 
 	    $start <= [set $var_typed] && 
 	    [set $var_typed] <= $stop} {
 	    set var_new [set $var_typed]
 	}
	
	# Force the update to be manual
  	global $this-continuous
  	set continuous [set $this-continuous]
	
  	set $this-continuous 0
	
  	set $var_slider $var_new
	
  	set $this-continuous $continuous
    }
}
