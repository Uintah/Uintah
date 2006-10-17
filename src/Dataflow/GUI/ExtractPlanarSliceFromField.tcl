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


#    File   : ExtractPlanarSliceFromField.tcl
#    Author : Allen Sanderson
#             SCI Institute
#             University of Utah
#    Date   : March 2006
#
#    Copyright (C) 2006 SCI Group

# This GUI interface is for selecting an axis and index for sub sampling a
# topologically structured field.

itcl_class SCIRun_NewField_ExtractPlanarSliceFromField {
    inherit Module
    constructor {config} {
        set name ExtractPlanarSliceFromField
        set_defaults
    }

    method set_defaults {} {
	global $this-continuous
	set $this-continuous 0

	global $this-active-slice-value-selection-tab
	set $this-active-slice-value-selection-tab 0

	global power_app_command
	set    power_app_command ""
    }

    method ui {} {
	global $this-function
	global $this-active-slice-value-selection-tab

	set oldmeth [set $this-active-slice-value-selection-tab]

        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }


        toplevel $w

# Function definition
	iwidgets::labeledframe $w.func -labelpos nw \
	    -labeltext "Function Definition"
	set func [$w.func childsite]

	label $func.label -text \
	    "(Note: you can only use the values 'x', 'y', and 'z')"

	option add *textBackground white	
	iwidgets::scrolledtext $func.text -height 60 -hscrollmode dynamic

	$func.text insert end [set $this-function]

	pack $func.label -side top -anchor w -padx 5 -pady 5
	pack $func.text  -side top -e y -f both -padx 5

# Optional through zero checks.

	checkbutton $func.zero -text "Do through zero ambiguity checks" \
	    -relief flat -variable $this-zero-checks 

	pack $func.zero -side top -anchor w -pady 5

# Slice Value Selection Methods
	iwidgets::labeledframe $w.slice -labelpos nw \
	    -labeltext "Slice Value Selection"
	set isf [$w.slice childsite]

	global Color
	iwidgets::tabnotebook $isf.tabs -raiseselect true -height 200 \
	    -backdrop $Color(Basecolor)
	pack $isf.tabs -side top -fill x -expand 1


###### Slice Value using slider
	set sliceslider [$isf.tabs add -label "Slider" \
		     -command "set $this-active-slice-value-selection-tab 0"]

	scaleEntry2 $sliceslider.sliceval \
	    [set $this-slice-value-min] [set $this-slice-value-max] \
	     4c $this-slice-value $this-slice-value-typed

	iwidgets::labeledframe $sliceslider.opt -labelpos nw -labeltext "Options"
	set opt [$sliceslider.opt childsite]
	
	iwidgets::optionmenu $opt.update -labeltext "Update:" \
		-labelpos w -command "$this set_update_type $opt.update"
	$opt.update insert end "On Release" Manual Auto

	$opt.update select [set $this-update_type]

	global $this-update
	set $this-update $opt.update

	pack $opt.update -side top -anchor w -pady 25

	pack $sliceslider.sliceval $sliceslider.opt -side top -anchor w -fill x

###### Slice Value using quantity	
	set slicequant [$isf.tabs add -label "Quantity" \
		     -command "set $this-active-slice-value-selection-tab 1"]
	

###### Save the sliceval-quantity since the iwidget resets it
	global $this-slice-value-quantity
	set quantity [set $this-slice-value-quantity]
	iwidgets::spinint $slicequant.q -labeltext "Number of evenly-spaced slices: " \
	    -range {0 100} -step 1 \
	    -textvariable $this-slice-value-quantity \
	    -width 10 -fixed 10 -justify right
	
	$slicequant.q delete 0 end
	$slicequant.q insert 0 $quantity

	frame $slicequant.f
	label $slicequant.f.l -text "List of Slice Values:"
	entry $slicequant.f.e -width 40 -text $this-quantity-list -state disabled
	pack $slicequant.f.l $slicequant.f.e -side left -fill both -expand 1

	frame $slicequant.m
	radiobutton $slicequant.m.f -text "Field MinMax" \
		-variable $this-quantity-range -value "field" \
		-command "$this-c needexecute"
	radiobutton $slicequant.m.m -text "Manual" \
		-variable $this-quantity-range -value "manual" \
		-command "$this-c needexecute"

	frame $slicequant.m.t 
	label $slicequant.m.t.minl -text "Min"
	entry $slicequant.m.t.mine -width 6 -text $this-quantity-min
	label $slicequant.m.t.maxl -text "Max"
	entry $slicequant.m.t.maxe -width 6 -text $this-quantity-max
	bind $slicequant.m.t.mine <Return> "$this-c needexecute"
	bind $slicequant.m.t.maxe <Return> "$this-c needexecute"
	pack $slicequant.m.t.minl $slicequant.m.t.mine $slicequant.m.t.maxl $slicequant.m.t.maxe \
		-side left -fill x -expand 1

	pack $slicequant.m.f -side top -anchor w
	pack $slicequant.m.m $slicequant.m.t -side left -anchor w

	frame $slicequant.t
	radiobutton $slicequant.t.e -text "Exclusive" \
		-variable $this-quantity-clusive -value "exclusive" \
		-command "$this-c needexecute"
	radiobutton $slicequant.t.i -text "Inclusive" \
		-variable $this-quantity-clusive -value "inclusive" \
		-command "$this-c needexecute"

	pack $slicequant.t.e $slicequant.t.i -side left -anchor w

	pack $slicequant.q $slicequant.m $slicequant.t -side top -expand 1 -fill x -pady 5

	pack $slicequant.f -fill x

###### Slice Value using list
	set slicelist [$isf.tabs add -label "List" \
			 -command "set $this-active-slice-value-selection-tab 2"]

	
	frame $slicelist.f
	label $slicelist.f.l -text "List of Slice Values:"
	entry $slicelist.f.e -width 40 -text $this-slice-value-list
	bind $slicelist.f.e <Return> "$this-c needexecute"
	pack $slicelist.f.l $slicelist.f.e -side left -fill both -expand 1
	pack $slicelist.f -fill x


###### Slice Value using matrix
	set slicematrix [$isf.tabs add -label "Matrix" \
			   -command "set $this-active-slice-value-selection-tab 3"]

	frame $slicematrix.f
	label $slicematrix.f.l -text "List of Slice Values:"
	entry $slicematrix.f.e -width 40 -text $this-matrix-list -state disabled
	pack $slicematrix.f.l $slicematrix.f.e -side left -fill both -expand 1
	pack $slicematrix.f -fill x


# Pack the Slice Value Selection Tabs

	$isf.tabs view $oldmeth
	$isf.tabs configure -tabpos "n"

	pack $isf.tabs -side top

# Pack everything
	pack $w.func $w.slice -side top -anchor w -expand 1 -fill x

	makeSciButtonPanel $w $w $this
	moveToCursor $w
    }

    method update_text {} {
	set w .ui[modname]
        if {[winfo exists $w]} {
	    set func [$w.func childsite]
	    set $this-function [$func.text get 1.0 end]
        }
    }

    method set-slice-value {} {
	global $this-update

	set type [[set $this-update] get]
	if { $type == "On Release" } {
	    eval "$this-c needexecute"
	}
    }
    
    method set-slice-quant-list { vals } {
	global $this-quantity-list
	
	set $this-quantity-list $vals
    }
    
    method set-slice-matrix-list { vals } {
	global $this-matrix-list
	
	set $this-matrix-list $vals
    }

    method update_type_callback { name1 name2 op } {
        set tmp [set $this-update_type]
        if { $tmp == "on release" } { set $this-update_type "On Release" }
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
	set_min_max
    }

    method set_min_max { } {
	set w .ui[modname]
	global $this-slice-value-min
	global $this-slice-value-max

	set min [set $this-slice-value-min]
	set max [set $this-slice-value-max]

	if [ expr [winfo exists $w] ] {
	    set lg [expr floor( log10($max-$min) ) ]
	    set range [expr pow(10.0, $lg )]

	    set scale 1.0

	    if { $lg > 5.0 } {
		set scale [expr pow(10.0, $lg-5 )]
	    }

	    set win $w.slice.childsite.tabs.canvas.notebook.cs.page1.cs.sliceval

	    $win.l.s configure -from $min -to $max
	    $win.l.s configure -resolution [expr $range/(1.0e4*$scale)]
	    $win.l.s configure -tickinterval [expr ($max - $min)]

	    bind $win.r.e <Return> "$this manualSliderEntryReturn \
             $min $max $this-slice-value $this-slice-value-typed"
	    bind $win.r.e <KeyRelease> "$this manualSliderEntry \
             $min $max $this-slice-value $this-slice-value-typed"
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

	bind $win.l.s <ButtonRelease> "$this set-slice-value"

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
	global $this-update_type
        global $var_typed
	set $var_typed [set $var_slider]
	
	if { [set $this-continuous] == 1.0 } {
	    eval "$this-c needexecute"
	} elseif { [set $this-update_type] == "Auto" } {
	    set $this-continuous 1
	}
    }

    method manualSliderEntryReturn { start stop var_slider var_typed } {
	# Because the user has typed in a value and hit return, we know
	# they are done and if their value is not valid or within range,
	# we can change it to be either the old value, or the min or max
	# depending on what is appropriate.
  	if { ![string is integer [set $var_typed]] } {
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

	if { [set $this-update_type] == "On Release" ||
	     [set $this-update_type] == "Auto" } {
	    eval "$this-c needexecute"
	}
    }


    method manualSliderEntry { start stop var_slider var_typed } {
	# Evaluate as the user types in an sliceval but never change the value
	# they are typing in because they might not be done. Only update the
	# actual sliceval when user has typed in a double and it is within range.
	
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

