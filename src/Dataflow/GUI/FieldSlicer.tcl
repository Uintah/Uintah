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


#    File   : FieldSlicer.tcl
#    Author : Michael Callahan &&
#             Allen Sanderson
#             SCI Institute
#             University of Utah
#    Date   : March 2006
#
#    Copyright (C) 2006 SCI Group

# This GUI interface is for selecting an axis and index for sub sampling a
# topologically structured field.

itcl_class SCIRun_FieldsCreate_FieldSlicer {
    inherit Module
    constructor {config} {
        set name FieldSlicer
        set_defaults
    }

    method set_defaults {} {
	global power_app_command
	set    power_app_command ""

	for {set i 0} {$i < 3} {incr i 1} {
	    if { $i == 0 } {
		set index i
	    } elseif { $i == 1 } {
		set index j
	    } elseif { $i == 2 } {
		set index k
	    }

	    global $this-dim-$index
	    trace variable $this-dim-$index w "$this update_set_size_callback"
	}

	global $this-dims
	trace variable $this-dims w "$this update_set_size_callback"

	global $this-update_type
	trace variable $this-update_type w "$this update_type_callback"
    }

    method ui {} {

	global $this-axis
	global $this-dims

        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }

        toplevel $w

	frame $w.main

	frame $w.main.l
	label $w.main.l.direction -text "Direction"  \
	    -width 9 -anchor w -just left
	label $w.main.l.index     -text "Slice Node" \
	    -width 11 -anchor w -just left

	pack $w.main.l.direction -side left
	pack $w.main.l.index     -side left -padx 75

	for {set i 0} {$i < 3} {incr i 1} {
	    if { $i == 0 } {
		set index i
	    } elseif { $i == 1 } {
		set index j
	    } elseif { $i == 2 } {
		set index k
	    }

	    global $this-dim-$index
	    global $this-index-$index
	    global $this-index2-$index

	    frame $w.main.$index

	    radiobutton $w.main.$index.l -text "$index axis" -width 6 \
		-anchor w -just left -variable $this-axis -value $i

	    pack $w.main.$index.l -side left

	    scaleEntry2 $w.main.$index.index \
		0 [expr [set $this-dim-$index] - 1] 200 \
		$this-index-$index $this-index2-$index

	    pack $w.main.$index.l $w.main.$index.index -side left
	}

	if { [set $this-dims] == 3 } {
	    pack $w.main.l $w.main.i $w.main.j $w.main.k \
		-side top -padx 10 -pady 5

	} elseif { [set $this-dims] == 2 } {
	    pack $w.main.l $w.main.i $w.main.j -side top -padx 10 -pady 5
	} elseif { [set $this-dims] == 1 } {
	    pack $w.main.l $w.main.i -side top -padx 10 -pady 5
	}

	#  Options

	iwidgets::labeledframe $w.opt -labelpos nw -labeltext "Options"
	set opt [$w.opt childsite]
	
	iwidgets::optionmenu $opt.update -labeltext "Update:" \
		-labelpos w -command "$this set_update_type $opt.update"
	$opt.update insert end Manual "On Release" Auto
	$opt.update select [set $this-update_type]

	global $this-update
	set $this-update $opt.update

	pack $opt.update -side top -anchor w

	pack $w.main $w.opt -side top -fill x -expand 1

	makeSciButtonPanel $w $w $this
	moveToCursor $w
    }


    method scaleEntry2 { win start stop length var1 var2 } {
	frame $win 
	pack $win -side top -padx 5

	scale $win.s -from $start -to $stop -length $length \
	    -variable $var1 -orient horizontal -showvalue false \
	    -command "$this updateSliderEntry $var1 $var2"

	entry $win.e -width 4 -text $var2

	bind $win.s <ButtonRelease> "$this sliderRelease"

	bind $win.e <Return> "$this manualSliderEntryReturn \
             $start $stop $var1 $var2"
	bind $win.e <KeyRelease> "$this manualSliderEntry \
             $start $stop $var1 $var2"

	pack $win.s -side left
	pack $win.e -side bottom -padx 5
    }


    method sliderRelease {} {
	global $this-update_type

	if { [set $this-update_type] == "On Release" } {
	    eval "$this-c needexecute"
	}
    }

    method updateSliderEntry {var_slider var_typed someUknownVar} {
	global $this-continuous
	global $this-update_type

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


    method update_index { } {
	global $this-i-index
	global $this-j-index
	global $this-k-index

	global $this-i-index2
	global $this-j-index2
	global $this-k-index2

	set $this-i-index2 [set $this-i-index]
	set $this-j-index2 [set $this-j-index]
	set $this-k-index2 [set $this-k-index]
    }


    method update_set_size_callback { name1 name2 op } {
	set_size
    }


    method set_size { } {
	global $this-dims
	global $this-i-dim
	global $this-j-dim
	global $this-k-dim
	global $this-axis

	if { [set $this-axis] >= [set $this-dims] } {
	    set $this-axis [expr [set $this-dims]-1]
	}

	set w .ui[modname]

	if {[winfo exists $w]} {

	    pack forget $w.main.i
	    pack forget $w.main.k
	    pack forget $w.main.j

	    if { [set $this-dims] == 3 } {
		pack $w.main.l $w.main.i $w.main.j $w.main.k \
		    -side top -padx 10 -pady 5
	    } elseif { [set $this-dims] == 2 } {
		pack $w.main.l $w.main.i $w.main.j \
		    -side top -padx 10 -pady 5	    
	    } elseif { [set $this-dims] == 1 } {
		pack $w.main.l $w.main.i -side top -padx 10 -pady 5	    
	    }
	}

	for {set i 0} {$i < 3} {incr i 1} {
	    if { $i == 0 } {
		set index i
	    } elseif { $i == 1 } {
		set index j
	    } elseif { $i == 2 } {
		set index k
	    }

	    global $this-index-$index
	    global $this-index2-$index

	    set stop_val [expr [set $this-dim-$index]-1]

	    if [ expr [winfo exists $w] ] {

		# Update the sliders to the new bounds.
		$w.main.$index.index.s configure -from 0 -to $stop_val

		bind $w.main.$index.index.e \
		    <KeyRelease> "$this manualSliderEntry 0 $stop_val $this-index-$index $this-index2-$index"
	    }

	    # Reset all of the slider values to the index values.
	    if { [set $this-index-$index] > $stop_val } {
		set $this-index-$index $stop_val
	    }

	    # Update the text values.
	    set $this-index2-$index [set $this-index-$index]
	}
    }


    method update_type_callback { name1 name2 op } {
	set window .ui[modname]
	if {[winfo exists $window]} {
	    [set $this-update] select [set $this-update_type]
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
}

