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

# GUI for FieldSlicer module
# by Michael Callahan &&
#    Allen Sanderson
# December 2002

# This GUI interface is for selecting an axis and index for sub sampling a
# topologically structured field

itcl_class SCIRun_FieldsCreate_FieldSlicer {
    inherit Module
    constructor {config} {
        set name FieldSlicer
        set_defaults
    }

    method set_defaults {} {

	global $this-dims
	global $this-axis

	set $this-axis 2
	set $this-dims 3

	for {set i 0} {$i < 3} {incr i 1} {
	    if { $i == 0 } {
		set index i
	    } elseif { $i == 1 } {
		set index j
	    } elseif { $i == 2 } {
		set index k
	    }

	    global $this-$index-dim
	    global $this-$index-index
	    global $this-$index-index2

	    set $this-$index-dim 1
	    set $this-$index-index 1
	    set $this-$index-index2 "0"
	}

	trace variable $this-dims w "$this set_size"
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

	frame $w.l
	label $w.l.direction -text "Direction"       -width 9 -anchor w -just left
	label $w.l.index     -text "Slice Node"      -width 11 -anchor w -just left

	pack $w.l.direction -side left
	pack $w.l.index     -side left -padx 75

	for {set i 0} {$i < 3} {incr i 1} {
	    if { $i == 0 } {
		set index i
	    } elseif { $i == 1 } {
		set index j
	    } elseif { $i == 2 } {
		set index k
	    }

	    global $this-$index-dim
	    global $this-$index-index
	    global $this-$index-index2

	    frame $w.$index

	    radiobutton $w.$index.l -text "$index axis" -width 6 \
		-anchor w -just left -variable $this-axis -value $i

	    pack $w.$index.l -side left

	    scaleEntry2 $w.$index.index \
		0 [expr [set $this-$index-dim] - 1] 200 \
		$this-$index-index $this-$index-index2

	    pack $w.$index.l $w.$index.index -side left
	}

	frame $w.misc
	button $w.misc.execute -text "Execute" -command "$this-c needexecute"
	button $w.misc.close -text Close -command "destroy $w"
	pack $w.misc.execute $w.misc.close -side left -padx 25

	if { [set $this-dims] == 3 } {
	    pack $w.l $w.i $w.j $w.k $w.misc -side top -padx 10 -pady 5
	} elseif { [set $this-dims] == 2 } {
	    pack $w.l $w.i $w.j $w.misc -side top -padx 10 -pady 5	    
	} elseif { [set $this-dims] == 1 } {
	    pack $w.l $w.i $w.misc -side top -padx 10 -pady 5	    
	}
    }


    method scaleEntry2 { win start stop length var1 var2 } {
	frame $win 
	pack $win -side top -padx 5

	scale $win.s -from $start -to $stop -length $length \
	    -variable $var1 -orient horizontal -showvalue false \
	    -command "$this updateSliderEntry $var1 $var2"

	entry $win.e -width 4 -text $var2

	bind $win.e <Return> "$this manualSliderEntry $start $stop $var1 $var2"

	pack $win.s -side left
	pack $win.e -side bottom -padx 5
    }

    method updateSliderEntry {var1 var2 someUknownVar} {
	set $var2 [set $var1]
    }

    method manualSliderEntry { start stop var1 var2 } {

	if { [set $var2] < $start } {
	    set $var2 $start }
	
	if { [set $var2] > $stop } {
	    set $var2 $stop }
	
	set $var1 [set $var2]
    }

    method set_size {name element op} {
	set w .ui[modname]

	global $this-axis
	global $this-dims

	global $this-i-dim
	global $this-j-dim
	global $this-k-dim

	if { [set $this-axis] >= [set $this-dims] } {
	    set $this-axis [expr [set $this-dims]-1]
	}

	if {[winfo exists $w]} {
	    pack forget $w.i
	    pack forget $w.k
	    pack forget $w.j
	    pack forget $w.misc

	    if { [set $this-dims] == 3 } {
		pack $w.l $w.i $w.j $w.k $w.misc -side top -padx 10 -pady 5
	    } elseif { [set $this-dims] == 2 } {
		pack $w.l $w.i $w.j $w.misc -side top -padx 10 -pady 5	    
	    } elseif { [set $this-dims] == 1 } {
		pack $w.l $w.i $w.misc -side top -padx 10 -pady 5	    
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

	    global $this-$index-index
	    global $this-$index-index2

	    set stop_val [expr [set $this-$index-dim]-1]

	    if [ expr [winfo exists $w] ] {

		# Update the sliders to the new bounds.
		$w.$index.index.s configure -from 0 -to $stop_val

		bind $w.$index.index.e \
		    <Return> "$this manualSliderEntry 0 $stop_val $this-$index-index $this-$index-index2"
	    }

	    # Reset all of the slider values to the index values.
	    if { [set $this-$index-index] > $stop_val } {
		set $this-$index-index $stop_val
	    }

	    # Update the text values.
	    set $this-$index-index2 [set $this-$index-index]
	}
    }
}

