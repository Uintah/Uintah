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

# GUI for FieldSubSample module
# by Allen R. Sanderson
# March 2002

# This GUI interface is for sub sampling a topologically structured field.

itcl_class SCIRun_FieldsCreate_FieldSubSample {
    inherit Module
    constructor {config} {
        set name FieldSubSample
        set_defaults
    }

    method set_defaults {} {

	global $this-wrap
	global $this-dims

	set $this-wrap 0
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
	    global $this-$index-start
	    global $this-$index-start2
	    global $this-$index-stop
	    global $this-$index-stop2
	    global $this-$index-stride
	    global $this-$index-stride2
	    global $this-$index-wrap

	    set $this-$index-dim     2
	    set $this-$index-start   0
	    set $this-$index-start2 "0"
	    set $this-$index-stop    1
	    set $this-$index-stop2  "1"
	    set $this-$index-stride    1
	    set $this-$index-stride2  "1"
	    set $this-$index-wrap    0
	}

	trace variable $this-dims w "$this set_size"
    }

    method ui {} {

	global $this-wrap
	global $this-dims

	set tmp 0.0

        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }

	if { [set $this-wrap] } {
	    set wrap normal
	} else {
	    set wrap disable
	}

        toplevel $w

	frame $w.l
	label $w.l.direction -text "Index"  -width 5 -anchor w -just left
	label $w.l.start     -text "Start"  -width 5 -anchor w -just left
	label $w.l.stop      -text "Stop"   -width 5 -anchor w -just left
	label $w.l.stride    -text "Stride" -width 6 -anchor w -just left
	label $w.l.wrap      -text "Wrap"   -width 4 -anchor w -just left

	pack $w.l.direction -side left
	pack $w.l.start     -side left -padx  70
	pack $w.l.stop      -side left -padx 110
	pack $w.l.stride    -side left -padx  40
	pack $w.l.wrap      -side left

#	grid $w.l.direction $w.l.start $w.l.stop $w.l.stride $w.l.wrap

	for {set i 0} {$i < 3} {incr i 1} {
	    if { $i == 0 } {
		set index i
	    } elseif { $i == 1 } {
		set index j
	    } elseif { $i == 2 } {
		set index k
	    }

	    global $this-$index-dim
	    global $this-$index-start
	    global $this-$index-start2
	    global $this-$index-stop
	    global $this-$index-stop2
	    global $this-$index-stride
	    global $this-$index-stride2
	    global $this-$index-wrap

	    # Update the sliders to have the new end values.
	    if { [set $this-wrap] == 0 } {    
		set $this-$index-wrap 0
	    }

	    if [set $this-$index-wrap] {
		set start_val 0
		set stop_val [expr [set $this-$index-dim] - 1]
	    } else {
		set start_val 1
		set stop_val [expr [set $this-$index-dim] - 2]
	    }

	    frame $w.$index

	    label $w.$index.l -text " $index :" -width 3 -anchor w -just left

	    pack $w.$index.l -side left

	    scaleEntry4 $w.$index.start \
		0 $stop_val 200 \
		$this-$index-start $this-$index-start2 $index

	    scaleEntry2 $w.$index.stop \
		$start_val [expr [set $this-$index-dim] - 1] 200 \
		$this-$index-stop $this-$index-stop2

	    scaleEntry2 $w.$index.stride \
		1 [expr [set $this-$index-dim] - 1] 100 $this-$index-stride $this-$index-stride2

	    checkbutton $w.$index.wrap -variable $this-$index-wrap \
		    -state $wrap -disabledforeground "" \
		    -command "$this wrap $index"

	    pack $w.$index.l $w.$index.start $w.$index.stop \
		    $w.$index.stride $w.$index.wrap -side left
#	    grid $w.$index.l $w.$index.start $w.$index.stop 
#		    $w.$index.stride $w.$index.wrap
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


    method wrap { index } {

	global $this-$index-start
	global $this-$index-start2
	global $this-$index-stop
	global $this-$index-stop2
	global $this-$index-dim
	global $this-$index-wrap

	set w .ui[modname]

	if [ expr [winfo exists $w] ] {

	    # Update the sliders to have the new end values.

	    if [ set $this-$index-wrap ] {
		set start_val 0
		set stop_val  [expr [set $this-$index-dim] - 1]
	    } else {
		set start_val [expr [set $this-$index-start] + 1]
		set stop_val  [expr [set $this-$index-dim] - 2]
	    }

	    $w.$index.start.s configure -from 0 -to $stop_val
	    $w.$index.stop.s configure -from $start_val -to [expr [set $this-$index-dim] - 1]

	    bind $w.$index.start.e <Return> \
		"$this manualSliderEntry4 0 $stop_val $this-$index-start $this-$index-start2 $index"
	    bind $w.$index.stop.e  <Return> \
		"$this manualSliderEntry $start_val  [expr [set $this-$index-dim] - 1] $this-$index-stop $this-$index-stop2"
	}
    }

    method scaleEntry4 { win start stop length var1 var2 index } {
	frame $win 
	pack $win -side top -padx 5

	scale $win.s -from $start -to $stop -length $length \
	    -variable $var1 -orient horizontal -showvalue false \
	    -command "$this updateSliderEntry4 $index"

	entry $win.e -width 4 -text $var2

	bind $win.e <Return> \
	    "$this manualSliderEntry4 $start $stop $var1 $var2 $index"

	pack $win.s -side left
	pack $win.e -side bottom -padx 5
    }

    method updateSliderEntry4 { index someUknownVar } {

	global $this-$index-start
	global $this-$index-start2
	global $this-$index-stop
	global $this-$index-stop2

	wrap $index

	set $this-$index-start2 [set $this-$index-start]
	set $this-$index-stop2  [set $this-$index-stop]
    }

    method manualSliderEntry4 { start stop var1 var2 index } {

	if { [set $var2] < $start } {
	    set $var2 $start }
	
	if { [set $var2] > $stop } {
	    set $var2 $stop }
	
	set $var1 [set $var2]

	updateSliderEntry4 $index 0
    }

    method set_size {name element op} {
	global $this-wrap
	global $this-dims
	global $this-i-dim
	global $this-j-dim
	global $this-k-dim

	set w .ui[modname]

	if [ expr [winfo exists $w] ] {
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

	    global $this-$index-start
	    global $this-$index-start2
	    global $this-$index-stop
	    global $this-$index-stop2
	    global $this-$index-stride
	    global $this-$index-stride2

	    set $this-$index-wrap 0

	    set stop_val1 [expr [set $this-$index-dim] - 1]
	    set stop_val2 [expr [set $this-$index-dim] - 2]

	    if [ expr [winfo exists $w] ] {

		if { [set $this-wrap ] } {
		    $w.$index.wrap configure -state normal
		} else {
		    $w.$index.wrap configure -state disabled 
		}

		# Update the sliders to the new bounds.
		$w.$index.start.s configure -from 0 -to $stop_val2
		$w.$index.stop.s  configure -from 0 -to $stop_val1
		$w.$index.stride.s  configure -from 1 -to $stop_val1

		bind $w.$index.start.e <Return> \
		    "$this manualSliderEntry4 0 $stop_val2 $this-$index-start $this-$index-start2 $index"
		bind $w.$index.stop.e  <Return> \
		    "$this manualSliderEntry  1 $stop_val1 $this-$index-stop $this-$index-stop2"
		bind $w.$index.stride.e  <Return> \
		    "$this manualSliderEntry  1 $stop_val1 $this-$index-stride $this-$index-stride2"
	    }

	    # Update the stop values to be at the initials values.
	    set $this-$index-start 0	    
	    set $this-$index-stop  $stop_val1
	    set $this-$index-stride  1

	    # Update the text values.
	    set $this-$index-start2 [set $this-$index-start]
	    set $this-$index-stop2  [set $this-$index-stop]
	    set $this-$index-stride2  [set $this-$index-stride]
	}
    }
}
