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

itcl_class SCIRun_Fields_FieldSubSample {
    inherit Module
    constructor {config} {
        set name FieldSubSample
        set_defaults
    }

    method set_defaults {} {

	global $this-dims

	global $this-idim
	global $this-jdim
	global $this-kdim

	global $this-istart
	global $this-jstart
	global $this-kstart

	global $this-istart2
	global $this-jstart2
	global $this-kstart2

	global $this-idelta
	global $this-jdelta
	global $this-kdelta

	global $this-idelta2
	global $this-jdelta2
	global $this-kdelta2

	global $this-iskip
	global $this-jskip
	global $this-kskip

	global $this-iskip2
	global $this-jskip2
	global $this-kskip2

	global $this-iwrap
	global $this-jwrap
	global $this-kwrap

	set $this-dims 3

	set $this-idim 10
	set $this-jdim 10
	set $this-kdim 10

	set $this-istart 0
	set $this-jstart 0
	set $this-kstart 0

	set $this-istart2 "0"
	set $this-jstart2 "0"
	set $this-kstart2 "0"

	set $this-idelta 1
	set $this-jdelta 1
	set $this-kdelta 1

	set $this-idelta2 "1"
	set $this-jdelta2 "1"
	set $this-kdelta2 "1"

	set $this-iskip 10
	set $this-jskip 5
	set $this-kskip 1

	set $this-iskip2 "10"
	set $this-jskip2 "5"
	set $this-kskip2 "1"

	set $this-iwrap 0
	set $this-jwrap 0
	set $this-kwrap 0
    }

    method ui {} {

	global $this-dims

	global $this-idim
	global $this-jdim
	global $this-kdim

	global $this-istart
	global $this-kstart
	global $this-kstart

	global $this-istart2
	global $this-kstart2
	global $this-kstart2

	global $this-idelta
	global $this-jdelta
	global $this-kdelta

	global $this-idelta2
	global $this-jdelta2
	global $this-kdelta2

	global $this-iskip
	global $this-jskip
	global $this-kskip

	global $this-iskip2
	global $this-jskip2
	global $this-kskip2

	global $this-iwrap
	global $this-jwrap
	global $this-kwrap

	set tmp 0.0

        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }

        toplevel $w

	frame $w.l
	label $w.l.direction -text "Direction"       -width  9 -anchor w -just left
	label $w.l.start     -text "Start Node"      -width 11 -anchor w -just left
	label $w.l.delta     -text "Number of Nodes" -width 16 -anchor w -just left
	label $w.l.skip      -text "Skip Nodes"      -width 16 -anchor w -just left
	label $w.l.wrap      -text "Wrap"            -width  6 -anchor w -just left

	pack $w.l.direction -side left
	pack $w.l.start     -side left -padx 60
	pack $w.l.delta     -side left -padx 90
	pack $w.l.skip      -side left -padx 15
	pack $w.l.wrap      -side left

	#	grid $w.l.direction $w.l.start $w.l.delta $w.l.skip $w.l.wrap

	# Update the sliders to have the new end values.
	if [ set $this-iwrap ] {
	    set cc 1
	} else {
	    set cc 2
	}

	frame $w.i

	label $w.i.l -text "i :" -width 3 -anchor w -just left

	pack $w.i.l -side left

	scaleEntry4 $w.i.start \
	    0 [expr [set $this-idim] - $cc] 200 \
	    $this-istart $this-istart2 $this-idelta $this-idelta2 $this-iwrap

	scaleEntry2 $w.i.delta \
	    1 [expr [set $this-idim] - [expr $cc - 1]] 200 $this-idelta $this-idelta2

	scaleEntry2 $w.i.skip \
	    1 50 100 $this-iskip $this-iskip2

	checkbutton $w.i.wrap -variable $this-iwrap \
	    -command "$this wrap $w.i.start $w.i.delta $this-iwrap $this-idim \
			  $this-istart $this-istart2 $this-idelta $this-idelta2"

	pack $w.i.l $w.i.start $w.i.delta $w.i.skip $w.i.wrap -side left
#	grid $w.i.l $w.i.start $w.i.delta $w.i.skip $w.i.wrap


	if [ set $this-jwrap ] {
	    set cc 1
	} else {
	    set cc 2
	}

	frame $w.j

	label $w.j.l -text "j :" -width 3 -anchor w -just left

	pack $w.j.l -side left

	scaleEntry4 $w.j.start \
	    0 [expr [set $this-jdim] - $cc] 200 \
	    $this-jstart $this-jstart2  $this-jdelta $this-jdelta2 $this-jwrap

	scaleEntry2 $w.j.delta \
	    1 [expr [set $this-jdim] - [expr $cc - 1]] 200 $this-jdelta $this-jdelta2

	scaleEntry2 $w.j.skip \
	    1 50 100 $this-jskip $this-jskip2

	checkbutton $w.j.wrap -variable $this-jwrap \
	    -command "$this wrap $w.j.start $w.j.delta $this-jwrap $this-jdim \
			  $this-jstart $this-jstart2 $this-jdelta $this-jdelta2"

	pack $w.j.start $w.j.delta $w.j.skip $w.j.wrap -side left
#	grid $w.j.start $w.j.delta $w.j.skip $w.j.wrap


	if [ set $this-kwrap ] {
	    set cc 1
	} else {
	    set cc 2
	}

	frame $w.k

	label $w.k.l -text "k :" -width 3 -anchor w -just left

	pack $w.k.l -side left

	scaleEntry4 $w.k.start \
	    0 [expr [set $this-kdim] - $cc] 200 \
	    $this-kstart $this-kstart2 $this-kdelta $this-kdelta2 $this-kwrap

	scaleEntry2 $w.k.delta \
	    1 [expr [set $this-kdim] - [expr $cc - 1]] 200 $this-kdelta $this-kdelta2

	scaleEntry2 $w.k.skip \
	    1  5 100 $this-kskip $this-kskip2

	checkbutton $w.k.wrap -variable $this-kwrap \
	    -command "$this wrap $w.k.start $w.k.delta $this-kwrap $this-kdim \
			  $this-kstart $this-kstart2 $this-kdelta $this-kdelta2"

	pack $w.k.start $w.k.delta $w.k.skip $w.k.wrap -side left
#	grid $w.k.start $w.k.delta $w.k.skip $w.k.wrap


	frame $w.misc

	button $w.misc.b -text "Execute" -command "$this-c needexecute"

	pack $w.misc.b  -side left -padx 25

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
	    set $var2 $start
	}
	
	if { [set $var2] > $stop } {
	    set $var2 $stop }
	
	set $var1 [set $var2]
    }


    method wrap { win1 win2 wrap stop var1 var2 var3 var4 } {

	# Update the sliders to have the new end values.
	if [ set $wrap ] {
	    set stop_node   [expr [set $stop] - 1]
	    set stop_number [expr [set $stop] - 0]
	} else {
	    set stop_node   [expr [set $stop] - 2]
	    set stop_number [expr [set $stop] - 1]
	}

 	if [ expr [winfo exists $win1] ] {

	    $win1.s configure -from 0 -to $stop_node

	    bind $win1.e \
		<Return> "$this manualSliderEntry 0 $stop_node $var1 $var2"
	}

 	if [ expr [winfo exists $win1] ] {

	    $win2.s configure -from 1 -to $stop_number

	    bind $win2.e \
		<Return> "$this manualSliderEntry 1 $stop_number $var3 $var4"
	}
    }

    method scaleEntry4 { win start stop length var1 var2 var3 var4 wrap } {
	frame $win 
	pack $win -side top -padx 5

	scale $win.s -from $start -to $stop -length $length \
	    -variable $var1 -orient horizontal -showvalue false \
	    -command "$this updateSliderEntry4 $stop $var1 $var2 $var3 $var4 $wrap"

	entry $win.e -width 4 -text $var2

	bind $win.e <Return> "$this manualSliderEntry $start $stop $var1 $var2"

	pack $win.s -side left
	pack $win.e -side bottom -padx 5
    }


    method updateSliderEntry4 { dim var1 var2 var3 var4 wrap someUknownVar } {

	if { [set $wrap] == 0 } {
	    if { [expr [set $var1] + [set $var3]] >
		 [expr $dim - 1] } {
		set $var3 [expr [expr $dim - 1] - [set $var1] ]
	    }
	}
	
	set $var2 [set $var1]
	set $var4 [set $var3]
    }

    method set_size {dims idim jdim kdim} {
	set w .ui[modname]

	global $this-dims

	global $this-idim
	global $this-jdim
	global $this-kdim

	global $this-istart
	global $this-jstart
	global $this-kstart

	global $this-idelta
	global $this-jdelta
	global $this-kdelta

	global $this-iskip
	global $this-jskip
	global $this-kskip

	set $this-dims $dims
	set $this-idim $idim
	set $this-jdim $jdim
	set $this-kdim $kdim

	# Reset all of the slider values to the start values.
	set $this-istart 0
	set $this-kstart 0
	set $this-jstart 0

	set $this-idelta 1
	set $this-jdelta 1
	set $this-kdelta 1

	set $this-iskip 10
	set $this-jskip 5
	set $this-kskip 1

	if [ expr [winfo exists $w] ] {

	    # Update the sliders to have the new end values.
	    if [ set $this-iwrap ] {
		$w.i.start.s configure -from 0 -to [expr $idim - 1]
		$w.i.delta.s configure -from 0 -to [expr $idim - 0]

		bind $w.i.start.e <Return> "$this manualSliderEntry 0 [expr $idim - 1] $this-istart $this-istart2"
		bind $w.i.delta.e <Return> "$this manualSliderEntry 1 [expr $idim - 0] $this-idelta $this-idelta2"

	    } else {
		$w.i.start.s configure -from 0 -to [expr $idim - 2]
		$w.i.delta.s configure -from 0 -to [expr $idim - 1]

		bind $w.i.start.e <Return> "$this manualSliderEntry 0 [expr $idim - 2] $this-istart $this-istart2"
		bind $w.i.delta.e <Return> "$this manualSliderEntry 1 [expr $idim - 1] $this-idelta $this-idelta2"
	    }

	    if [ set $this-iwrap ] {
		$w.j.start.s configure -from 0 -to [expr $jdim - 1]
		$w.j.delta.s configure -from 0 -to [expr $jdim - 0]

		bind $w.j.start.e <Return> "$this manualSliderEntry 0 [expr $jdim - 1] $this-jstart $this-jstart2"
		bind $w.j.delta.e <Return> "$this manualSliderEntry 1 [expr $jdim - 0] $this-jdelta $this-jdelta2"
	    } else {
		$w.j.start.s configure -from 0 -to [expr $jdim - 2]
		$w.j.delta.s configure -from 0 -to [expr $jdim - 1]

		bind $w.j.start.e <Return> "$this manualSliderEntry 0 [expr $jdim - 2] $this-jstart $this-jstart2"
		bind $w.j.delta.e <Return> "$this manualSliderEntry 1 [expr $jdim - 1] $this-jdelta $this-jdelta2"
	    }

	    if [ set $this-iwrap ] {
		$w.k.start.s configure -from 0 -to [expr $kdim - 1]
		$w.k.delta.s configure -from 0 -to [expr $kdim - 0]
		bind $w.k.start.e <Return> "$this manualSliderEntry 0 [expr $kdim - 1] $this-kstart $this-kstart2"
		bind $w.k.delta.e <Return> "$this manualSliderEntry 1 [expr $kdim - 0] $this-kdelta $this-kdelta2"
	    } else {
		$w.k.start.s configure -from 0 -to [expr $kdim - 2]
		$w.k.delta.s configure -from 0 -to [expr $kdim - 1]

		bind $w.k.start.e <Return> "$this manualSliderEntry 0 [expr $kdim - 2] $this-kstart $this-kstart2"
		bind $w.k.delta.e <Return> "$this manualSliderEntry 1 [expr $kdim - 1] $this-kdelta $this-kdelta2"
	    }
	}

	# Update the delta values to be at the initials values.
	if [ set $this-iwrap ] {
	    set $this-idelta $idim
	} else {
	    set $this-idelta [expr $idim - 1]
	}

	if [ set $this-jwrap ] {
	    set $this-jdelta $jdim
	} else {
	    set $this-jdelta [expr $jdim - 1]
	}

	if [ set $this-kwrap ] {
	    set $this-kdelta $kdim
	} else {
	    set $this-kdelta [expr $kdim - 1]
	}

	# Update the text values.
	set $this-istart2 [set $this-istart]
	set $this-jstart2 [set $this-jstart]
	set $this-kstart2 [set $this-kstart]

	set $this-idelta2 [set $this-idelta]
	set $this-jdelta2 [set $this-jdelta]
	set $this-kdelta2 [set $this-kdelta]

	set $this-iskip2 [set $this-iskip]
	set $this-jskip2 [set $this-jskip]
	set $this-kskip2 [set $this-kskip]

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
    }
}
