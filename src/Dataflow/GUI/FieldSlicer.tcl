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

itcl_class SCIRun_Fields_FieldSlicer {
    inherit Module
    constructor {config} {
        set name FieldSlicer
        set_defaults
    }

    method set_defaults {} {

	global $this-axis

	global $this-idim
	global $this-jdim
	global $this-kdim

	global $this-iindex
	global $this-jindex
	global $this-kindex

	global $this-iindex2
	global $this-jindex2
	global $this-kindex2

	global $this-dims

	set $this-axis 2

	set $this-idim 1
	set $this-jdim 1
	set $this-kdim 1

	set $this-iindex 1
	set $this-jindex 0
	set $this-kindex 0

	set $this-iindex2 "0"
	set $this-jindex2 "0"
	set $this-kindex2 "0"

	set $this-dims 3
    }

    method ui {} {

	global $this-idim
	global $this-jdim
	global $this-kdim

	global $this-iindex
	global $this-kindex
	global $this-jindex

	global $this-iindex2
	global $this-kindex2
	global $this-jindex2

	global $this-axis

	global $this-dims

	set $this-axis [expr [set $this-dims]-1]

	set tmp 0.0

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

	frame $w.i

	radiobutton $w.i.l -text "i axis" -width 9 -anchor w -just left -variable $this-axis -value 0

	pack $w.i.l -side left

	scaleEntry2 $w.i.index \
		0 [expr [set $this-idim] - 1] 200 $this-iindex $this-iindex2

	pack $w.i.l $w.i.index -side left


	frame $w.j

	radiobutton $w.j.l -text "j axis" -width 9 -anchor w -just left -variable $this-axis -value 1

	pack $w.j.l -side left

	scaleEntry2 $w.j.index \
		0 [expr [set $this-jdim] - 1] 200 $this-jindex $this-jindex2

	pack $w.j.index -side left


	frame $w.k

	radiobutton $w.k.l -text "k axis" -width 9 -anchor w -just left -variable $this-axis -value 2

	pack $w.k.l -side left

	scaleEntry2 $w.k.index \
		0 [expr [set $this-kdim] - 1] 200 $this-kindex $this-kindex2

	pack $w.k.index -side left


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

    method set_size {dim idim jdim kdim} {
	set w .ui[modname]

	global $this-idim
	global $this-jdim
	global $this-kdim

	global $this-iindex
	global $this-jindex
	global $this-kindex

	global $this-iindex2
	global $this-jindex2
	global $this-kindex2

	global $this-axis

	global $this-dims

	set $this-dims $dim
	set $this-idim $idim
	set $this-jdim $jdim
	set $this-kdim $kdim

	# Reset all of the slider values to the index values.
	if { [set $this-iindex] > [expr [set $this-idim]-1] } {
	    set $this-iindex [expr [set $this-idim]-1]
	}
	if { [set $this-jindex] > [expr [set $this-jdim]-1] } {
	    set $this-jindex [expr [set $this-jdim]-1]
	}
	if { [set $this-kindex] > [expr [set $this-kdim]-1] } {
	    set $this-kindex [expr [set $this-kdim]-1]
	}

	if { [set $this-axis] > [expr [set $this-dims]-1] } {
	    set $this-axis [expr [set $this-dims]-1]
	}

	if [ expr [winfo exists $w] ] {

	    # Update the sliders to have the new end values.
	    $w.i.index.s configure -from 0 -to [expr $idim - 1]
	    $w.j.index.s configure -from 0 -to [expr $jdim - 1]
	    $w.k.index.s configure -from 0 -to [expr $kdim - 1]

	    bind $w.i.index.e \
		<Return> "$this manualSliderEntry 0 [expr $idim - 1] $this-iindex $this-iindex2"
	    bind $w.j.index.e \
		<Return> "$this manualSliderEntry 0 [expr $jdim - 1] $this-jindex $this-jindex2"
	    bind $w.k.index.e \
		<Return> "$this manualSliderEntry 0 [expr $kdim - 1] $this-kindex $this-kindex2"
	}

	# Update the text values.
	set $this-iindex2 [set $this-iindex]
	set $this-jindex2 [set $this-jindex]
	set $this-kindex2 [set $this-kindex]

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



