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

# GUI for EditFusionField module
# by Allen R. Sanderson
# March 2002

itcl_class Fusion_Fields_EditFusionField {
    inherit Module
    constructor {config} {
        set name EditFusionField
        set_defaults
    }

    method set_defaults {} {

	global $this-isoVal

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

	set $this-idim 1
	set $this-jdim 1
	set $this-kdim 1

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
    }

    method ui {} {

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

	set tmp 0.0

        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }
        toplevel $w

	frame $w.l
	label $w.l.direction -text "Direction"       -width 9 -anchor w -just left
	label $w.l.start     -text "Start Node"      -width 11 -anchor w -just left
	label $w.l.delta     -text "Number of Nodes" -width 16 -anchor w -just left
	label $w.l.skip      -text "Skip Nodes"      -width 16 -anchor w -just left

	pack $w.l.direction -side left
	pack $w.l.start     -side left -padx 75
	pack $w.l.delta     -side left -padx 80
	pack $w.l.skip      -side left -padx 20

#	grid $w.l.direction $w.l.start $w.l.delta $w.l.skip 

	frame $w.i

	label $w.i.l -text "Radial :" -width 9 -anchor w -just left

	pack $w.i.l -side left

	scaleEntry2 $w.i.start \
		0 [expr [set $this-idim] - 2] 200 $this-istart $this-istart2

	scaleEntry2 $w.i.delta \
		1 [set $this-idim] 200 $this-idelta $this-idelta2

	scaleEntry2 $w.i.skip \
		1 50 100 $this-iskip $this-iskip2

	pack $w.i.l $w.i.start $w.i.delta $w.i.skip -side left
#	grid $w.i.l $w.i.start $w.i.delta $w.i.skip


	frame $w.j

	label $w.j.l -text "Theta :" -width 9 -anchor w -just left

	pack $w.j.l -side left

	scaleEntry2 $w.j.start \
		0 [expr [set $this-jdim] - 1] 200 $this-jstart $this-jstart2

	scaleEntry2 $w.j.delta \
		1 [set $this-jdim] 200 $this-jdelta $this-jdelta2

	scaleEntry2 $w.j.skip \
		1 50 100 $this-jskip $this-jskip2

	pack $w.j.start $w.j.delta $w.j.skip -side left
#	grid $w.j.start $w.j.delta $w.j.skip


	frame $w.k

	label $w.k.l -text "Phi :" -width 9 -anchor w -just left

	pack $w.k.l -side left

	scaleEntry2 $w.k.start \
		0 [expr [set $this-kdim] - 1] 200 $this-kstart $this-kstart2

	scaleEntry2 $w.k.delta \
		1 [set $this-kdim] 200 $this-kdelta $this-kdelta2

	scaleEntry2 $w.k.skip \
		1  5 100 $this-kskip $this-kskip2

	pack $w.k.start $w.k.delta $w.k.skip -side left
#	grid $w.k.start $w.k.delta $w.k.skip


	frame $w.misc

	button $w.misc.b -text "Execute" -command "$this-c needexecute"

	pack $w.misc.b  -side left -padx 25

        pack $w.l $w.i $w.j $w.k $w.misc -side top -padx 10 -pady 5
    }


    method scaleEntry2 { win start stop length var1 var2 } {
	frame $win 
	pack $win -side top -padx 5

	scale $win.s -from $start -to $stop -length $length \
		-variable $var1 -orient horizontal -showvalue false -command "$this checkScale"

	entry $win.e -width 4 -text $var2

	bind $win.e <Return> "$this updateScale"

	pack $win.s -side left
	pack $win.e -side bottom -padx 5
    }

    method checkScale { iTemp } {

	global $this-idim

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

	if { [expr [set $this-istart] + [set $this-idelta]] >
	     [expr [set $this-idim]  - 1] } {
	    set $this-idelta [expr [expr [set $this-idim]  - 1] - [set $this-istart] ]
	}

	set $this-istart2 [set $this-istart]
	set $this-jstart2 [set $this-jstart]
	set $this-kstart2 [set $this-kstart]

	set $this-idelta2 [set $this-idelta]
	set $this-jdelta2 [set $this-jdelta]
	set $this-kdelta2 [set $this-kdelta]

	set $this-iskip2 [set $this-iskip]
	set $this-jskip2 [set $this-jskip]
	set $this-kskip2 [set $this-kskip]
    }

    method updateScale { } {

	global $this-istart
	global $this-jstart
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


	if { [set $this-istart2] < 0 } { set $this-istart2 0 }
	if { [set $this-jstart2] < 0 } { set $this-jstart2 0 }
	if { [set $this-kstart2] < 0 } { set $this-kstart2 0 }
	
	if { [set $this-idelta2] < 1 } { set $this-idelta2 1 }
	if { [set $this-jdelta2] < 1 } { set $this-jdelta2 1 }
	if { [set $this-kdelta2] < 1 } { set $this-kdelta2 1 }
	
	if { [set $this-iskip2] < 1 } { set $this-iskip2 1 }
	if { [set $this-jskip2] < 1 } { set $this-jskip2 1 }
	if { [set $this-kskip2] < 1 } { set $this-kskip2 1 }


	if { [set $this-istart2] > [expr [set $this-idim] - 2] } {
	    set $this-istart2 [expr [set $this-idim] - 2] }
	if { [set $this-jstart2] > [expr [set $this-jdim] - 1] } {
	    set $this-jstart2 [expr [set $this-jdim] - 1] }
	if { [set $this-kstart2] > [expr [set $this-kdim] - 1] } {
	    set $this-kstart2 [expr [set $this-kdim] - 1] }
	
	if { [expr [set $this-istart] + [set $this-idelta2]] >
	     [expr [set $this-idim]  - 1] } {
	    set $this-idelta2 [expr [expr [set $this-idim]  - 1] - [set $this-istart] ]
	}

	if { [set $this-jdelta2] > [set $this-jdim] } {
	    set $this-jdelta2 [set $this-jdim] }
	if { [set $this-kdelta2] > [set $this-kdim] } {
	    set $this-kdelta2 [set $this-kdim] }


	if { [set $this-iskip2] > 50 } { set $this-iskip2 50 }
	if { [set $this-jskip2] > 50 } { set $this-jskip2 50 }
	if { [set $this-kskip2] > 10 } { set $this-kskip2 10 }
	

	set $this-istart [set $this-istart2]
	set $this-jstart [set $this-jstart2]
	set $this-kstart [set $this-kstart2]

	set $this-idelta [set $this-idelta2]
	set $this-jdelta [set $this-jdelta2]
	set $this-kdelta [set $this-kdelta2]

	set $this-iskip [set $this-iskip2]
	set $this-jskip [set $this-jskip2]
	set $this-kskip [set $this-kskip2]
    }

    method set_size {idim jdim kdim} {
	set w .ui[modname]

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
	    $w.i.start.s configure -from 0 -to [expr $idim - 2]
	    $w.j.start.s configure -from 0 -to [expr $jdim - 1]
	    $w.k.start.s configure -from 0 -to [expr $kdim - 1]

	    $w.i.delta.s configure -from 1 -to $idim
	    $w.j.delta.s configure -from 1 -to $jdim
	    $w.k.delta.s configure -from 1 -to $kdim
	}

	# Update the delta values to be at the initials values.
	set $this-idelta [expr $idim - 1]
	set $this-jdelta $jdim
	set $this-kdelta $kdim

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
   }
}



