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

# GUI for FusionSlicer module
# by Allen R. Sanderson
# March 2002

itcl_class Fusion_Fields_FusionSlicer {
    inherit Module
    constructor {config} {
        set name FusionSlicer
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
    }

    method ui {} {

	global $this-idim
	global $this-jdim
	global $this-kdim

	global $this-iindex
	global $this-kindex
	global $this-kindex

	global $this-iindex2
	global $this-kindex2
	global $this-kindex2

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

	radiobutton $w.i.l -text "Radial :" -width 9 -anchor w -just left -variable $this-axis -value 0

	pack $w.i.l -side left

	scaleEntry2 $w.i.index \
		0 [expr [set $this-idim] - 1] 200 $this-iindex $this-iindex2

	pack $w.i.l $w.i.index -side left


	frame $w.j

	radiobutton $w.j.l -text "Theta :" -width 9 -anchor w -just left -variable $this-axis -value 1

	pack $w.j.l -side left

	scaleEntry2 $w.j.index \
		0 [expr [set $this-jdim] - 1] 200 $this-jindex $this-jindex2

	pack $w.j.index -side left


	frame $w.k

	radiobutton $w.k.l -text "Phi :" -width 9 -anchor w -just left -variable $this-axis -value 2

	pack $w.k.l -side left

	scaleEntry2 $w.k.index \
		0 [expr [set $this-kdim] - 1] 200 $this-kindex $this-kindex2

	pack $w.k.index -side left


	frame $w.misc

	button $w.misc.b -text "Execute" -command "$this-c needexecute"

	pack $w.misc.b  -side left -padx 25

        pack $w.l $w.i $w.j $w.k $w.misc -side top -padx 10 -pady 5
    }


    method scaleEntry2 { win index stop length var1 var2 } {
	frame $win 
	pack $win -side top -padx 5

	scale $win.s -from $index -to $stop -length $length \
		-variable $var1 -orient horizontal -showvalue false -command "$this checkScale"

	entry $win.e -width 4 -text $var2

	bind $win.e <Return> "$this updateScale"

	pack $win.s -side left
	pack $win.e -side bottom -padx 5
    }

    method checkScale { iTemp } {

	global $this-idim

	global $this-iindex
	global $this-jindex
	global $this-kindex

	global $this-iindex2
	global $this-jindex2
	global $this-kindex2

	set $this-iindex2 [set $this-iindex]
	set $this-jindex2 [set $this-jindex]
	set $this-kindex2 [set $this-kindex]
    }

    method updateScale { } {

	global $this-iindex
	global $this-jindex
	global $this-kindex

	global $this-iindex2
	global $this-kindex2
	global $this-kindex2

	if { [set $this-iindex2] < 0 } { set $this-iindex2 0 }
	if { [set $this-jindex2] < 0 } { set $this-jindex2 0 }
	if { [set $this-kindex2] < 0 } { set $this-kindex2 0 }
	
	if { [set $this-iindex2] > [expr [set $this-idim] - 1] } {
	    set $this-iindex2 [expr [set $this-idim] - 1] }
	if { [set $this-jindex2] > [expr [set $this-jdim] - 1] } {
	    set $this-jindex2 [expr [set $this-jdim] - 1] }
	if { [set $this-kindex2] > [expr [set $this-kdim] - 1] } {
	    set $this-kindex2 [expr [set $this-kdim] - 1] }
	
	set $this-iindex [set $this-iindex2]
	set $this-jindex [set $this-jindex2]
	set $this-kindex [set $this-kindex2]
    }

    method set_size {idim jdim kdim} {
	set w .ui[modname]

	global $this-idim
	global $this-jdim
	global $this-kdim

	global $this-iindex
	global $this-jindex
	global $this-kindex

	set $this-idim $idim
	set $this-jdim $jdim
	set $this-kdim $kdim

	# Reset all of the slider values to the index values.
	set $this-iindex 0
	set $this-kindex 0
	set $this-jindex 0

	if [ expr [winfo exists $w] ] {

	    # Update the sliders to have the new end values.
	    $w.i.index.s configure -from 0 -to [expr $idim - 1]
	    $w.j.index.s configure -from 0 -to [expr $jdim - 1]
	    $w.k.index.s configure -from 0 -to [expr $kdim - 1]
	}

	# Update the text values.
	set $this-iindex2 [set $this-iindex]
	set $this-jindex2 [set $this-jindex]
	set $this-kindex2 [set $this-kindex]
   }
}



