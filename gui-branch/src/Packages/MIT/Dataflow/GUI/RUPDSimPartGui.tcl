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


package require Iwidgets 3.0

itcl::class RUPDSimPartGui {

    variable w
    variable n
    variable len

    constructor {} {
	set n 0
	set len 0
    }

    method ui { window } {
	global $this-par

	set $this-par ""

	set w $window

	frame $w.par
	label $w.par.label -text "Par: "
	pack $w.par.label -side left

	frame $w.children

	pack $w.par -anchor w
	pack $w.children

    }

    method set-par { args } {
	set l [llength $args]
	if { $l != $len } {
	    for {set i $len} {$i < $l} {incr i} {
		entry $w.par.$i -textvariable $this-par$i -width 6 
		bind $w.par.$i <Return>  "$this-c par $i \[$w.par.$i get \]"
		pack $w.par.$i -side left
	    }
	}
	for {set i 0} {$i < $l} {incr i} {
	    set $this-par$i [lindex $args $i]
	}
    }

    method new-child-window { name } {
	set child [iwidgets::Labeledframe $w.children.$n -labeltext $name]
	pack $child -side top -anchor w
	incr n
	return [$child childsite]
    }
}


