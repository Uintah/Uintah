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

itcl::class ItPDSimPartGui {

    variable w
    variable n
    variable len

    constructor {} {
	set n 0
	set len 0
    }

    method ui { window } {
	global $this-df
	global $this-mean

	set $this-df 3
	set $this-mean ""

	set w $window

        iwidgets::entryfield $w.df -labeltext "df:" \
	    -textvariable $this-df  \
	    -width 4 \
	    -command "$this-c df \[$w.df get\]" 

	frame $w.mean
	label $w.mean.label -text "Mean: "
	pack $w.mean.label -side left

	frame $w.children

	pack $w.df $w.mean -anchor w
	pack $w.children

    }

    method set-df { v } {
	global $this-df
	if { [set $this-df] != $v } {
	    set $this-df $v
	}
    }

    method set-mean { args } {
	set l [llength $args]
	if { $l != $len } {
	    for {set i $len} {$i < $l} {incr i} {
		entry $w.mean.$i -textvariable $this-mean$i -width 6 
		bind $w.mean.$i <Return>  "$this-c mean $i \[$w.mean.$i get \]"
		pack $w.mean.$i -side left
	    }
	}
	for {set i 0} {$i < $l} {incr i} {
	    set $this-mean$i [lindex $args $i]
	}
    }

    method new-child-window { name } {
	set child [iwidgets::Labeledframe $w.children.$n -labeltext $name]
	pack $child -side top -anchor w
	incr n
	return [$child childsite]
    }
}


