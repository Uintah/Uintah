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

#  Diagram.tcl
#  Written by:
#   Yarden Livnat
#   Department of Computer Science
#   University of Utah
#   July 2001
#  Copyright (C) 2001 SCI Group

package require Iwidgets 3.0

class Diagram {

    variable w 
    variable n
    variable initialized
    
    constructor { args } {
	set initialized 0
	set val(0) 1
    }

    destructor {
	delete object  $w.diagram
    }

    method ui { window name } {
	set w $window
	set n $name

    }

    method init {} {
	frame $w.b
	iwidgets::labeledframe $w.b.all -labeltext $n -labelpos nw
	iwidgets::labeledframe $w.b.one -labeltext $n -labelpos nw
	pack $w.b.all 

	frame $w.select 
	label $w.select.label -text "Select:"
	set $this-select 2
	radiobutton $w.select.one -text "One" \
	    -variable $this-select -value 1 \
	    -command "$this select one"
	radiobutton $w.select.many -text "Many" \
	    -variable $this-select -value 2 \
	    -command "$this select many"
	pack $w.select.label $w.select.one $w.select.many -side left

	frame $w.scale 
	label $w.scale.label -text "Scale:"
	set $this-scale 1
	radiobutton $w.scale.all -text "All" -variable $this-scale -value 1 \
	    -command "$this-c redraw" -anchor w
	radiobutton $w.scale.each -text "Each" -variable $this-scale -value 2 \
	    -command "$this-c redraw" -anchor w
	pack $w.scale.label $w.scale.all $w.scale.each -side left 
	
	pack $w.b $w.select $w.scale -side top 

	set initialized 1
    }
	
    method add { n name } {
	if { $initialized == 0 } {
	    $this init
	}
	    
	set cs [$w.b.all childsite]
	checkbutton $cs.$name -text $name -variable val($name) \
	    -command "$this-c select $n \$val($name)"
	$cs.$name select
	pack $cs.$name -side left

	set cs [$w.b.one childsite]
	radiobutton $cs.$name -text $name -variable select-one -value $n \
	    -command "$this-c select-one $n"
	pack $cs.$name -side left

    }

    method select {which} {
	if { $which == "one" } {
	    pack forget $w.b.all
	    pack $w.b.one -side left
	} else {
	    pack forget $w.b.one 
	    pack $w.b.all -side left
	}
	$this-c redraw
    }
}
