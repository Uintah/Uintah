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

    variable opt
    variable menu
    variable gl
    variable parent
    variable n
    variable initialized
    variable last

    constructor { args } {
	set initialized 0
	set val(0) 1
	set last 0
    }

    destructor {
	delete object  $w.diagram
    }

    method ui { window name } {
	set opt $window.opt
	set menu $window.menu
	set n $name

    }

    method init {} {
	# widgets menu
	$menu add menubutton .widgets -text "Widgets"

	$menu add command .widgets.hairline -label Hairline -underline 0 \
	    -command "$this widget hairline"

	# option
	frame $opt.b
	iwidgets::labeledframe $opt.b.all -labeltext $n -labelpos nw
	iwidgets::labeledframe $opt.b.one -labeltext $n -labelpos nw
	pack $opt.b.all 

	frame $opt.select 
	label $opt.select.label -text "Select:"
	set $this-select 2
	radiobutton $opt.select.one -text "One" \
	    -variable $this-select -value 1 \
	    -command "$this select one"
	radiobutton $opt.select.many -text "Many" \
	    -variable $this-select -value 2 \
	    -command "$this select many"
	pack $opt.select.label $opt.select.one $opt.select.many -side left

	frame $opt.scale 
	label $opt.scale.label -text "Scale:"
	set $this-scale 1
	radiobutton $opt.scale.all -text "All" -variable $this-scale \
	    -value 1 \
	    -command "$this-c redraw" -anchor w
	radiobutton $opt.scale.each -text "Each" -variable $this-scale \
	    -value 2 \
	    -command "$this-c redraw" -anchor w
	pack $opt.scale.label $opt.scale.all $opt.scale.each -side left 
	
	pack $opt.b $opt.select $opt.scale -side top 


	bind DiagramTags <ButtonPress> "$this-c ButtonPress %x %y %b "
	bind DiagramTags <B1-Motion> "$this-c Motion %x %y %b "
	bind DiagramTags <ButtonRelease> "$this-c ButtonRelease %x %y %b "
	
	set initialized 1
    }
	
    method add { n name } {
	if { $initialized == 0 } {
	    $this init
	}
	    
	set cs [$opt.b.all childsite]
	checkbutton $cs.$name -text $name -variable val($name) \
	    -command "$this-c select $n \$val($name)"
	$cs.$name select
	pack $cs.$name -side left

	set cs [$opt.b.one childsite]
	radiobutton $cs.$name -text $name -variable select-one -value $n \
	    -command "$this-c select-one $n"
	pack $cs.$name -side left

    }

    method select {which} {
	if { $which == "one" } {
	    pack forget $opt.b.all
	    pack $opt.b.one -side left
	} else {
	    pack forget $opt.b.one 
	    pack $opt.b.all -side left
	}
	$this-c redraw
    }

    method widget { name } {
	$this-c widget $name
    }

    method getbinds {} {
	return DiagramTags
    }

    method new-opt {} {
	set w $opt.$last
	frame $w
	pack $w -side left
	
	set last [expr $last + 1]
	return $w
    }
}
