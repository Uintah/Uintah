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
    variable w

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
	#
	# widgets menu
	#
	$menu add menubutton .widgets -text "Widgets"

	$menu add command .widgets.hairline -label Hairline -underline 0 \
	    -command "$this widget hairline"
	$menu add command .widgets.zoom -label Zoom -underline 0 \
	    -command "$this widget zoom"

	#
	# option area
	#
	iwidgets::labeledframe $opt.d -labeltext $n -labelpos nw

	set w [$opt.d childsite]

	# list of polys
	frame $w.poly
	pack $w.poly -side left -anchor n

	   # two different options
	   frame $w.poly.all
	   frame $w.poly.one

	   # select the 'all' option
	   pack $w.poly.all 
	

	# select polys
	frame $w.s

  	  label $w.s.select -text "Select:"
	  set $this-select 2

	  frame $w.s.b1
	  radiobutton $w.s.b1.one -text "One" \
	      -variable $this-select -value 1 \
	      -command "$this select one"
	  radiobutton $w.s.b1.many -text "Many" \
	      -variable $this-select -value 2 \
	      -command "$this select many"

	  pack $w.s.b1.one $w.s.b1.many -side left

	  label $w.s.scale -text "Scale:"
	  set $this-scale 1

	  frame $w.s.b2
	  radiobutton $w.s.b2.all -text "All" \
	      -variable $this-scale -value 1 \
	      -command "$this-c redraw" -anchor w
	  radiobutton $w.s.b2.each -text "Each" \
	      -variable $this-scale -value 2 \
	      -command "$this-c redraw" -anchor w
	  pack $w.s.b2.all $w.s.b2.each -side left
	
	pack $w.s.select -side top -anchor w
	pack $w.s.b1 -side top -anchor e
	pack $w.s.scale -side top -anchor w
	pack $w.s.b2 -side top -anchor e 

	pack $w.s -side left -ipadx 5 -anchor n

	pack $opt.d

	bind DiagramTags <ButtonPress> "$this-c ButtonPress %x %y %b "
	bind DiagramTags <B1-Motion> "$this-c Motion %x %y %b "
	bind DiagramTags <ButtonRelease> "$this-c ButtonRelease %x %y %b "
	
	set initialized 1
    }
	
    method add { n name color} {
	if { $initialized == 0 } {
	    $this init
	}
	    
	checkbutton $w.poly.all.$name -text $name -variable val($name) \
	    -fg $color \
	    -command "$this-c select $n \$val($name)"
	$w.poly.all.$name select
	pack $w.poly.all.$name -side top

	radiobutton $w.poly.one.$name -text $name \
	    -variable select-one -value $n \
	    -bg $color \
	    -command "$this-c select-one $n"
	pack $w.poly.one.$name -side top -ipady 2

    }

    method select {which} {
	if { $which == "one" } {
	    pack forget $w.poly.all
	    pack $w.poly.one
	} else {
	    pack forget $w.poly.one 
	    pack $w.poly.all
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
	set win $w.$last
	frame $win
	pack $win -side left -anchor n
	
	set last [expr $last + 1]
	return $win
    }
}
