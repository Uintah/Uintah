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

#  Hairline.tcl
#  Written by:
#   Yarden Livnat
#   Department of Computer Science
#   University of Utah
#   July 2001
#  Copyright (C) 2001 SCI Group

package require Iwidgets 3.0

itcl::class Hairline {

    variable opt
    variable menu
    variable gl
    variable parent
    variable n
    variable initialized
    variable current
    variable allocated

    constructor { args } {
	set initialized 0
	set val(0) 1
	set allocated 0
	set current 0
    }

    destructor {
	delete object  $w.hairline
    }

    method ui { window name } {
	set opt $window.$name
	set n $name

    }

    method init {} {
	iwidgets::labeledframe $opt -labeltext $n -labelpos nw 
	pack $opt
	set initialized 1
    }
	
    method add { n color value} {

	if { $initialized == 0 } {
	    $this init
	}

	set w [$opt childsite].$n
	    
 	frame $w-line -background $color -width 10 -height 2
 	label $w-val -text $value
#	grid $w-line $w-val -padx 3 -sticky w
    }


    method widget { name } {
	$this-c widget $name
    }

    method values { args } {
	if { $initialized == 0 } {
	    $this init
	}
  	set w [$opt childsite]

	set l [llength $args]
	if { $allocated < $l } {
	    for {set i $allocated} {$i < $l} {incr i} {
		frame $w.$i-line -width 10 -height 2
		label $w.$i-val 
	    }
	    set allocated $l
	}
	
	if { $l < $current } {
	    for {set i $l} {$i < $current} {incr i} {
		grid forget $w.$i-line $w.$i-val
	    }
	} 

	if { $current < $l } {
	    for {set i $current} {$i < $l} {incr i} {
		grid $w.$i-line $w.$i-val -padx 3 -sticky w 
	    }
	} 
	    

  	for {set i 0} {$i < $l} {incr i} {
	    set item [lindex $args $i]
	    set c [lindex $item 0]
	    set v [lindex $item 1]
  	    $w.$i-line config -bg $c
  	    $w.$i-val config -text $v
  	}

	set current $l
    }
}
