
#
#  PartManager.tcl
#
#  Written by:
#   Yarden Livant
#   Department of Computer Science
#   University of Utah
#   Oct 2001
#
#  Copyright (C) 2001 SCI Group
#

package require Iwidgets 3.1

class PartManager {

    variable w
    variable n

    constructor {} {
	set n 0
    }

    method ui { window } {
	set w $window

	iwidgets::tabnotebook  $w.tabs -raiseselect true
	$w.tabs configure -tabpos "n" 
	pack $w.tabs -side top -anchor w
    }

    method add-part { name } {
	puts "PartManager tcl: add-part  $name"
    }

    method new-child-window { name } {
	set child [$w.tabs add -label "$name" -command "$this-c select $n"]
	incr n
	return $child
    }
}
	
