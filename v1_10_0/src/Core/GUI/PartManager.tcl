
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

package require Iwidgets 3.0

itcl::class PartManager {

    variable w
    variable n

    constructor {} {
	set n 0
    }

    method ui { window } {
	set w $window

	iwidgets::tabnotebook  $w.tabs -raiseselect true -auto true -width 400
	$w.tabs configure -tabpos "n" 
	pack $w.tabs -side top -anchor w -expand true -fill both
    }

    method lock {} {
	set current [$w.tabs view]
	puts "lock $current"
	for {set i 0} {$i < $n} {incr i} {
	    if { $i != $current } {
		$w.tabs pageconfigure $i -state disable
	    }
	}
    }

    method unlock {} {
	for {set i 0} {$i < $n} {incr i} {
	    $w.tabs pageconfigure $i -state normal
	}
    }

    method new-child-window { name } {
	set child [$w.tabs add -label "$name" -command "$this-c select $n"]
	incr n
	return $child
    }
}
	
