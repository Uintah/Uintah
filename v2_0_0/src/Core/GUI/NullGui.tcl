
#
#  NullGui.tcl
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

itcl::class NullGui {

    variable w

    constructor {} {
    }

    method ui { window } {
	set w $window
    }

    method new-child-window { name } {
	frame $w.f
	return $w.f
    }
}
	
