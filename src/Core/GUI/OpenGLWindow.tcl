
#
#  OpenGLWindow
#
#  Written by:
#   Yarden Livant
#   Department of Computer Science
#   University of Utah
#   July 2001
#
#  Copyright (C) 2001 SCI Group
#

class OpenGLWindow {

    variable w

    constructor {} {
    }

    method ui { window parent } {
	set w $window

	opengl $w -geometry 500x300 -doublebuffer true \
	    -direct true -rgba true 
	bind $w <Map> "$this-c map $w"
	bind $w <Expose> "$parent-c redraw $w"
	pack $w
    }

    method change_params { n } {
    }
	
    method setobj { obj } {
	bindtags $w [list [$obj getbinds] $w all]
    }

    method resize { width height } {
	$w configure -geometry ${width}x$height
    }

    method set_cursor { c } {
	$w configure -cursor $c
    }
}
	
	