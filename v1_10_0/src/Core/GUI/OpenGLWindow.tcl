
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

itcl::class OpenGLWindow {

    variable w
    variable prev

    constructor {} {
    }

    method ui { window parent } {
	set w $window

	opengl $w -geometry 500x300 -doublebuffer true \
	    -direct true -rgba true 
	bind $w <Map> "$this-c map $w"
	bind $w <Expose> "$parent-c redraw $w"
	bind $w <Enter> { set prev [focus]; focus %W }
	bind $w <Leave> { focus $prev }
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

    method set-cursor { c } {
	set current [$w cget -cursor]
	$w configure -cursor $c
	return $current
    }


    method add-bind { b } {
	bindtags $w [concat $b [bindtags $w] ]
    }

    method rem-bind { b } {
	bindtags $w [ldelete $b [bindtags $w] ]
    }

    method ldelete { item list } {
	set i [lsearch -exact $list $item]
	if { $i > -1 } {
	    return [lreplace $list $i $i]
	} else {
	    return $list
	}
    }
}
	
	
