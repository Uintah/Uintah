
#
#  Graph.tcl
#
#  Written by:
#   Yarden Livant
#   Department of Computer Science
#   University of Utah
#   July 2001
#
#  Copyright (C) 2001 SCI Group
#

package require Iwidgets 3.1

class Graph {
    inherit OpenGLWindow

    constructor {args} {
	set name Graph
	set_defaults
    }

    method set_defaults {} {
	set $this-graph 0
	set $this-parent-window 0
    }

    method ui { w } {
	global $this-ogl
	
	set $this-parent-window $w

	# Info winfow
	# iwidgets::Labeledframe $w.info -labelpos nw -labeltext "Info"

	# Control
	frame $w.ctrl
#
#       Graph
#
	iwidgets::scrolledframe $w.graph -width 450 -height 350 \
	    -hscrollmode dynamic -vscrollmode dynamic 

#
#
#          OpenGL window of Graph
#  
	set graph [$w.graph childsite]
	OpenGLWindow $this-ogl
	$this-ogl ui $this $graph

	
	# pack $w.info 
	pack $w.ctrl -anchor w 
	pack $w.graph -expand yes  -fill y
    }

}
	
	