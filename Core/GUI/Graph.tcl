
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

    variable graph
    variable parent-window

    constructor {} {
	global $this-gl-window 
    }

    method ui { w } {
	global $this-gl-window

	puts "Graph ui $w"
	set parent-window $w

	# menubar
	iwidgets::menubar $w.menu 
	puts "tcl graph menu at $w.menu"
	
	# Options
	frame $w.opt

	#
	# Graph
	#
	iwidgets::scrolledframe $w.graph -width 450 -height 350 \
	    -hscrollmode dynamic -vscrollmode dynamic 

	#
	#
	# OpenGL window of Graph
	#  
	set $this-gl-window [$w.graph childsite].gl
	#OpenGLWindow ogl
	#ogl ui $graph.gl $this

	
	# Info winfow
	#iwidgets::Labeledframe $w.info -labelpos nw -labeltext "Info"
	
	# pack $w.info 
	pack $w.menu -fill x -expand yes
	pack $w.opt -anchor w 
	pack $w.graph -expand yes  -fill y
    }

}
	
	