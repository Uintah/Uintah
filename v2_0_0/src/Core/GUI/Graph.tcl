
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

package require Iwidgets 3.0

itcl::class Graph {

    variable graph
    variable parent-window
    variable status

    constructor {} {
    }

    method ui { w } {
	set parent-window $w

	# menubar
	iwidgets::menubar $w.menu 
	$w.menu add menubutton .opt -text "Opt"


	# UIs
	frame $w.ui

	# status
	label $w.status -textvariable status 

	frame $w.f
	iwidgets::toolbar $w.f.tb -helpvariable status -orient vertical

	# graph area
	
	frame $w.f.graph

	pack $w.f.tb  -fill y -side left
	pack $w.f.graph -expand yes  -fill both

	# Info winfow
	#iwidgets::Labeledframe $w.info -labelpos nw -labeltext "Info"
	
	# pack $w.info 

	pack $w.menu -fill x 
	pack $w.ui -anchor w 
	pack $w.f -expand yes -fill both
	pack $w.status  -fill x
	
    }

}
	

	#
	# Graph
	#
#	iwidgets::scrolledframe $w.f.graph -width 450 -height 350 \
\#	    -hscrollmode dynamic -vscrollmode dynamic 

	#
	#
	# OpenGL window of Graph
	#  
#	set $this-gl-window [$w.f.graph childsite].gl

	
