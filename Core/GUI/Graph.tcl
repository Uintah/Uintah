
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
    variable status

    constructor {} {
	global $this-gl-window 
    }

    method ui { w } {
	global $this-gl-window

	set parent-window $w

	# menubar
	iwidgets::menubar $w.menu 
	$w.menu add menubutton .opt -text "Opt"


	# UIs
	frame $w.ui

	# toolbar
	label $w.status -textvariable status 

	frame $w.f
	iwidgets::toolbar $w.f.tb -helpvariable status -orient vertical
	
	$w.f.tb add button select \
	    -helpstr "Select" -command {puts "select"}

	$w.f.tb add button zoom \
	    -helpstr "Zoom" \
	    -image [image create photo -file "/local/home/yarden/mag_glass_3d.ppm"] \
	    -command {puts "zoom"}

	$w.f.tb add button sub \
	    -helpstr "SubWindow" -command {puts "sub window"}
	#
	# Graph
	#
	iwidgets::scrolledframe $w.f.graph -width 450 -height 350 \
	    -hscrollmode dynamic -vscrollmode dynamic 

	pack $w.f.tb  -fill y -side left
	pack $w.f.graph -expand yes  -fill both

	#
	#
	# OpenGL window of Graph
	#  
	set $this-gl-window [$w.f.graph childsite].gl

	# Info winfow
	#iwidgets::Labeledframe $w.info -labelpos nw -labeltext "Info"
	
	# pack $w.info 
	pack $w.menu -fill x 
	pack $w.ui -anchor w 
	pack $w.f -expand yes -fill both
	pack $w.status  -fill x
	
    }

}
	
	