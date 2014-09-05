########################################
#CLASS
#    VizControl
#    Visualization control for simulation data that contains
#    information on both a regular grid in particle sets.
#OVERVIEW TEXT
#    This module receives a ParticleGridReader object.  The user
#    interface is dynamically created based information provided by the
#    ParticleGridReader.  The user can then select which variables he/she
#    wishes to view in a visualization.
#KEYWORDS
#    ParticleGridReader, Material/Particle Method
#AUTHOR
#    Kurt Zimmerman
#    Department of Computer Science
#    University of Utah
#    January 1999
#    Copyright (C) 1999 SCI Group
#LOG
#    Created January 5, 1999
########################################

package require Iwidgets 3.0 

catch {rename TimestepSelector ""}

itcl_class Uintah_Selectors_TimestepSelector { 
    inherit Module 

    constructor {config} { 
        set name TimestepSelector 
        set_defaults

    } 

    method set_defaults {} { 
        global $this-tcl_status 
	global $this-animate
	global $this-time;
	global $this-timeval 
	global $this-max_time
	global $this-def-color-r
	global $this-def-color-g
	global $this-def-color-b
	global $this-font_size
	global $this-tinc
	set $this-time 0
	set $this-max_time 100
	set $this-timeval 0
	set $this-animate 0
	set $this-tinc 1
	set $this-def-color-r 1.0
	set $this-def-color-g 1.0
	set $this-def-color-b 1.0
	set $this-font_size "*"
    } 
    
    method ui {} { 
        set w .ui[modname] 

        if {[winfo exists $w]} {
	    return
	}
	$this buildTopLevel
	moveToCursor $w
    }

    method buildTopLevel {} {
        set w .ui[modname]

        toplevel $w 
	
	set n "$this-c needexecute"
	frame $w.f -relief flat
 	pack $w.f -side top -expand yes -fill both

	frame $w.tframe
	set f $w.tframe
	frame $f.tframe
	frame $f.lframe
	label $f.lframe.step -text "Time Step"
	label $f.lframe.val -text "    Time      "
	global $this-max_time

	set tickinterval [expr ([set $this-max_time] -1)/2.0]
	scale $f.tframe.time -orient horizontal -from 0 \
	    -to [set $this-max_time]  \
	    -resolution 1 -bd 2 -variable $this-time \
	    -tickinterval $tickinterval
	entry $f.tframe.tval -state disabled \
	     -textvariable $this-timeval -justify left
	pack $f -side top -fill x -padx 2 -pady 2
	pack $f.lframe -side top -expand yes -fill x
	pack $f.tframe -side top -expand yes -fill x
	pack $f.lframe.step -side left -anchor w
	pack $f.lframe.val -side right -anchor e
	pack $f.tframe.time -side left -expand yes \
	    -fill x -padx 2 -pady 2 -anchor w
	pack $f.tframe.tval -side right  -padx 2 -pady 2 -anchor e \
	    -expand yes -fill x

	frame $w.aframe -relief groove -borderwidth 2
	pack $w.aframe -side top -fill x -padx 2 -pady 2
	checkbutton $w.aframe.abutton -text Animate \
	    -variable $this-animate
	entry $w.aframe.status  -width 15  -relief sunken -bd 2 \
	    -textvariable $this-tcl_status 
	pack $w.aframe.abutton -side left
	pack $w.aframe.status -side right  -padx 2 -pady 2

	frame $w.tincf 
	pack $w.tincf -side top -fill x -padx 2 -pady 2
	label $w.tincf.l -text "Time step increment"
	pack $w.tincf.l -side left
	entry $w.tincf.e -textvariable $this-tinc
	pack $w.tincf.e -side right

	frame $w.cs -relief groove -borderwidth 2
	addColorSelection $w.cs
	frame $w.cs.fs -relief flat
	pack $w.cs.fs -side top -fill x -expand yes
	
	set bf $w.cs.fs
	label $bf.fs -text "Font Size:"
	radiobutton $bf.b1 -text "*" -variable $this-font_size -value "*" \
	    -command $n
	radiobutton $bf.b2 -text "10" -variable $this-font_size -value "10" \
	    -command $n
	radiobutton $bf.b3 -text "11" -variable $this-font_size -value "11" \
	    -command $n
	radiobutton $bf.b4 -text "13" -variable $this-font_size -value "13" \
	    -command $n
	radiobutton $bf.b5 -text "15" -variable $this-font_size -value "15" \
	    -command $n

	pack $bf.fs $bf.b1 $bf.b2 $bf.b3 $bf.b4 $bf.b5 -side left
	pack $w.cs -side top -anchor w -fill x

	makeSciButtonPanel $w $w $this
 
	bind $f.tframe.time <ButtonRelease> $n
    }


    method SetTimeRange { timesteps } {
	set w .ui[modname].tframe.tframe
	set $this-max_time [expr $timesteps -1 ]
	if { [winfo exists $w.time] } {
	    set interval [expr ([set $this-max_time] -1)/2.0]
	    $w.time configure -from 0
	    $w.time configure -to [set $this-max_time]
	    $w.time configure -tickinterval $interval
	}
    }

    method isOn { bval } {
	return  [set $this-$bval]
    }
    method raiseColor {} {
	 set window .ui[modname]
	 if {[winfo exists $window.color]} {
	     SciRaise $window.color
	     return
	 } else {
	     makeColorPicker $window.color $this-def-color \
		     "$this setColor" \
		     "destroy $window.color"
	 }
    }
    method setColor {} {
	global $this-def-color-r
	global $this-def-color-g
	global $this-def-color-b
	set ir [expr int([set $this-def-color-r] * 65535)]
	set ig [expr int([set $this-def-color-g] * 65535)]
	set ib [expr int([set $this-def-color-b] * 65535)]

	set w .ui[modname]
	$w.cs.colorFrame.col config -background [format #%04x%04x%04x $ir $ig $ib]
        $this-c needexecute
    }
    method addColorSelection {frame} {
	#add node color picking 
	global $this-def-color-r
	global $this-def-color-g
	global $this-def-color-b
	set ir [expr int([set $this-def-color-r] * 65535)]
	set ig [expr int([set $this-def-color-g] * 65535)]
	set ib [expr int([set $this-def-color-b] * 65535)]
	
	frame $frame.colorFrame
	frame $frame.colorFrame.col -relief ridge -borderwidth \
	    4 -height 0.8c -width 1.0c \
	    -background [format #%04x%04x%04x $ir $ig $ib]
	set cmmd "$this raiseColor"
        button $frame.colorFrame.set_color \
            -text "Text Color (In Viewer)" -command $cmmd
        #pack the node color frame
	pack $frame.colorFrame.set_color $frame.colorFrame.col -side left -padx 5 -pady 3
	pack $frame.colorFrame -side left
    }
	    	    
}
