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
	set $this-time 0
	set $this-max_time 100
	set $this-timeval 0
	set $this-animate 0
	set $this-anisleep 0
	set $this-def-color-r 1.0
	set $this-def-color-g 1.0
	set $this-def-color-b 1.0
	set $this-font_size "*"
    } 
    
    method ui {} { 
        set w .ui[modname] 

        if {[winfo exists $w]} { 
	    wm deiconify $w
            raise $w 
        } else { 
	    $this buildTopLevel
	    wm deiconify $w
            raise $w 
	}
    }

    method buildTopLevel {} {
        set w .ui[modname] 

        if {[winfo exists $w]} { 
            return;
        } 
	
        toplevel $w 
	wm withdraw $w
	
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

	scale $f.tframe.time -orient horizontal -from 0 \
	    -to [set $this-max_time]  \
	    -resolution 1 -bd 2 -variable $this-time \
	    -tickinterval 50
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
	    -variable $this-animate -command $n
	entry $w.aframe.status  -width 15  -relief sunken -bd 2 \
	    -textvariable $this-tcl_status 
	pack $w.aframe.abutton -side left
	pack $w.aframe.status -side right  -padx 2 -pady 2
	label $w.l -text "Animation Sleep (seconds)" 
	scale $w.s  -variable $this-anisleep  \
	    -orient horizontal -from 0 -to 600 -resolution 1
	pack $w.l $w.s -side top -fill x

	frame $w.cs -relief groove -borderwidth 2
	addColorSelection $w.cs  $this-def-color \
		"default_color_change"
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

	button $w.b -text "Close" -command "wm withdraw $w"
	pack $w.b -side top -fill x -padx 2 -pady 2

	bind $f.tframe.time <ButtonRelease> $n
    }


    method isVisible {} {
	if {[winfo exists .ui[modname]]} {
	    return 1
	} else {
	    return 0
	}
    }
    

    method SetTimeRange { timesteps } {
	set w .ui[modname].tframe.tframe
	if { [winfo exists $w.time] } {
	    set interval [expr ($timesteps -1)/2.0]
	    set $this-max_time [expr $timesteps -1 ]
	    $w.time configure -from 0
	    $w.time configure -to [expr $timesteps -1 ]
	    $w.time configure -tickinterval $interval
	}
    }

    method isOn { bval } {
	return  [set $this-$bval]
    }
    method raiseColor {col color colMsg} {
	 global $color
	 set window .ui[modname]
	 if {[winfo exists $window.color]} {
	     raise $window.color
	     return;
	 } else {
	     toplevel $window.color
	     makeColorPicker $window.color $color \
		     "$this setColor $col $color $colMsg" \
		     "destroy $window.color"
	 }
    }
    method setColor {col color colMsg} {
	 global $color
	 global $color-r
	 global $color-g
	 global $color-b
	 set ir [expr int([set $color-r] * 65535)]
	 set ig [expr int([set $color-g] * 65535)]
	 set ib [expr int([set $color-b] * 65535)]

	 set window .ui[modname]
	 $col config -background [format #%04x%04x%04x $ir $ig $ib]
	 $this-c $colMsg
    }
    method addColorSelection {frame color colMsg} {
	#add node color picking 
	global $color
	global $color-r
	global $color-g
	global $color-b
	set ir [expr int([set $color-r] * 65535)]
	set ig [expr int([set $color-g] * 65535)]
	set ib [expr int([set $color-b] * 65535)]
	
	frame $frame.colorFrame
	frame $frame.colorFrame.col -relief ridge -borderwidth \
	    4 -height 0.8c -width 1.0c \
	    -background [format #%04x%04x%04x $ir $ig $ib]
	set cmmd "$this raiseColor $frame.colorFrame.col $color $colMsg"
        button $frame.colorFrame.set_color \
            -text "Display Color in Viewer" -command $cmmd
        #pack the node color frame
	pack $frame.colorFrame.set_color $frame.colorFrame.col -side left
	pack $frame.colorFrame -side left
    }
	    	    
}
