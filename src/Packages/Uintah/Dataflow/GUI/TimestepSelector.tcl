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
	set $this-time 0
	set $this-timeval 0
	set $this-animate 0
	set $this-anisleep 0
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
	scale $f.tframe.time -orient horizontal -from 0 -to 100 \
	    -resolution 1 -bd 2 -variable $this-time \
	    -tickinterval 50
	entry $f.tframe.tval -state disabled \
	    -width 10 -textvariable $this-timeval
	pack $f -side top -fill x -padx 2 -pady 2
	pack $f.lframe -side top -expand yes -fill x
	pack $f.tframe -side top -expand yes -fill x
	pack $f.lframe.step -side left -anchor w
	pack $f.lframe.val -side right -anchor e
	pack $f.tframe.time -side left -expand yes \
	    -fill x -padx 2 -pady 2 -anchor w
	pack $f.tframe.tval -side right  -padx 2 -pady 2 -anchor e

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
	    $w.time configure -from 0
	    $w.time configure -to [expr $timesteps -1 ]
	    $w.time configure -tickinterval $interval
	}
    }

    method isOn { bval } {
	return  [set $this-$bval]
    }

	    	    
}
