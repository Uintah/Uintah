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
    protected exposed

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
        global $this-timeposition_x
        global $this-timeposition_y
	set $this-time 0
	set $this-max_time 100
	set $this-timeval 0
	set $this-animate 0
	set $this-tinc 1
        set $this-timeposition_x 0.5
        set $this-timeposition_y 0.5
	set $this-def-color-r 1.0
	set $this-def-color-g 1.0
	set $this-def-color-b 1.0
	set $this-font_size "*"
	set exposed 0
    } 
    
    method close {} {
	set w .ui[modname]
	set exposed 0
	destroy $w
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

        #######  start timeVis section
        frame $w.timeVis -relief groove -borderwidth 2

	frame $w.timeVis.cs -relief flat -borderwidth 0
	addColorSelection $w.timeVis.cs
	frame $w.timeVis.cs.fs -relief flat
	pack $w.timeVis.cs.fs -side top -fill x -expand yes
	
	set bf $w.timeVis.cs.fs
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
	pack $w.timeVis.cs -side top -anchor w -fill x

        # This is the text position
        frame $w.timeVis.position -relief flat -borderwidth 2
        pack $w.timeVis.position -side top -fill x -expand yes

        canvas $w.timeVis.position.canvas -bg "#ffffff" -height 70 -width 70
	pack $w.timeVis.position.canvas -side top -anchor w -expand yes \
            -fill x -fill y

	bind $w.timeVis.position.canvas <Expose> "$this canvasExpose"
	bind $w.timeVis.position.canvas <Button-1> "$this moveNode %x %y"
	bind $w.timeVis.position.canvas <B1-Motion> "$this moveNode %x %y"
#	bind $w.f.f1.canvas <ButtonRelease> "$this update; $this-c needexecute"

        pack $w.timeVis -side top -fill x
        ######    end timeVis section

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

    method moveNode { x y } {
	global $this-timeposition_x
	global $this-timeposition_y
	set w .ui[modname]
	set c $w.timeVis.position.canvas
	set cw [winfo width $c]
	set ch [winfo height $c]

        # Normalize the pixel locations to [-1, 1]
        set $this-timeposition_x [expr $x/double($cw)*2 - 1]
        set $this-timeposition_y [expr -$y/double($ch)*2 + 1]

        drawTimePosition
    }

    method drawTimePosition { } {
	global $this-timeposition_x
	global $this-timeposition_y
	set w .ui[modname]
	set c $w.timeVis.position.canvas
	set cw [winfo width $c]
	set ch [winfo height $c]

        # Clear the canvas
        $c create rectangle 0 0 $cw $ch -fill white

        # Denormalize the positions to screen locations
        set x [expr $cw * (([set $this-timeposition_x] + 1) / 2)]
        set y [expr $ch * ((-[set $this-timeposition_y] + 1) / 2)]
#        puts "Time Position = ($x, $y)"
        if { $x < 0 } {
            set x 0
        } elseif { $x > $cw } {
            set x $cw 
        }

        if { $y < 0 } {
            set y 0
        } elseif { $y > $ch } {
            set y $ch 
        }
	
        $c create oval [expr $x - 5] [expr $y - 5] \
            [expr $x+5] [expr $y+5] -outline black \
            -fill red -tags node
    }

    method canvasExpose {} {
	set w .ui[modname]
	
	if { [winfo viewable $w.timeVis.position.canvas] } { 
	    if { $exposed } {
		return
	    } else {
		set exposed 1
		$this drawTimePosition
	    } 
	} else {
	    return
	}
    }
    
        
}
