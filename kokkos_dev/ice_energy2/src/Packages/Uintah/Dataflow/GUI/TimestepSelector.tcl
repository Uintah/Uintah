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
        # Clock variables
        global $this-short_hand_res
        global $this-long_hand_res
        global $this-short_hand_ticks
        global $this-long_hand_ticks
        global $this-clock_position_x
        global $this-clock_position_y
        global $this-clock_radius
        set $this-time 0
        set $this-max_time 100
        set $this-timeval 0
        set $this-animate 0
        set $this-tinc 1
        set $this-timeposition_x 0.25
        set $this-timeposition_y 0.85
        set $this-def-color-r 1.0
        set $this-def-color-g 1.0
        set $this-def-color-b 1.0
        set $this-font_size "*"
        set exposed 0
        # Clock variables
        set $this-short_hand_res    0.001
        set $this-long_hand_res     0.0001
        set $this-short_hand_ticks  5
        set $this-long_hand_ticks  10
        set $this-clock_position_x  0.80
        set $this-clock_position_y -0.3
        set $this-clock_radius      0.25
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

    method make_entry {w text v c} {
        frame $w
        label $w.l -text "$text"
        pack $w.l -side left
        entry $w.e -textvariable $v
        bind $w.e <Return> $c
        pack $w.e -side right
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
        
        iwidgets::labeledframe $w.timeVis.position \
            -labeltext "Time Step Position"
        set location [$w.timeVis.position childsite]

        set can [makeStickyLocator $location.sl \
                     $this-timeposition_x $this-timeposition_y 100 100]
        $can bind movable <ButtonRelease> "$this-c update_timeposition"

        pack $location.sl -fill x -expand yes -side left

        iwidgets::labeledframe $w.timeVis.clock_pos \
            -labeltext "Clock Position"
        set location [$w.timeVis.clock_pos childsite]

        set can [makeStickyLocator $location.sl \
                     $this-clock_position_x $this-clock_position_y 100 100]
        $can bind movable <ButtonRelease> "$this-c update_clock"

        pack $location.sl -fill x -expand yes -side left

        frame $w.timeVis.clock_vals
        make_entry $w.timeVis.clock_vals.shr "Short Hand Res" \
            $this-short_hand_res "$this-c update_clock"
        make_entry $w.timeVis.clock_vals.lhr "Long Hand Res" \
            $this-long_hand_res "$this-c update_clock"
        make_entry $w.timeVis.clock_vals.sht "Short Hand Ticks" \
            $this-short_hand_ticks "$this-c update_clock"
        make_entry $w.timeVis.clock_vals.lht "Long Hand Ticks" \
            $this-long_hand_ticks "$this-c update_clock"
        make_entry $w.timeVis.clock_vals.cr "Clock Radius" \
            $this-clock_radius "$this-c update_clock"
        pack $w.timeVis.clock_vals.shr $w.timeVis.clock_vals.lhr \
             $w.timeVis.clock_vals.sht $w.timeVis.clock_vals.lht \
             $w.timeVis.clock_vals.cr \
            -fill x -expand yes -side top

        pack $w.timeVis.position $w.timeVis.clock_pos $w.timeVis.clock_vals \
            -fill x -expand yes -side left
        
        # This is the clock
#        frame $w.timeVis.clock -relief  -borderwidth 2
#        pack $w.timeVis.clock -side left -fill x -expand yes

        pack $w.timeVis -side top -fill x
        ######    end timeVis section

        makeSciButtonPanel $w $w $this
 
        bind $f.tframe.time <ButtonRelease> $n

        # Bind the left and right arrow keys
        bind $w <KeyPress-Right> "$this incTime"
        bind $w <KeyPress-Left> "$this decTime"
    }
    
    method incTime {} {
        if { [set $this-time] == [set $this-max_time] } {
            set $this-time 0
        } else { 
            incr $this-time 1
        }
        $this-c needexecute
    }
    method decTime {} {
        if { [set $this-time] == 0 } {
            set $this-time [set $this-max_time]
        } else { 
            incr $this-time -1
        }
        $this-c needexecute
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
        $w.timeVis.cs.colorFrame.col config -background [format #%04x%04x%04x $ir $ig $ib]
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
