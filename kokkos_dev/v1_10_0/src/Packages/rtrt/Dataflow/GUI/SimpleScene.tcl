#
#  SimpleScene.tcl
#
#  Written by:
#   David Weinstein
#   Department of Computer Science
#   University of Utah
#   January 2003
#
#  Copyright (C) 2003 SCI Group
#

itcl_class rtrt_Scenes_SimpleScene {
    inherit Module

    constructor {config} {
	set name SimpleScene
	set_defaults
    }
    
    method set_defaults {} {
	global $this-color-r
	set $this-color-r 0.1
	global $this-color-g
	set $this-color-g 0.3
	global $this-color-b
	set $this-color-b 0.1
	global $this-color-a
	set $this-color-a 0.2
	global $this-reflectance
	set $this-reflectance 0.5
	global $this-shininess
	set $this-shininess 30.0
    }

    method raiseColor {swatch color msg} {
	 global $color
	 set window .ui[modname]
	 if {[winfo exists $window.color]} {
	     raise $window.color
	     return;
	 } else {
	     toplevel $window.color
	     makeColorPicker $window.color $color \
		     "$this setColor $swatch $color $msg" \
		     "destroy $window.color"
	 }
    }

    method setColor {swatch color msg} {
	 global $color
	 global $color-r
	 global $color-g
	 global $color-b
	 set ir [expr int([set $color-r] * 65535)]
	 set ig [expr int([set $color-g] * 65535)]
	 set ib [expr int([set $color-b] * 65535)]

	 set window .ui[modname]
	 $swatch config -background [format #%04x%04x%04x $ir $ig $ib]
	 $this-c $msg
    }

    method addColorSelection {w text color msg} {
	 #add node color picking 
	 global $color
	 global $color-r
	 global $color-g
	 global $color-b
	 set ir [expr int([set $color-r] * 65535)]
	 set ig [expr int([set $color-g] * 65535)]
	 set ib [expr int([set $color-b] * 65535)]
	 
	 frame $w.frame
	 frame $w.frame.swatch -relief ridge -borderwidth \
		 4 -height 0.8c -width 1.0c \
		 -background [format #%04x%04x%04x $ir $ig $ib]
	 
	 set cmd "$this raiseColor $w.frame.swatch $color $msg"
	 button $w.frame.set_color -text $text -command $cmd
	 
	 #pack the node color frame
	 pack $w.frame.set_color $w.frame.swatch -side left
	 pack $w.frame -side left

    }

    # this is the main function which creates the initial window
    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    wm deiconify $w
	    raise $w
	    return;
	}
	toplevel $w
	wm minsize $w 100 50
	set n "$this-c needexecute"
	
	frame $w.options
	pack $w.options -side top -fill x -padx 2 -pady 2
	scale $w.options.reflectance -variable $this-reflectance \
		-label "Reflectance" -showvalue true -orient horizontal \
		-relief groove -length 200 -from 0.00 -to 1.00 \
		-tickinterval 0.20 -resolution 0.01
	scale $w.options.shininess -variable $this-shininess \
		-label "Shininess" -showvalue true -orient horizontal \
		-relief groove -length 200 -from 0 -to 100 \
		-tickinterval 20 -resolution 1
	pack $w.options.reflectance $w.options.shininess -side top -expand 1 \
		-fill x

	frame $w.buttons
	button $w.buttons.execute -text "Execute" -command $n
	button $w.buttons.update -text "Update" \
		-command "$this-c update_sphere_material"
#	button $w.buttons.update -text "Update" -command $n
	button $w.buttons.dismiss -text "Dismiss" -command "wm withdraw $w"
	pack $w.buttons.execute $w.buttons.update $w.buttons.dismiss \
		-side left -padx 10
#	pack $w.buttons.update $w.buttons.dismiss \
#		-side left -padx 15

	frame $w.c
	addColorSelection $w.c "Sphere Color" $this-color \
	    "update_sphere_material"

	pack $w.options -side top -expand yes -fill x -pady 5
	pack $w.buttons $w.c -side top -pady 5
    }
}
