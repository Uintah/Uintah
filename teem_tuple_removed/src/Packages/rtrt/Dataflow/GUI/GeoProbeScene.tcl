#
#  GeoProbeScene.tcl
#
#  Written by:
#   David Weinstein
#   Department of Computer Science
#   University of Utah
#   January 2003
#
#  Copyright (C) 2003 SCI Group
#

itcl_class rtrt_Scenes_GeoProbeScene {
    inherit Module

    constructor {config} {
	set name GeoProbeScene
	set_defaults
    }
    
    method set_defaults {} {
	global $this-isoval
	set $this-isoval 100
	global $this-xa
	set $this-xa 0.1
	global $this-xb
	set $this-xb 0.9
	global $this-ya
	set $this-ya 0.1
	global $this-yb
	set $this-yb 0.9
	global $this-za
	set $this-za 0.1
	global $this-zb
	set $this-zb 0.9
	global $this-gpfilename
	set $this-gpfilename "/usr/sci/data/Seismic/BP/k12bvox.vol"

	global $this-xa-active
	set $this-xa-active 1
	global $this-xa-usemat
	set $this-xa-usemat 1
	global $this-xb-active
	set $this-xb-active 1
	global $this-xb-usemat
	set $this-xb-usemat 1

	global $this-ya-active
	set $this-ya-active 1
	global $this-ya-usemat
	set $this-ya-usemat 1
	global $this-yb-active
	set $this-yb-active 1
	global $this-yb-usemat
	set $this-yb-usemat 1

	global $this-za-active
	set $this-za-active 1
	global $this-za-usemat
	set $this-za-usemat 1
	global $this-zb-active
	set $this-zb-active 1
	global $this-zb-usemat
	set $this-zb-usemat 1

	global $this-color-r
	global $this-color-g
	global $this-color-b
	set $this-color-r 0.5
	set $this-color-g 0.5
	set $this-color-b 0.5

	global $this-iso_min
	global $this-iso_ma
	global $this-iso_val
	set $this-iso_min 0
	set $this-iso_max 255
	set $this-iso_val 40
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
	 pack $w.frame -side top

    }

    method addCutPlane {w cp label} {
	set var "$this-$cp"
	scale $w.scale -variable $var \
	    -label $label -showvalue true -orient horizontal \
	    -relief groove -length 200 -from 0 -to 1 \
	    -tickinterval 0.25 -resolution 0.01 \
	    -command "$this-c update_cut $cp"
	set cpactive "$var-active"
	checkbutton $w.act -text "Active" \
	    -variable $cpactive -command "$this-c update_active $cp"
	set cpusemat "$var-usemat"
	checkbutton $w.mat -text "Use Material" \
	    -variable $cpusemat -command "$this-c update_usemat $cp"
	
	pack $w.scale -side left -expand 1
	pack $w.act $w.mat -side top -anchor w -fill x -expand 1
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

	frame $w.cut
	
	frame $w.cut.xa
	addCutPlane $w.cut.xa "xa" "X Low"
	
	frame $w.cut.xb
	addCutPlane $w.cut.xb "xb" "X High"
	
	frame $w.cut.ya
	addCutPlane $w.cut.ya "ya" "Y Low"
	
	frame $w.cut.yb
	addCutPlane $w.cut.yb "yb" "Y High"
	
	frame $w.cut.za
	addCutPlane $w.cut.za "za" "Z Low"
	
	frame $w.cut.zb
	addCutPlane $w.cut.zb "zb" "Z High"
	
	pack $w.cut.xa $w.cut.xb $w.cut.ya $w.cut.yb $w.cut.za $w.cut.zb \
	    -side top -expand 1 -fill x
	
	frame $w.isosurface
	scale $w.isosurface.slider -variable $this-iso_val \
	    -label "Isovalue" -showvalue true -orient horizontal \
	    -relief groove -length 200 -from 0 -to 255 \
	    -tickinterval 80 -resolution 0.01 \
	    -command "$this-c update_isosurface_value"

	frame $w.isosurface.color
	addColorSelection $w.isosurface.color "Isosurface Color" $this-color \
	    "update_isosurface_material"

	pack $w.isosurface.slider $w.isosurface.color \
	    -side top -expand 1 -fill x


	frame $w.buttons
	button $w.buttons.dismiss -text "Dismiss" -command "wm withdraw $w"
	pack $w.buttons.dismiss \
		-side left -padx 10

	pack $w.cut $w.isosurface $w.buttons -side top -pady 5
    }
}
