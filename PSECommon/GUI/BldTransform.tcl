#
#  BldTransform.tcl
#
#  Written by:
#   David Weinstein
#   Department of Computer Science
#   University of Utah
#   March 1999
#
#  Copyright (C) 1999 SCI Group
#
catch {rename BldTransform ""}

itcl_class PSECommon_Matrix_BldTransform {
    inherit Module
    constructor {config} {
	set name BldTransform
	set_defaults
    }
    method set_defaults {} {
	global $this-rx
	global $this-ry
	global $this-rz
	global $this-th
	global $this-scale 
	global $this-scalex
	global $this-scaley
	global $this-scalez
	global $this-flipx
	global $this-flipy
	global $this-flipz
	global $this-tx
	global $this-ty
	global $this-tz
	global $this-td
	global $this-shu
	global $this-shv
	global $this-pre
	global $this-lastxform
	global $this-whichxform
	set $this-rx 0
	set $this-ry 0
	set $this-rz 1
	set $this-th 0
	set $this-scale 0
	set $this-scalex 0
	set $this-scaley 0
	set $this-scalez 0
	set $this-flipx 0
	set $this-flipy 0
	set $this-flipz 0
	set $this-tx 0
	set $this-ty 0
	set $this-tz 0
	set $this-td 0
	set $this-pre 1
	set $this-shu 0
	set $this-shv 0
	set $this-lastxform translate
	set $this-whichxform 0
    }
    method ui {} {
	set w .ui[modname]
		if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	wm minsize $w 200 50

	frame $w.f
	pack $w.f -side top -fill x -padx 2 -pady 2

	global $this-tx
	global $this-ty
	global $this-tz
	global $this-pre
	global $this-whichxform

	frame $w.f.prepost
	radiobutton $w.f.prepost.pre -variable $this-pre \
		-text "Pre-multiply" -value 0
	radiobutton $w.f.prepost.post -variable $this-pre \
		-text "Post-multiply" -value 1
	button $w.f.prepost.b -text "DoIt" -command "$this-c needexecute"
	pack $w.f.prepost.pre $w.f.prepost.post $w.f.prepost.b -side left \
		-fill x -expand 1

	frame $w.f.which
	radiobutton $w.f.which.trans -command "$this setxform $w translate" \
		-text Translate -variable $this-whichxform -value 0
	radiobutton $w.f.which.scale -command "$this setxform $w scale" \
		-text Scale -variable $this-whichxform -value 1
	radiobutton $w.f.which.rot -command "$this setxform $w rotate" \
		-text Rotate -variable $this-whichxform -value 2
	radiobutton $w.f.which.shear -command "$this setxform $w shear" \
		-text Shear -variable $this-whichxform -value 3
	pack $w.f.which.trans $w.f.which.scale $w.f.which.rot \
		$w.f.which.shear -side left -fill x -expand 1
	pack $w.f.prepost $w.f.which -side top -fill x -expand 1

	frame $w.f.t -relief groove -borderwidth 5
	label $w.f.t.l -text "Translate"
	expscale $w.f.t.x -orient horizontal -variable $this-tx \
		-label "X:"
	expscale $w.f.t.y -orient horizontal -variable $this-ty \
		-label "Y:"
	expscale $w.f.t.z -orient horizontal -variable $this-tz \
		-label "Z:"
	pack $w.f.t.l $w.f.t.x $w.f.t.y $w.f.t.z -side top -fill x
	pack $w.f.t -side top -fill x -expand 1
	
	global $this-rx
	global $this-ry
	global $this-rz
	global $this-th
	frame $w.f.r -relief groove -borderwidth 5
	scale $w.f.r.x -orient horizontal -variable $this-rx \
		-label "Rotate Axis X:" -showvalue true -from -1.0 -to 1.0 \
		-resolution .01
	scale $w.f.r.y -orient horizontal -variable $this-ry \
		-label "Rotate Axis Y:" -showvalue true -from -1.0 -to 1.0 \
		-resolution .01
	scale $w.f.r.z -orient horizontal -variable $this-rz \
		-label "Rotate Axis Z:" -showvalue true -from -1.0 -to 1.0 \
		-resolution .01
	scale $w.f.r.th -orient horizontal -variable $this-th \
		-label "Rotate Theta (degrees):" -showvalue true -from -360.0 \
		-to 360.0 -resolution 0.1
	pack $w.f.r.x $w.f.r.y $w.f.r.z $w.f.r.th -fill x -expand 1 -side top

	global $this-scale
	global $this-scalex
	global $this-scaley
	global $this-scalez
	frame $w.f.s
	frame $w.f.s.g -relief groove -borderwidth 5
	label $w.f.s.g.l -text "Log Scale: "
	scale $w.f.s.g.s -variable $this-scale -orient horizontal \
		-from -3.000 -to 3.000 -resolution .001 -showvalue true
	frame $w.f.s.sx -relief groove -borderwidth 5
	label $w.f.s.sx.l -text "Log ScaleX: "
	scale $w.f.s.sx.s -variable $this-scalex -orient horizontal \
		-from -3.000 -to 3.000 -resolution .001 -showvalue true
	frame $w.f.s.sy -relief groove -borderwidth 5
	label $w.f.s.sy.l -text "Log ScaleY: "
	scale $w.f.s.sy.s -variable $this-scaley -orient horizontal \
		-from -3.000 -to 3.000 -resolution .001 -showvalue true
	frame $w.f.s.sz -relief groove -borderwidth 5
	label $w.f.s.sz.l -text "Log ScaleZ: "
	scale $w.f.s.sz.s -variable $this-scalez -orient horizontal \
		-from -3.000 -to 3.000 -resolution .001 -showvalue true
	global $this-flipx
	global $this-flipy
	global $this-flipz
	frame $w.f.s.flip -relief groove -borderwidth 5
	checkbutton $w.f.s.flip.x -variable $this-flipx -text "Flip X"
	checkbutton $w.f.s.flip.y -variable $this-flipy -text "Flip Y"
	checkbutton $w.f.s.flip.z -variable $this-flipz -text "Flip Z"
	pack $w.f.s.g.l -side left
	pack $w.f.s.g.s -side left -expand 1 -fill x
	pack $w.f.s.sx.l -side left
	pack $w.f.s.sx.s -side left -expand 1 -fill x
	pack $w.f.s.sy.l -side left
	pack $w.f.s.sy.s -side left -expand 1 -fill x
	pack $w.f.s.sz.l -side left
	pack $w.f.s.sz.s -side left -expand 1 -fill x
	pack $w.f.s.flip.x $w.f.s.flip.y $w.f.s.flip.z -padx 3 -fill x \
		-expand 1
	pack $w.f.s.g $w.f.s.sx $w.f.s.sy $w.f.s.sz $w.f.s.flip -side top \
		-fill x -expand 1

	global $this-shu
	global $this-shv
	global $this-td
	frame $w.f.sh -relief groove -borderwidth 5
	expscale $w.f.sh.d -orient horizontal -variable $this-td \
		-label "D:"
	scale $w.f.sh.u -orient horizontal -variable $this-shu \
		-label "Shear Vector U:" -showvalue true -from -1.0 -to 1.0 \
		-resolution .01
	scale $w.f.sh.v -orient horizontal -variable $this-shv \
		-label "Shear Vector V:" -showvalue true -from -1.0 -to 1.0 \
		-resolution .01
	pack $w.f.sh.d $w.f.sh.u $w.f.sh.v -fill x -expand 1 \
		-side top
	
	pack $w.f -fill x -expand 1 -side top
    }	

    method setxform {w t} {
	global $this-lastxform
	if {[set $this-lastxform] == $t} return
	if {[set $this-lastxform] == "rotate"} {pack forget $w.f.r}
	if {[set $this-lastxform] == "scale"} {pack forget $w.f.s}
	if {[set $this-lastxform] == "shear"} {pack forget $w.f.sh}

	set $this-lastxform $t
	
	if {$t == "translate"} {
	    $w.f.t.l configure -text "Translate Vector"
	    return
	}
	if {$t == "rotate"} {
	    $w.f.t.l configure -text "Rotation Fixed Point"
	    pack $w.f.r -side top -fill x -expand 1
	    return
	}
	if {$t == "scale"} {
	    $w.f.t.l configure -text "Scale Fixed Point"
	    pack $w.f.s -side top -fill x -expand 1
	    return
	}
	if {$t == "shear"} {
	    $w.f.t.l configure -text "Shear Plane"
	    pack $w.f.sh -side top -fill x -expand 1
	    return
	}
    }
}

