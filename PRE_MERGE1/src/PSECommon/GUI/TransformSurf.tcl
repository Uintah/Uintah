#
#  TransformSurf.tcl
#
#  Written by:
#   David Weinstein
#   Department of Computer Science
#   University of Utah
#   July 1995
#
#  Copyright (C) 1995 SCI Group
#

itcl_class PSECommon_Surface_TransformSurf {
    inherit Module
    constructor {config} {
	set name TransformSurf
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
	global $this-coreg
	global $this-tx
	global $this-ty
	global $this-tz
	global $this-origin
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
	set $this-coreg 0
	set $this-tx 0
	set $this-ty 0
	set $this-tz 0
	set $this-origin 0
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
	frame $w.f.t -relief groove -borderwidth 5
#	expscale $w.f.t.x -orient horizontal -variable $this-tx \
#		-label "Translate X:"
#	expscale $w.f.t.y -orient horizontal -variable $this-ty \
#		-label "Translate Y:"
#	expscale $w.f.t.z -orient horizontal -variable $this-tz \
#		-label "Translate Z:"
#	pack $w.f.t.x $w.f.t.y $w.f.t.z -side top -fill x
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
		-label "Rotate Theta:" -showvalue true -from -7.0 -to 7.0 \
		-resolution .01
	pack $w.f.r.x $w.f.r.y $w.f.r.z $w.f.r.th -fill x -expand 1 -side top
	pack $w.f.r -side top -fill x -expand 1

	global $this-scale
	global $this-coreg
	global $this-scalex
	global $this-scaley
	global $this-scalez
	frame $w.f.s -relief groove -borderwidth 5
	label $w.f.s.l -text "Log Scale: "
	scale $w.f.s.s -variable $this-scale -orient horizontal \
		-from -3.000 -to 3.000 -resolution .001 -showvalue true
	frame $w.f.sx -relief groove -borderwidth 5
	label $w.f.sx.l -text "Log ScaleX: "
	scale $w.f.sx.s -variable $this-scalex -orient horizontal \
		-from -3.000 -to 3.000 -resolution .001 -showvalue true
	frame $w.f.sy -relief groove -borderwidth 5
	label $w.f.sy.l -text "Log ScaleY: "
	scale $w.f.sy.s -variable $this-scaley -orient horizontal \
		-from -3.000 -to 3.000 -resolution .001 -showvalue true
	frame $w.f.sz -relief groove -borderwidth 5
	label $w.f.sz.l -text "Log ScaleZ: "
	scale $w.f.sz.s -variable $this-scalez -orient horizontal \
		-from -3.000 -to 3.000 -resolution .001 -showvalue true
	pack $w.f.s.l -side left
	pack $w.f.s.s -side left -expand 1 -fill x
	pack $w.f.sx.l -side left
	pack $w.f.sx.s -side left -expand 1 -fill x
	pack $w.f.sy.l -side left
	pack $w.f.sy.s -side left -expand 1 -fill x
	pack $w.f.sz.l -side left
	pack $w.f.sz.s -side left -expand 1 -fill x
	frame $w.f.flip -relief groove -borderwidth 5
	global $this-flipx
	global $this-flipy
	global $this-flipz
	checkbutton $w.f.flip.x -variable $this-flipx -text "Flip X"
	checkbutton $w.f.flip.y -variable $this-flipy -text "Flip Y"
	checkbutton $w.f.flip.z -variable $this-flipz -text "Flip Z"
	pack $w.f.flip.x $w.f.flip.y $w.f.flip.z -padx 3 -fill x -expand 1
	checkbutton $w.f.coreg -variable $this-coreg -text "Coreg Sized" \
		-command "$this-c needexecute"
	checkbutton $w.f.origin -variable $this-origin -text "Centered About Crigin" \
		-command "$this-c needexecute"
	pack $w.f.s $w.f.sx $w.f.sy $w.f.sz $w.f.flip $w.f.origin $w.f.coreg -side top \
		-fill x -expand 1
	button $w.f.b -text "Transform" -command "$this-c needexecute"
	pack $w.f.b -side top
	pack $w.f -fill x -expand 1 -side top
    }	
}
