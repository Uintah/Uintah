
#
#  TransformField.tcl
#
#  Written by:
#   David Weinstein
#   Department of Computer Science
#   University of Utah
#   December 1995
#
#  Copyright (C) 1995 SCI Group
#

itcl_class PSECommon_Fields_TransformField {
    inherit Module
    constructor {config} {
	set name TransformField
	set_defaults
    }
    
    method set_defaults {} {
    }
    
    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w		    
	    return;
	}
	toplevel $w
	wm minsize $w 100 100
	frame $w.f
	label $w.f.l -text "Field Map"
	pack $w.f.l -side top -fill both -expand 1
	frame $w.f.m -relief sunken -bd 2
	frame $w.f.m.l
	button $w.f.m.l.x -text "FlipX" -command "$this-c flipx" -padx 8
	button $w.f.m.l.y -text "FlipY" -command "$this-c flipy" -padx 8
	button $w.f.m.l.z -text "FlipZ" -command "$this-c flipz" -padx 8
	pack $w.f.m.l.x $w.f.m.l.y $w.f.m.l.z -side top
	pack $w.f.m.l -side left -expand 1 -fill x
	frame $w.f.m.r
	global $this-xmap
	global $this-ymap
	global $this-zmap
	set $this-xmap "x <- x+"
	set $this-ymap "y <- y+"
	set $this-zmap "z <- z+"
	label $w.f.m.r.x -textvariable $this-xmap
	label $w.f.m.r.y -textvariable $this-ymap
	label $w.f.m.r.z -textvariable $this-zmap
	pack $w.f.m.r.x $w.f.m.r.y $w.f.m.r.z -side top
	pack $w.f.m.r -side left -expand 1 -fill x
	pack $w.f.m -side top -fill x -expand 1
	frame $w.f.b -relief sunken -bd 2
	frame $w.f.b.l
	frame $w.f.b.r
	button $w.f.b.l.cp -text "Cycle+" -command "$this-c cyclePos"
	button $w.f.b.l.cn -text "Cycle-" -command "$this-c cycleNeg"
	button $w.f.b.l.res -text "Reset" -command "$this-c reset"
	button $w.f.b.r.sxy -text "SwapXY" -command "$this-c swapXY"
	button $w.f.b.r.syz -text "SwapYZ" -command "$this-c swapYZ"
	button $w.f.b.r.sxz -text "SwapXZ" -command "$this-c swapXZ"
	pack $w.f.b.l.cp $w.f.b.l.cn $w.f.b.l.res -side top -expand 1 -fill both
	pack $w.f.b.r.sxy $w.f.b.r.syz $w.f.b.r.sxz -side top -expand 1 -fill both
	pack $w.f.b.l $w.f.b.r -side left -expand 1 -fill both
	pack $w.f.b -side bottom -fill both -expand 1
	pack $w.f -side right -fill both -expand 1
    }
}
