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
	global $this-tx
	global $this-ty
	global $this-tz
	global $this-sha
	global $this-shb
	global $this-shc
	global $this-shd
	global $this-pre
	global $this-lastxform
	global $this-whichxform
	global $this-xmapTCL
	global $this-ymapTCL
	global $this-zmapTCL
	global $this-widgetAutoUpdate
	global $this-widgetIgnoreChanges
	global $this-widgetScale
	set $this-rx 0
	set $this-ry 0
	set $this-rz 1
	set $this-th 0
	set $this-scale 0
	set $this-scalex 0
	set $this-scaley 0
	set $this-scalez 0
	set $this-tx 0
	set $this-ty 0
	set $this-tz 0
	set $this-sha 0
	set $this-shb 0
	set $this-shc 1
	set $this-sha 0
	set $this-lastxform translate
	set $this-whichxform 0
	set $this-pre 1
	set $this-xmapTCL 1
	set $this-ymapTCL 2
	set $this-zmapTCL 3
	set $this-widgetAutoUpdate 0
	set $this-widgetScale 1
	set $this-widgetIgnoreChanges 1
    }
    method ui {} {
	set w .ui[modname]
		if {[winfo exists $w]} {
	    raise $w
	    return;
	}

	toplevel $w
	wm minsize $w 200 50

	set mouseXLoc [winfo pointerx .]
	set mouseYLoc [winfo pointery .]
	wm geometry $w +$mouseXLoc+$mouseYLoc

	frame $w.f
	pack $w.f -side top -fill x -padx 2 -pady 2

	global $this-tx
	global $this-ty
	global $this-tz
	global $this-pre
	global $this-whichxform

	frame $w.f.which
	radiobutton $w.f.which.trans -command "$this setxform $w translate" \
		-text Translate -variable $this-whichxform -value 0
	radiobutton $w.f.which.scale -command "$this setxform $w scale" \
		-text Scale -variable $this-whichxform -value 1
	radiobutton $w.f.which.rot -command "$this setxform $w rotate" \
		-text Rotate -variable $this-whichxform -value 2
	radiobutton $w.f.which.shear -command "$this setxform $w shear" \
		-text Shear -variable $this-whichxform -value 3
	radiobutton $w.f.which.permute -command "$this setxform $w permute" \
		-text Permute -variable $this-whichxform -value 4
	radiobutton $w.f.which.widget -command "$this setxform $w widget" \
		-text Widget -variable $this-whichxform -value 5
	pack $w.f.which.trans $w.f.which.scale $w.f.which.rot \
		$w.f.which.shear $w.f.which.permute $w.f.which.widget \
		-side left -fill x -expand 1

	frame $w.f.b 
	button $w.f.b.doit -text "Apply Transform" -command \
		"$this-c needexecute"
	button $w.f.b.comp -text "Composite Transform" -command "$this-c composite; $this setxform $w translate; $this set_defaults"
	pack $w.f.b.doit $w.f.b.comp -side left -fill x -padx 10 -pady 3

	frame $w.f.prepost
	radiobutton $w.f.prepost.pre -variable $this-pre \
		-text "Pre-multiply" -value 0
	radiobutton $w.f.prepost.post -variable $this-pre \
		-text "Post-multiply" -value 1
	pack $w.f.prepost.pre $w.f.prepost.post -side left \
		-fill x -expand 1

	pack $w.f.which -side top -fill x -expand 1
	pack $w.f.b -side top
	pack $w.f.prepost -side top -fill x -expand 1

	frame $w.f.t -relief groove -borderwidth 5
	label $w.f.t.l -text "Translate Vector"
	frame $w.f.t.f
	expscale $w.f.t.f.x -orient horizontal -variable $this-tx \
		-label "X:"
	expscale $w.f.t.f.y -orient horizontal -variable $this-ty \
		-label "Y:"
	expscale $w.f.t.f.z -orient horizontal -variable $this-tz \
		-label "Z:"
	pack $w.f.t.f.x $w.f.t.f.y $w.f.t.f.z -side top -fill x
	pack $w.f.t.l $w.f.t.f -side top -fill both -expand 1
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

	global $this-xstr
	global $this-ystr
	global $this-zstr
	$this bldAllMapStr
	frame $w.f.p -relief groove -borderwidth 5
	label $w.f.p.l -text "Field Map"
	pack $w.f.p.l -side top -fill both -expand 1
	frame $w.f.p.m -relief sunken -bd 2
	frame $w.f.p.m.l
	button $w.f.p.m.l.x -text "FlipX" -command "$this flipx" -padx 8
	button $w.f.p.m.l.y -text "FlipY" -command "$this flipy" -padx 8
	button $w.f.p.m.l.z -text "FlipZ" -command "$this flipz" -padx 8
	pack $w.f.p.m.l.x $w.f.p.m.l.y $w.f.p.m.l.z -side top
	pack $w.f.p.m.l -side left -expand 1 -fill x
	frame $w.f.p.m.r
	label $w.f.p.m.r.x -textvariable $this-xstr
	label $w.f.p.m.r.y -textvariable $this-ystr
	label $w.f.p.m.r.z -textvariable $this-zstr
	pack $w.f.p.m.r.x $w.f.p.m.r.y $w.f.p.m.r.z -side top
	pack $w.f.p.m.r -side left -expand 1 -fill x
	pack $w.f.p.m -side top -fill x -expand 1
	frame $w.f.p.b -relief sunken -bd 2
	frame $w.f.p.b.l
	frame $w.f.p.b.r
	button $w.f.p.b.l.cp -text "Cycle+" -command "$this cyclePos"
	button $w.f.p.b.l.cn -text "Cycle-" -command "$this cycleNeg"
	button $w.f.p.b.l.res -text "Reset" -command "$this reset"
	button $w.f.p.b.r.sxy -text "SwapXY" -command "$this swapXY"
	button $w.f.p.b.r.syz -text "SwapYZ" -command "$this swapYZ"
	button $w.f.p.b.r.sxz -text "SwapXZ" -command "$this swapXZ"
	pack $w.f.p.b.l.cp $w.f.p.b.l.cn $w.f.p.b.l.res -side top -expand 1 -fill both
	pack $w.f.p.b.r.sxy $w.f.p.b.r.syz $w.f.p.b.r.sxz -side top -expand 1 -fill both
	pack $w.f.p.b.l $w.f.p.b.r -side left -expand 1 -fill both
	pack $w.f.p.b -side bottom -fill both -expand 1
	
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
	pack $w.f.s.g.l -side left
	pack $w.f.s.g.s -side left -expand 1 -fill x
	pack $w.f.s.sx.l -side left
	pack $w.f.s.sx.s -side left -expand 1 -fill x
	pack $w.f.s.sy.l -side left
	pack $w.f.s.sy.s -side left -expand 1 -fill x
	pack $w.f.s.sz.l -side left
	pack $w.f.s.sz.s -side left -expand 1 -fill x
	pack $w.f.s.g $w.f.s.sx $w.f.s.sy $w.f.s.sz -side top -fill x -expand 1

	global $this-sha
	global $this-shb
	global $this-shc
	global $this-shd
	frame $w.f.sh -relief groove -borderwidth 5
	label $w.f.sh.l -text "Shear Fixed Plane"
	scale $w.f.sh.a -orient horizontal -variable $this-sha \
		-label "A:" -showvalue true -from -1.0 -to 1.0 \
		-resolution .01
	scale $w.f.sh.b -orient horizontal -variable $this-shb \
		-label "B:" -showvalue true -from -1.0 -to 1.0 \
		-resolution .01
	scale $w.f.sh.c -orient horizontal -variable $this-shc \
		-label "C:" -showvalue true -from -1.0 -to 1.0 \
		-resolution .01
	expscale $w.f.sh.d -orient horizontal -variable $this-shd \
		-label "D:"
	pack $w.f.sh.l $w.f.sh.a $w.f.sh.b $w.f.sh.c -fill x \
		-expand 1 -side top
	pack $w.f.sh.d -fill x \
		-expand 1 -side bottom
	
	global $this-widgetScale
	global $this-widgetAutoUpdate
	global $this-widgetIgnoreChanges
	frame $w.f.w -relief groove -borderwidth 5
	label $w.f.w.l -text "Widget"
	pack $w.f.w.l -side top
	expscale $w.f.w.d -orient horizontal -variable $this-widgetScale \
		-label "Uniform Scale"
	frame $w.f.w.b 
        set $this-widgetAutoUpdate 0
        set $this-widgetChangeIgnore 1

	checkbutton $w.f.w.b.auto -text "Auto Update" \
		-variable $this-widgetAutoUpdate \
		-command "$this change_autoupdate"
	checkbutton $w.f.w.b.ignore -text "Ignore Changes" \
		-variable $this-widgetIgnoreChanges \
		-command "$this change_ignore"
	pack $w.f.w.b.auto $w.f.w.b.ignore -side left -fill x \
		-expand 1
	pack $w.f.w.d $w.f.w.b -side top -fill x -expand 1

	pack $w.f -fill x -expand 1 -side top
	if {[set $this-whichxform] == 0} {
	    $this setxform $w translate
	} elseif {[set $this-whichxform] == 1} {
	    $this setxform $w scale
	} elseif {[set $this-whichxform] == 2} {
	    $this setxform $w rotate
	} elseif {[set $this-whichxform] == 3} {
	    $this setxform $w shear
	} elseif {[set $this-whichxform] == 4} {
	    $this setxform $w permute
	} elseif {[set $this-whichxform] == 5} {
	    $this setxform $w widget
	} else {
	    puts "BldTransform.tcl::ui setxform [set $this-whichxform] not recognized"
	}
    }	

    method change_ignore { } {
	global $this-widgetIgnoreChanges
	$this-c change_ignore [set $this-widgetIgnoreChanges]
    }

    method change_autoupdate { } {
	global $this-widgetAutoUpdate
	$this-c change_autoupdate [set $this-widgetAutoUpdate]
    }

    method setxform {w t} {
	global $this-lastxform
	if {[set $this-lastxform] == $t} return
	if {[set $this-lastxform] == "rotate"} {pack forget $w.f.r}
	if {[set $this-lastxform] == "scale"} {pack forget $w.f.s}
	if {[set $this-lastxform] == "shear"} {pack forget $w.f.sh}
	if {[set $this-lastxform] == "permute"} {
	    pack forget $w.f.p
	    pack $w.f.t -side top -fill x -expand 1
	}
	if {[set $this-lastxform] == "widget"} {
	    pack forget $w.f.w
	    pack $w.f.t -side top -fill x -expand 1
	    $this-c hide_widget
	}
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
	    $w.f.t.l configure -text "Shear Vector"
	    pack $w.f.sh -side top -fill x -expand 1
	    return
	}
	if {$t == "permute"} {
	    pack forget $w.f.t
	    pack $w.f.p -side top -fill x -expand 1
	    $this-c show_widget
	    return
	}
	if {$t == "widget"} {
	    pack forget $w.f.t
	    pack $w.f.w -side top -fill x -expand 1
	    $this-c show_widget
	}
    }

    method valToStr { v } {
	if {$v == 1} {
	    return x+
	}
	if {$v == -1} {
	    return x-
	}
	if {$v == 2} {
	    return y+
	}
	if {$v == -2} {
	    return y-
	}
	if {$v == 3} {
	    return z+
	}
	return "z-"
    }

    method bldAllMapStr { } {
	global $this-xmapTCL
	global $this-ymapTCL
	global $this-zmapTCL
	global $this-xstr
	global $this-ystr
	global $this-zstr
	
	set xx [$this valToStr [set $this-xmapTCL]]
	set yy [$this valToStr [set $this-ymapTCL]]
	set zz [$this valToStr [set $this-zmapTCL]]
	set $this-xstr "x <- $xx"
	set $this-ystr "y <- $yy"
	set $this-zstr "z <- $zz"
    }

    method flipx { } {
	global $this-xmapTCL
	set $this-xmapTCL [expr [set $this-xmapTCL] * -1]
	$this bldAllMapStr
    }
    
    method flipy { } {
	global $this-ymapTCL
	set $this-ymapTCL [expr [set $this-ymapTCL] * -1]
	$this bldAllMapStr
    }
    
    method flipz { } {
	global $this-zmapTCL
	set $this-zmapTCL [expr [set $this-zmapTCL] * -1]
	$this bldAllMapStr
    }

    method cyclePos { } {
	global $this-xmapTCL
	global $this-ymapTCL
	global $this-zmapTCL
	set tmp [set $this-xmapTCL]
	set $this-xmapTCL [set $this-ymapTCL]
	set $this-ymapTCL [set $this-zmapTCL]
	set $this-zmapTCL $tmp
	$this bldAllMapStr
    }

    method cycleNeg { } {
	global $this-xmapTCL
	global $this-ymapTCL
	global $this-zmapTCL
	set tmp [set $this-zmapTCL]
	set $this-zmapTCL [set $this-ymapTCL]
	set $this-ymapTCL [set $this-xmapTCL]
	set $this-xmapTCL $tmp
	$this bldAllMapStr
    }

    method reset { } {
	global $this-xmapTCL
	global $this-ymapTCL
	global $this-zmapTCL
	set $this-xmapTCL 1
	set $this-ymapTCL 2
	set $this-zmapTCL 3
	$this bldAllMapStr
    }

    method swapXY { } {
	global $this-xmapTCL
	global $this-ymapTCL
	set tmp [set $this-xmapTCL]
	set $this-xmapTCL [set $this-ymapTCL]
	set $this-ymapTCL $tmp
	$this bldAllMapStr
    }

    method swapXZ { } {
	global $this-xmapTCL
	global $this-zmapTCL
	set tmp [set $this-xmapTCL]
	set $this-xmapTCL [set $this-zmapTCL]
	set $this-zmapTCL $tmp
	$this bldAllMapStr
    }

    method swapYZ { } {
	global $this-ymapTCL
	global $this-zmapTCL
	set tmp [set $this-ymapTCL]
	set $this-ymapTCL [set $this-zmapTCL]
	set $this-zmapTCL $tmp
	$this bldAllMapStr
    }

}

