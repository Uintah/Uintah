#
#  CrosshairWidget.tcl
#
#  Written by:
#   James Purciful
#   Department of Computer Science
#   University of Utah
#   Apr. 1995
#
#  Copyright (C) 1995 SCI Group
#


catch {rename CrosshairWidget ""}

itcl_class CrosshairWidget {
    inherit BaseWidget
    constructor {config} {
	BaseWidget::constructor
	set name CrosshairWidget
    }

    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	wm minsize $w 100 100

	Dialbox $w.dialbox "CrosshairWidget - Translate"
	$w.dialbox unbounded_dial 0 "Translate X" 0.0 1.0 \
		"$this-c translate x" "$this-c dialdone"
	$w.dialbox unbounded_dial 2 "Translate Y" 0.0 1.0 \
		"$this-c translate y" "$this-c dialdone"
	$w.dialbox unbounded_dial 4 "Translate Z" 0.0 1.0 \
		"$this-c translate z" "$this-c dialdone"

	frame $w.f
	Base_ui $w.f
	button $w.f.dials -text "Dialbox" -command "$w.dialbox connect"
	pack $w.f.dials -pady 5
	pack $w.f
    }

    method scale_changed {newscale} {
	ui
	set w .ui[modname]
	$w.dialbox dial_scale 0 $newscale
	$w.dialbox dial_scale 2 $newscale
	$w.dialbox dial_scale 4 $newscale
    }
}
