itcl_class PSECommon_Fields_FieldGainCorrect {
    inherit Module
    constructor {config} {
	set name FieldGainCorrect
	set_defaults
    }
    method set_defaults {} {
	global $this-filterType
	global $this-Offset
	set $this-filterType Subtract
	set $this-Offset 50
    }
    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	wm minsize $w 100 50
	frame $w.f
	frame $w.f.filt
	global $this-filterType
	global $this-Offset
        make_labeled_radio $w.f.filt.b "Filter:" "" top $this-filterType \
                {Subtract Divide Hybrid}
	pack $w.f.filt.b -fill both -expand 1
	scale $w.f.s -label "Hybrid Offset" -from 1 -to 250 -showvalue true \
		-orient horizontal -variable $this-Offset
	pack $w.f.filt $w.f.s -side top -fill both -expand 1
	pack $w.f
    }
}
