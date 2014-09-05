#
#

itcl_class SCIRun_Image_ImageSel {
    inherit Module
    constructor {config} {
	set name ImageSel
	set_defaults
    }
    method set_defaults {} {
	global $this-sel
	set $this-sel 0
    }

    method ui {} {
	set w .ui[modname]
		if {[winfo exists $w]} {
	    raise $w
	    return;
	}

	toplevel $w
	wm minsize $w 400 200    
	frame $w.f -width 400
	pack $w.f -padx 2 -pady 2 -fill x
	set n "$this-c needexecute "

	label $w.label -text "Image Sel Module"
	pack $w.label -fill x -pady 2
#	expscale $w.low -orient horizontal -label "Which Image? " -variable $this-sel -command $n
#	$w.low-win- configure
#	pack $w.low -fill x -pady 2
    }
}

