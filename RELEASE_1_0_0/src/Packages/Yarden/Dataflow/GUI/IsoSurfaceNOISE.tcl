
catch {rename Noise ""}

itcl_class Yarden_Visualization_IsoSurfaceNOISE {
    inherit Module
    constructor {config} {
	set name IsoSurfaceNOISE
	set_defaults
    }
    method set_defaults {} {
	global $this-isoval
	global $this-isoval_min
	global $this-isoval_max
        global $this-continuous
	global $this-alpha
	global $this-trans
	global $this-bbox
	global $this-np
	global $this-dl

	set $this-isoval 0;
	set $this-isoval_min 0.1
	set $this-isoval_max 100.1
        set $this-continuous 0
	set $this-alpha 0.2
	set $this-trans 0
	set $this-bbox 0
	set $this-dl 0
    }
    method ui {} {
	global $this-isoval
	global $this-isoval_min
	global $this-isoval_max
        global $this-continuous
	global $this-alpha
	global $this-trans
	global $this-bbox

	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	frame $w.f 
	pack $w.f -padx 2 -pady 2 -fill x
	set n "$this-c needexecute "

	frame $w.f.controls
	pack $w.f.controls -side top

	button $w.f.controls.run -text "Extract" -relief raised \
	    -command "$this-c needexecute"

	checkbutton $w.f.controls.continuous -text "continuous" -relief flat \
		-variable $this-continuous

	checkbutton $w.f.controls.bbox -text "bbox" -relief flat \
		-variable $this-bbox

	checkbutton $w.f.controls.dl -text "DL" -relief flat \
		-variable $this-dl

	checkbutton $w.f.controls.trans -text "Transparency" -relief flat \
		-variable $this-trans -command $n

	pack $w.f.controls.run $w.f.controls.continuous $w.f.controls.bbox \
	    $w.f.controls.trans $w.f.controls.dl -side left

	scale $w.f.np -label "Threads:" \
	    -variable $this-np \
	    -from 1 -to 60 \
	    -length 5c \
	    -showvalue true \
	    -orient horizontal  

	scale $w.f.isoval -label "Iso Value:" \
	    -variable $this-isoval \
	    -from [set $this-isoval_min] -to [set $this-isoval_max] \
	    -length 5c \
	    -showvalue true \
	    -orient horizontal  \
	    -digits 5 -resolution 0.0001 \
	    -command "$this change_isovalue"

	trace variable $this-isoval_min w "$this change_isoval_min"
	trace variable $this-isoval_max w "$this change_isoval_max"
	
	pack $w.f.np -side top  -fill x
	pack $w.f.isoval -side top  -fill x

	
	scale $w.f.alpha -label "Opacity" -from 0.0 -to 1.0 \
		-orient horizontal \
		-variable $this-alpha -resolution 0.001

	pack $w.f.alpha -side top -fill x
    }

    method change_isovalue { n } {
        global $this-continuous
	
	if { [set $this-continuous] == 1.0 } {
	    eval "$this-c needexecute"
	}
    }
    method change_isoval_min {n1 n2 op} {
	global $this-isoval_min
	global .ui$this.f.isoval
	.ui$this.f.isoval configure -from [set $this-isoval_min]
	puts "change_min [set $this-isoval_min]"
    }
    
    method change_isoval_max {n1 n2 op} {
	global $this-isoval_max
	global .ui$this.f.isoval
	.ui$this.f.isoval configure -to [set $this-isoval_max]
    }
    
}
