
catch {rename IsoSurfaceSAGE ""}

itcl_class Yarden_Visualization_IsoSurfaceSAGE {
    inherit Module

    constructor {config} {
	set name Sage
	set_defaults
    }
    method set_defaults {} {
	global $this-isoval_min $this-isoval_max 
	global $this-visibility $this-value $this-scan
	global $this-bbox
	global $this-cutoff_depth 
	global $this-reduce
	global $this-all
	global $this-continuous
	global $this-rebuild
	global $this-min_size
	global $this-poll

	set $this-isoval_min 0
	set $this-isoval_max 4095
	set $this-visiblilty 0
	set $this-value 1
	set $this-scan 1
	set $this-bbox 1
	set $this-reduce 1
	set $this-all 0
	set $this-continuous 0
	set $this-rebuild 0
	set $this-min_size 1
	set $this-poll 0
    }

    method raiseGL {} {
	set w .ui[modname]
	if {[winfo exists $w.gl]} {
	    raise $w.gl
	} else {
	    toplevel $w.gl
	    wm title $w.gl "Sage Tiles"
	    opengl $w.gl.gl -geometry 512x512 -doublebuffer false \
		-rgba true  -depthsize 0 \
		-redsize 4 
#	    opengl $w.gl.gl -geometry 512x512  -visual 42
	    bind $w.gl.gl <Button> "$this-c redraw "
	    pack $w.gl.gl -fill both -expand 1
	}
    }

    method ui {} {
	global $this-isoval_min $this-isoval_max 
	global $this-cutoff_depth $this-bbox
	global $this-reduce $this-all
	global $this-continuous
	global $this-rebuild

	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w		    
#	    raiseGL
	    return;
	}      

	toplevel $w
	frame $w.f 
	pack $w.f -padx 2 -pady 2 -expand 1 -fill x
	set n "$this-c needexecute "

	scale $w.f.isoval -label "Iso Value:" \
	    -variable $this-isoval \
	    -from [set $this-isoval_min] -to [set $this-isoval_max] \
	    -length 5c \
	    -showvalue true \
	    -orient horizontal  \
	    -digits 5 \
	    -command "$this change_isoval"

	pack $w.f.isoval -fill x

	trace variable $this-isoval_min w "$this change_isoval_min"
	trace variable $this-isoval_max w "$this change_isoval_max"

	button $w.f.reset_view -text "Reset View" -relief raised -command $n

	checkbutton $w.f.continuous -text "continuous" -relief flat \
		-variable $this-continuous

	pack $w.f.reset_view $w.f.continuous -side left

	frame $w.f.prune -relief ridge

	label $w.f.prune.title -text "Prune:"


	checkbutton $w.f.prune.reduce -text "reduce" -relief flat \
		-variable $this-reduce

	checkbutton $w.f.prune.skip -text "bbox" -relief flat \
		-variable $this-bbox 

	checkbutton $w.f.prune.visibility -text "visibility" -relief flat \
		-variable $this-visibility

	checkbutton $w.f.prune.scan -text "scan" -relief flat \
		-variable $this-scan 

	checkbutton $w.f.prune.value -text "value" -relief flat \
		-variable $this-value 

	checkbutton $w.f.prune.all -text "all" -relief flat \
		-variable $this-all 

	checkbutton $w.f.prune.size -text "min size" -relief flat \
		-variable $this-min_size

	checkbutton $w.f.prune.poll -text "poll" -relief flat \
		-variable $this-poll


	pack $w.f.prune -side bottom

	pack $w.f.prune.reduce $w.f.prune.skip \
	    $w.f.prune.scan $w.f.prune.value $w.f.prune.visibility \
	    $w.f.prune.all $w.f.prune.size $w.f.prune.poll \
	    -padx 2 -pady 3 -expand 1 -fill x  -side left
	
#	raiseGL
    }
    
    method change_isoval { n } {
	global $this-isoval
        global $this-continuous
	global .ui[modname].f.isoval
	
	if { [set $this-continuous] == 1.0 } {
	    eval "$this-c needexecute"
	}
    }

    method change_isoval_min {n1 n2 op} {
	global $this-isoval_min
	global .ui[modname].f.isoval
	.ui[modname].f.isoval configure -from [set $this-isoval_min]
	puts "change_min [set $this-isoval_min]"
    }
    
    method change_isoval_max {n1 n2 op} {
	global $this-isoval_max
	global .ui[modname].f.isoval
	.ui[modname].f.isoval configure -to [set $this-isoval_max]
    }
    
} 


