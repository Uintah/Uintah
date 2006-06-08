
catch {rename Yarden_Visulaization_Sage ""}

itcl_class Yarden_Visualization_Sage {
    inherit Module

    constructor {config} {
	set name Sage
	set_defaults
	puts "cons done"
    }
    method set_defaults {} {
	puts "set defaults"
	global $this-isoval_min $this-isoval_max 
	global $this-visibility $this-value $this-scan
	global $this-bbox
	global $this-cutoff_depth 
	global $this-reduce
	global $this-cover
	global $this-all
	global $this-continuous
	global $this-rebuild

	set $this-isoval_min 0
	set $this-isoval_max 4095
	set $this-visiblilty 0
	set $this-value 1
	set $this-scan 1
	set $this-bbox 1
	set $this-reduce 0
	set $this-cover 0
	set $this-all 1
	set $this-continuous 0
	set $this-rebuild 0
    }

    method raiseGL {} {
	set w .ui[modname]
	if {[winfo exists $w.gl]} {
	    raise $w.gl
	} else {
	    toplevel $w.gl
	    wm title $w.gl "Sage2 Tiles"
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
	global $this-reduce $this-cover $this-all
	global $this-continuous
	global $this-rebuild

	puts "Sage ui"

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

	puts "****SAGE  [set $this-isoval_min] [set $this-isoval_max]"
# 	set $this-isoval_min 0
# 	set $this-isoval_max 4095
# 	puts "****SAGE2   [set $this-isoval_min] [set $this-isoval_max]"
	
	scale $w.f.isoval -label "Iso Value:" \
	    -variable $this-isoval \
	    -from [set $this-isoval_min] -to [set $this-isoval_max] \
	    -length 5c \
	    -showvalue true \
	    -orient horizontal  \
	    -digits 5 -resolution 0.1 \
	    -command "$this change_isoval"

# "$this change_isoval"

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

	checkbutton $w.f.prune.cover -text "cover" -relief flat \
		-variable $this-cover
 
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

	checkbutton $w.f.prune.rebuild -text "rebuild" -relief flat \
		-variable $this-rebuild


	scale $w.f.prune.depth -label "Depth:" \
	    -variable $this-cutoff_depth \
	    -from 0 -to 8 \
	    -length 2c \
	    -showvalue true \
	    -orient horizontal  

	pack $w.f.prune -side bottom

	pack $w.f.prune.reduce $w.f.prune.cover $w.f.prune.skip \
	    $w.f.prune.scan $w.f.prune.value $w.f.prune.visibility \
	    $w.f.prune.all $w.f.prune.rebuild $w.f.prune.depth \
	    -padx 2 -pady 3 -expand 1 -fill x  -side left
	
	raiseGL
	puts "ready"
	
	
    }
    
    method change_isoval { n } {
	global $this-isoval
        global $this-continuous
	global .ui[modname].f.isoval
	
#	puts "[set $this-continuous]"
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


