#  XYZtoRGB.tcl
#  Written by:
#   David Weinstein
#   Department of Computer Science
#   University of Utah
#   March 1997
#  Copyright (C) 1997 SCI Group

catch {rename DaveW_CS684_XYZtoRGB ""}

itcl_class DaveW_CS684_XYZtoRGB {
    inherit Module
    constructor {config} {
	set name XYZtoRGB
	set_defaults
    }
    
    method set_defaults {} {
	global $this-max
	global $this-wx $this-rx $this-gx $this-bx
	global $this-wy $this-ry $this-gy $this-by
	set $this-wx 0.406
	set $this-wy 0.344
	set $this-rx 0.670
	set $this-ry 0.330
	set $this-gx 0.210
	set $this-gy 0.710
	set $this-bx 0.140
	set $this-by 0.080
	set $this-max 0.026
	set $this-meth Custom
    }
    
    method raiseGL {} {
	set w .ui[modname]
	if {[winfo exists $w.gl]} {
	    raise $w.gl
	} else {
	    toplevel $w.gl
	    wm title $w.gl "XYZtoRGB Image"
	    opengl $w.gl.gl -geometry 512x512 -doublebuffer true -direct false -rgba true -redsize 2 -greensize 2 -bluesize 2 -depthsize 0
	    bind $w.gl.gl <Expose> "$this-c redraw"
	    pack $w.gl.gl -fill both -expand 1
	}
    }
	
    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w		    
	    raiseGL
	    return;
	}
	toplevel $w
	wm minsize $w 550 100
	frame $w.f
	global $this-wx $this-wy $this-rx $this-ry $this-gx $this-gy
	global $this-bx $this-by $this-meth
	
	make_labeled_radio $w.f.r "Method:" "$this new_method" \
		left $this-meth \
		{{Glassner "Glassner"} \
		{Bourke "Bourke"} \
		{Custom "Custom"}}
	set $this-meth "Custom"
	scale $w.f.rx -variable $this-rx -from 0.00 -to 1.00 \
		-orient horizontal -showvalue true -label "RX" \
		-resolution 0.001
	scale $w.f.ry -variable $this-ry -from 0.00 -to 1.00 \
		-orient horizontal -showvalue true -label "RY" \
		-resolution 0.001
	scale $w.f.gx -variable $this-gx -from 0.00 -to 1.00 \
		-orient horizontal -showvalue true -label "GX" \
		-resolution 0.001
	scale $w.f.gy -variable $this-gy -from 0.00 -to 1.00 \
		-orient horizontal -showvalue true -label "GY" \
		-resolution 0.001
	scale $w.f.bx -variable $this-bx -from 0.00 -to 1.00 \
		-orient horizontal -showvalue true -label "BX" \
		-resolution 0.001
	scale $w.f.by -variable $this-by -from 0.00 -to 1.00 \
		-orient horizontal -showvalue true -label "BY" \
		-resolution 0.001
	scale $w.f.wx -variable $this-wx -from 0.00 -to 1.00 \
		-orient horizontal -showvalue true -label "WX" \
		-resolution 0.001
	scale $w.f.wy -variable $this-wy -from 0.00 -to 1.00 \
		-orient horizontal -showvalue true -label "WY" \
		-resolution 0.001
	scale $w.f.m -variable $this-max -from 0.00 -to 0.30 \
		-orient horizontal -showvalue true -label "Max" \
		-resolution 0.001
	frame $w.f.bu
	button $w.f.bu.ex -text "Execute" -command "$this-c tcl_exec"
	button $w.f.bu.save -text "Save" -command "$this-c save"
	pack $w.f.bu.ex $w.f.bu.save -side left -expand 1 -pady 4
	pack $w.f.r $w.f.rx $w.f.ry $w.f.gx $w.f.gy $w.f.bx $w.f.by \
		$w.f.wx $w.f.wy $w.f.m $w.f.bu -side top -fill x
	pack $w.f -fill x -expand 1
	raiseGL
	new_method
    }
    
    method new_method {} {
	set w .ui[modname]
	global $this-meth
	global $this-wx $this-rx $this-gx $this-bx
	global $this-wy $this-ry $this-gy $this-by
	if {[set $this-meth] == "Glassner"} {
	    set $this-wx 0.406
	    set $this-wy 0.344
	    set $this-rx 0.670
	    set $this-ry 0.330
	    set $this-gx 0.210
	    set $this-gy 0.710
	    set $this-bx 0.140
	    set $this-by 0.080
	} {
	    if {[set $this-meth] == "Bourke"} {
		set $this-wx 0.313
		set $this-wy 0.329
		set $this-rx 0.628
		set $this-ry 0.346
		set $this-gx 0.268
		set $this-gy 0.588
		set $this-bx 0.150
		set $this-by 0.070
	    }
	}
	set $this-meth "Custom"
    }
}

