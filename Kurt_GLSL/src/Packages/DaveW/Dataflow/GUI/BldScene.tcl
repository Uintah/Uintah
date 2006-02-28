
#  BldScene.tcl
#  Written by:
#   David Weinstein
#   Department of Computer Science
#   University of Utah
#   March 1997
#  Copyright (C) 1997 SCI Group

catch {rename DaveW_CS684_BldScene ""}

itcl_class DaveW_CS684_BldScene {
    inherit Module
    constructor {config} {
	set name BldScene
	set_defaults
    }
    
    method set_defaults {} {
	global $this-nb
	global $this-atten
	global $this-tcl_exec
	global $this-material
	initMaterial $this-material
	set $this-nb 4
	set $this-atten 1
	set $this-tcl_exec 0
    }
	
    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w		    
	    return;
	}
	toplevel $w
	wm minsize $w 400 100
	frame $w.f
	global $this-nb
	global $this-atten
	frame $w.f.nb
	label $w.f.nb.l -text "Bounces"
	scale $w.f.nb.s -variable $this-nb -from 0 -to 15 \
		-orient horizontal -showvalue true
	pack $w.f.nb.l -side left
	pack $w.f.nb.s -side left -expand 1 -fill x
	frame $w.f.atten
	label $w.f.atten.l -text "Attenuate?"
	checkbutton $w.f.atten.s -variable $this-atten
	button $w.f.atten.ex -text "Execute" -command "$this-c tcl_exec"
        button $w.f.atten.m -text "Edit Material" -command "$this editMatl"
	button $w.f.atten.toggle -text "Toggle Visibility" -command \
		"$this-c toggle"
	pack $w.f.atten.l $w.f.atten.s $w.f.atten.ex $w.f.atten.m \
		$w.f.atten.toggle -side left -fill x
	pack $w.f.nb -side top -fill x
	pack $w.f.atten -side top
	pack $w.f -fill x -expand 1
	set $this-nb 4
	set $this-atten 1
    }

    method editMatl { } {
	$this-c getmatl
	set w .ui[modname]
	if {[winfo exists $w.matl]} {
	    raise $w.matl
	} else {
	    toplevel $w.matl
	    makeMaterialEditor $w.matl $this-material "$this matlCommit" "$this matlCancel"
	}
    }

    method matlCommit { } {
	$this-c setmatl
	destroy .ui[modname].matl
    }

    method matlCancel { } {
	destroy .ui[modname].matl
    }

}

proc initColor {c r g b} {
    global $c-r $c-g $c-b
    set $c-r $r
    set $c-g $g
    set $c-b $b
}

proc initMaterial {matter} {
    initColor $matter-ambient 0.0 0.0 0.0
    initColor $matter-diffuse 0.0 0.0 0.0
    initColor $matter-specular 0.0 0.0 0.0
    global $matter-shininess
    set $matter-shininess 10.0
    initColor $matter-emission 0.0 0.0 0.0
    global $matter-reflectivity
    set $matter-reflectivity 0.5
    global $matter-transparency
    set $matter-transparency 0
    global $matter-refraction_index
    set $matter-refraction_index 1.0
}
