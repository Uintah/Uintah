
#  RTrace.tcl
#  Written by:
#   David Weinstein
#   Department of Computer Science
#   University of Utah
#   March 1997
#  Copyright (C) 1997 SCI Group

# source $sci_root/TCL/MaterialEditor.tcl

catch {rename DaveW_CS684_RTrace ""}

itcl_class DaveW_CS684_RTrace {
    inherit Module
    constructor {config} {
	set name RTrace
	set_defaults
    }
    
    method set_defaults {} {
	global $this-nx
	global $this-ny
	global $this-ns
	global $this-app
	global $this-tcl_exec
	global $this-abrt
	global $this-specMin
	global $this-specMax
	global $this-specNum
	global $this-bmin
	global $this-bmax
	global $this-tweak
	global $this-bGlMin
	global $this-bGlMax
	global $this-maxProc
	global $this-numProc
	global $this-scale

	set $this-nx 128
	set $this-ny 128
	set $this-ns 0
	set $this-app 0
	set $this-tcl_exec 0
	set $this-abrt 0
	set $this-specMin 300
	set $this-specMax 700
	set $this-specNum 41
	set $this-bmin 0.0
	set $this-bmax 0.42
	set $this-tweak 0
	set $this-bGlMin 0.0
	set $this-bGlMax 1.0
	set $this-numProc 1
	set $this-maxProc 1
	set $this-scale 1
    }
    
    method raiseGL {} {
	set w .ui[modname]
	if {[winfo exists $w.gl]} {
	    raise $w.gl
	} else {
	    toplevel $w.gl
	    wm title $w.gl "Raytraced Image"
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
	global $this-nx
	global $this-ny
	global $this-ns
	global $this-tweak
	global $this-maxProc
	trace variable $this-tweak w "$this changed_range"
	trace variable $this-maxProc w "$this changed_proc"
	frame $w.f.nx
	label $w.f.nx.l -text "X Size"
	scale $w.f.nx.s -variable $this-nx -from 1 -to 512 \
		-orient horizontal -showvalue true
	pack $w.f.nx.l -side left
	pack $w.f.nx.s -side left -expand 1 -fill x
	frame $w.f.ny
	label $w.f.ny.l -text "Y Size"
	scale $w.f.ny.s -variable $this-ny -from 1 -to 512 \
		-orient horizontal -showvalue true
	pack $w.f.ny.l -side left
	pack $w.f.ny.s -side left -expand 1 -fill x
	frame $w.f.sc
	label $w.f.sc.l -text "Scale"
	scale $w.f.sc.s -variable $this-scale -from 0.25 -to 20 \
		-orient horizontal -showvalue true -resolution 0.25 -digits 4
	pack $w.f.sc.l -side left
	pack $w.f.sc.s -side left -expand 1 -fill x
	frame $w.f.ns
	label $w.f.ns.l -text "Samples (sqrt)"
	scale $w.f.ns.s -variable $this-ns -from 0 -to 30 \
		-orient horizontal -showvalue true
	pack $w.f.ns.l -side left
	pack $w.f.ns.s -side left -expand 1 -fill x
	frame $w.f.app
	label $w.f.app.l -text "Apperture"
	scale $w.f.app.s -variable $this-app -from 0 -to 5.0 \
		-orient horizontal -showvalue true -resolution 0.05
	pack $w.f.app.l -side left
	pack $w.f.app.s -side left -expand 1 -fill x
	frame $w.f.spr
	global $this-specMin $this-specMax
	label $w.f.spr.l -text "Spectral Range"
	range $w.f.spr.r -var_min $this-specMin -var_max $this-specMax \
		-from 200 -to 1000 -nonzero 1 -showvalue true \
		-orient horizontal
	pack $w.f.spr.l -side left
	pack $w.f.spr.r -side left -expand 1 -fill x
	frame $w.f.sps
	global $this-specNum
	label $w.f.sps.l -text "Number of Spectral Samples"
	scale $w.f.sps.s -variable $this-specNum -from 2 -to 100 \
		-orient horizontal -showvalue true
	pack $w.f.sps.l -side left
	pack $w.f.sps.s -side left -expand 1 -fill x
	frame $w.f.color
	label $w.f.color.l -text "Color range:"
	global $this-bmin $this-bmax
	range $w.f.color.r -var_min $this-bmin -var_max $this-bmax \
		-from 0.0 -to 1.0 -nonzero 1 -showvalue true \
		-resolution .01 -orient horizontal
	pack $w.f.color.l -side left
	pack $w.f.color.r -side left -expand 1 -fill x
	frame $w.f.proc
	global $this-numProc
	global $this-maxProc
	label $w.f.proc.l -text "Number of Processors:"
	scale $w.f.proc.s -variable $this-numProc -from 1 -to \
		[set $this-maxProc] -orient horizontal -showvalue true
	pack $w.f.proc.l -side left
	pack $w.f.proc.s -side left -expand 1 -fill x
	frame $w.f.bu
	button $w.f.bu.ex -text "Execute" -command "$this-c tcl_exec"
	button $w.f.bu.save -text "Save Image" -command "$this-c save"
	button $w.f.bu.xyz -text "Send XYZ" -command "$this-c sendXYZ"
	button $w.f.bu.color -text "Color" -command "$this-c bound"
	global $this-abrt
        checkbutton $w.f.bu.a -text "Abort" -variable $this-abrt
	pack $w.f.bu.ex $w.f.bu.save $w.f.bu.xyz $w.f.bu.color $w.f.bu.a \
		-side left -expand 1 -pady 4 
	pack $w.f.nx $w.f.ny $w.f.sc $w.f.ns $w.f.app $w.f.spr $w.f.sps \
		$w.f.color $w.f.proc $w.f.bu -side top -fill x -expand 1
	pack $w.f -fill x -expand 1
	raiseGL
    }

    method changed_range {v vtmp op} {
	set w .ui[modname]
	global $this-bGlMin
	global $this-bGlMax
	$w.f.color.r configure -from [expr [set $this-bGlMin] / 4]
	$w.f.color.r configure -to [set $this-bGlMax]
    }   

    method changed_proc {v vtmp op} {
	set w .ui[modname]
	global $this-numProc
	global $this-maxProc
	$w.f.proc.s configure -to [set $this-maxProc]
	set $this-numProc [set $this-maxProc]
	puts -nonewline "NUMBER OF PROCESSORS = "
	puts [set $this-maxProc]
	puts -nonewline "USING: "
	puts [set $this-numProc]
    }
}
