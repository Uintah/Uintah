#  RayMatrix.tcl
#  Written by:
#   David Weinstein
#   Department of Computer Science
#   University of Utah
#   March 1997
#  Copyright (C) 1997 SCI Group

catch {rename DaveW_CS684_RayMatrix ""}

itcl_class DaveW_CS684_RayMatrix {
    inherit Module
    constructor {config} {
	set name RayMatrix
	set_defaults
    }
    
    method set_defaults {} {
	global $this-tclSpectrum
	global $this-tclFname
	global $this-tclLType
	global $this-tclSpec
	global $this-tclMin
	global $this-tclMax
	global $this-tclNum
	global $this-scale

	set $this-tclSpectrum ""
	set $this-tclFname "./"
	set $this-tclLType ambient
	set $this-tclSpec 0.3
	set $this-tclMin 300
	set $this-tclMax 750
	set $this-tclNum 20
	set $this-scale 1
    }
    
    method raiseGL {} {
	set w .ui[modname]
	if {[winfo exists $w.gl]} {
	    raise $w.gl
	} else {
	    toplevel $w.gl
	    wm title $w.gl "RayMatrix Image"
	    opengl $w.gl.gl -geometry 512x512 -doublebuffer true -direct false -rgba true -redsize 2 -greensize 2 -bluesize 2 -depthsize 0
	    bind $w.gl.gl <Expose> "$this-c redraw"
	    pack $w.gl.gl -fill both -expand 1
	}
    }
	
    method ui {} {
	global $this-tclSpectrum
	global $this-tclFname
	global $this-tclLType
	global $this-tclSpec
	global $this-tclMin
	global $this-tclMax
	global $this-tclNum

	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w		    
	    raiseGL
	    return;
	}
	toplevel $w
	wm minsize $w 200 50
	frame $w.f

	frame $w.f.l -relief groove -borderwidth 2
	make_labeled_radio $w.f.l.tr "Spectrum to change: " "" \
		top $this-tclLType \
		{{"Ambient" ambient} \
		{"Light" light} \
		{"Material" material}}
	frame $w.f.l.n
	label $w.f.l.n.l -text "Spectrum name: "
	entry $w.f.l.n.e -relief sunken -width 6 -textvariable $this-tclSpectrum
	pack $w.f.l.n.l $w.f.l.n.e -side left

	frame $w.f.l.b
	button $w.f.l.b.e -text "Execute" -command "$this-c ex"
	button $w.f.l.b.s -text "Save" -command "$this-c save"
	pack $w.f.l.b.e $w.f.l.b.s -side left -padx 7
	pack $w.f.l.tr $w.f.l.n $w.f.l.b -side top -fill y -pady 3
	
	frame $w.f.r -relief groove -borderwidth 2

	frame $w.f.r.t -relief groove -borderwidth 2

	frame $w.f.r.t.s
	label $w.f.r.t.s.l -text "Spec Coeff: "
	entry $w.f.r.t.s.e -relief sunken -width 4 -textvariable $this-tclSpec
	button $w.f.r.t.s.b -text "Change" -command "$this-c changespec"
	pack $w.f.r.t.s.l $w.f.r.t.s.e $w.f.r.t.s.b -side left -fill x -padx 3
	
#	frame $w.f.r.t.d
#	label $w.f.r.t.d.l -text "Diff Coeff: "
#	entry $w.f.r.t.d.e -relief sunken -width 4 -textvariable $this-tclDiff
#	button $w.f.r.t.d.b -text "Change" -command "$this-c changediff"
#	pack $w.f.r.t.d.l $w.f.r.t.d.e $w.f.r.t.d.b -side left -fill x -padx 3
	pack $w.f.r.t.s -side top -fill x -expand 1 -pady 3
#	pack $w.f.r.t.s $w.f.r.t.d -side top -fill x -expand 1 -pady 3

	frame $w.f.r.m -relief groove -borderwidth 2

	frame $w.f.r.m.l
	label $w.f.r.m.l.l -text "Spectrum Parameters"
	button $w.f.r.m.l.e -text "Recompute" -command "$this-c changesparam"
	pack $w.f.r.m.l.l $w.f.r.m.l.e -side left -fill x -padx 3
	frame $w.f.r.m.v
	label $w.f.r.m.v.minl -text "Min"
	entry $w.f.r.m.v.minv -relief sunken -width 4 \
		-textvariable $this-tclMin
	label $w.f.r.m.v.maxl -text "  Max"
	entry $w.f.r.m.v.maxv -relief sunken -width 4 \
		-textvariable $this-tclMax
	label $w.f.r.m.v.numl -text "  Num"
	entry $w.f.r.m.v.numv -relief sunken -width 3 \
		-textvariable $this-tclNum
	pack $w.f.r.m.v.minl $w.f.r.m.v.minv $w.f.r.m.v.maxl $w.f.r.m.v.maxv \
		$w.f.r.m.v.numl $w.f.r.m.v.numv -side left
	pack $w.f.r.m.l $w.f.r.m.v -side top -pady 3 -fill x -expand 1

	frame $w.f.r.b -relief groove -borderwidth 2
	
	label $w.f.r.b.l -text "File: "
	entry $w.f.r.b.e -relief sunken -width 20 -textvariable $this-tclFname
	button $w.f.r.b.b -text "Load" -command "$this-c changefile"
	pack $w.f.r.b.l $w.f.r.b.e $w.f.r.b.b -side left -fill x -padx 2
	
	pack $w.f.r.t $w.f.r.m $w.f.r.b -side top -fill y -fill x -expand 1

	pack $w.f.l $w.f.r -side left -fill both -expand 1

	pack $w.f -side top -expand 1 -fill both
	frame $w.sc
	label $w.sc.l -text "Scale"
	scale $w.sc.s -variable $this-scale -from 0.25 -to 20 \
		-orient horizontal -showvalue true -resolution 0.25 -digits 4
	pack $w.sc.l -side left
	pack $w.sc.s -side left -expand 1 -fill x
	pack $w.sc -side top -fill x
	button $w.b -text "Print Matrices" -command "$this-c print"
	pack $w.b -side top
	raiseGL
    }
}