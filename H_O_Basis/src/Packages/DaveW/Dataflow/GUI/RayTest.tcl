#  RayTest.tcl
#  Written by:
#   David Weinstein
#   Department of Computer Science
#   University of Utah
#   April 1997
#  Copyright (C) 1997 SCI Group

catch {rename DaveW_CS684_RayTest ""}

itcl_class DaveW_CS684_RayTest {
    inherit Module
    constructor {config} {
	set name RayTest
	set_defaults
    }
    
    method set_defaults {} {
	global $this-density
	global $this-nu
	global $this-energy
	set $this-density 2
	set $this-nu 1.05
	set $this-energy 0.90
    }

    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w		    
	    return;
	}
	set n "$this-c needexecute"
	toplevel $w
	frame $w.f
	global $this-density
	global $this-nu
	global $this-energy
	frame $w.f.density
	label $w.f.density.l -text "Density: "
	scale $w.f.density.s -variable $this-density -command $n \
	        -from 1 -to 10 -orient horizontal \
		-showvalue 1
	pack $w.f.density.l $w.f.density.s -side left -expand 1 -fill x
	frame $w.f.nu
	label $w.f.nu.l -text "Nu: "
	scale $w.f.nu.s -variable $this-nu -command $n \
		-from 0.5 -to 2.5 -orient horizontal \
		-resolution 0.01 -showvalue 1
	pack $w.f.nu.l $w.f.nu.s -side left -expand 1 -fill x
	frame $w.f.energy
	label $w.f.energy.l -text "Energy: "
	scale $w.f.energy.s -variable $this-energy -command $n \
		-from 0.05 -to 1.00 -orient horizontal \
		-resolution 0.05 -showvalue 1
	pack $w.f.energy.l $w.f.energy.s -side left -expand 1 -fill x
	pack $w.f.density $w.f.nu $w.f.energy -side top -expand 1 -fill x
	pack $w.f -expand 1 -fill x
	set $this-density 2
	set $this-nu 1.05
	set $this-energy 0.90
    }
}
