##
 #  CStoSFRG.tcl: Turn a contour set into a seg field
 #  Written by:
 #   David Weinstein
 #   Department of Computer Science
 #   University of Utah
 #   June 1999
 #  Copyright (C) 1999 SCI Group
 ##

catch {rename DaveW_FEM_CStoSFRG ""}

itcl_class DaveW_FEM_CStoSFRG {
    inherit Module
    constructor {config} {
        set name CStoSFRG
        set_defaults
    }
    method set_defaults {} {
	global $this-nxTCL
	set $this-nxTCL "16"
	global $this-nyTCL
	set $this-nyTCL "16"
    }
    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }

        toplevel $w
        wm minsize $w 100 30
        frame $w.f
        set n "$this-c needexecute "
	global $this-nxTCL
	global $this-nyTCL
	frame $w.f.nx
	label $w.f.nx.l -text "NX: "
	entry $w.f.nx.e -relief sunken -width 4 -textvariable $this-nxTCL
	pack $w.f.nx.l $w.f.nx.e -side left
	frame $w.f.ny
	label $w.f.ny.l -text "NY: "
	entry $w.f.ny.e -relief sunken -width 4 -textvariable $this-nyTCL
	pack $w.f.ny.l $w.f.ny.e -side left
	pack $w.f.nx $w.f.ny -side top
        pack $w.f -side top -expand yes
    }
}
