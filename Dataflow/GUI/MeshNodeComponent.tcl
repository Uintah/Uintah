##
 #  MeshNodeCore/CCA/Component.tcl: Make a column vector of x, y, or z positions
 #  Written by:
 #   David Weinstein
 #   Department of Computer Science
 #   University of Utah
 #   May 2000
 #  Copyright (C) 2000 SCI Group
 #  Log Information:
 ##

catch {rename MeshNodeCore/CCA/Component ""}

itcl_class Dataflow_Mesh_MeshNodeComponent {
    inherit Module
    constructor {config} {
        set name MeshNodeCore/CCA/Component
        set_defaults
    }
    method set_defaults {} {
	global $this-compTCL
	set $this-compTCL z
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
	global $this-compTCL
	make_labeled_radio $w.f.c "Core/CCA/Component: " "" \
		top $this-compTCL \
		{{"X" x} \
		{"Y" y} \
		{"Z" z}}
	pack $w.f.c -side top -fill x
        pack $w.f -side top -expand yes
    }
}
