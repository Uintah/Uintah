##
 #  MeshNodeComponent.tcl: Make a column vector of x, y, or z positions
 #
 #  Written by:
 #   David Weinstein
 #   Department of Computer Science
 #   University of Utah
 #   May 2000
 #
 #  Copyright (C) 2000 SCI Group
 # 
 #  Log Information:
 #
 #  $Log$
 #  Revision 1.1.2.1  2000/10/31 02:33:08  dmw
 #  Merging SCIRun changes in HEAD into FIELD_REDESIGN branch
 #
 #  Revision 1.1  2000/10/29 04:42:22  dmw
 #  MeshInterpVals -- fixed a bug
 #  MeshNodeComponent -- build a columnmatrix of the x/y/z position of the nodes
 #  MeshFindSurfNodes -- the surface nodes in a mesh
 #
 #
 #
 ##

catch {rename MeshNodeComponent ""}

itcl_class SCIRun_Mesh_MeshNodeComponent {
    inherit Module
    constructor {config} {
        set name MeshNodeComponent
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
	make_labeled_radio $w.f.c "Component: " "" \
		top $this-compTCL \
		{{"X" x} \
		{"Y" y} \
		{"Z" z}}
	pack $w.f.c -side top -fill x
        pack $w.f -side top -expand yes
    }
}
