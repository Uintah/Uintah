##
 #  MeshInterpVals.tcl: General Mesh interpolation module
 #
 #  Written by:
 #   David Weinstein
 #   Department of Computer Science
 #   University of Utah
 #   Jully 1997
 #
 #  Copyright (C) 1997 SCI Group
 # 
 #  Log Information:
 #
 #  $Log$
 #  Revision 1.2  1999/12/10 06:58:12  dmw
 #  added another flag to MeshInterpVals
 #
 #  Revision 1.1  1999/09/04 23:14:52  dmw
 #  took out unused files and added the Mesh module GUI's
 #
 #  Revision 1.1  1999/01/04 05:32:33  dmw
 #  See Dave for details...
 #
 #  Revision 1.1  1997/08/23 06:27:19  dweinste
 #  Some trivial modules that I needed...
 #
 #
 ##

catch {rename MeshInterpVals ""}

itcl_class SCIRun_Mesh_MeshInterpVals {
    inherit Module
    constructor {config} {
        set name MeshInterpVals
        set_defaults
    }
    method set_defaults {} {
	global $this-method
	set $this-method project
	global $this-zeroTCL
	set $this-zeroTCL 0
	global $this-potMatTCL
	set $this-potMatTCL 0
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
	global $this-method
	make_labeled_radio $w.f.method "Method: " "" \
		top $this-method \
		{{"S2->S1 Project" project}}
	global $this-zeroTCL
	checkbutton $w.f.zero -text "Don't use mesh node zero" -variable $this-zeroTCL
	global $this-potMatTCL
	checkbutton $w.f.pot -text "Build potential difference matrix (ground=0)" -variable $this-potMatTCL
	pack $w.f.method $w.f.zero $w.f.pot -side top -fill x
        pack $w.f -side top -expand yes
    }
}
