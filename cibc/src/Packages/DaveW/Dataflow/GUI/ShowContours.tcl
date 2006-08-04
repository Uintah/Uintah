##
 #  CStoGeom.tcl
 #  Written by:
 #   David Weinstein
 #   Department of Computer Science
 #   University of Utah
 #   February 1999
 #  Copyright (C) 1999 SCI Group
 ##

catch {rename DaveW_FEM_CStoGeom ""}

itcl_class DaveW_FEM_CStoGeom {
    inherit Module
    constructor {config} {
        set name CStoGeom
        set_defaults
    }
    method set_defaults {} {
        global $this-showLines
        global $this-showPoints
        set $this-showLines 1
        set $this-showPoints 0
    }
    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }

        toplevel $w
        wm minsize $w 300 80
        frame $w.f
        pack $w.f -padx 2 -pady 2 -side top -expand yes
        global $this-showLines
        global $this-showPoints
	checkbutton $w.f.n -variable $this-showLines -text "Show lines?"
	checkbutton $w.f.e -variable $this-showPoints -text "Show points?"
	pack $w.f.n $w.f.e -expand 1 -fill x
	pack $w.f -expand 1 -fill x
    }
}
