##
 #  EditPath.tcl
 #
 #  Written by:
 #   David Weinstein
 #   Department of Computer Science
 #   University of Utah
 #   February 1999
 #
 #  Copyright (C) 1999 SCI Group
 #
 ##

catch {rename EditPath ""}

itcl_class DaveW_Path_EditPath {
    inherit Module
    constructor {config} {
        set name EditPath
        set_defaults
    }
    method set_defaults {} {
        global $this-showNodes
        global $this-showElems
        set $this-showNodes 1
        set $this-showElems 0
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
        global $this-showNodes
        global $this-showElems
	checkbutton $w.f.n -variable $this-showNodes -text "Show nodes?"
	checkbutton $w.f.e -variable $this-showElems -text "Show elems?"
	pack $w.f.n $w.f.e -expand 1 -fill x
	pack $w.f -expand 1 -fill x
    }
}
