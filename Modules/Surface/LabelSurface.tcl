##
 #  LabelSurface.tcl: Label a specific surface in a surftree
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
 #  Revision 1.2  1999/01/04 05:32:31  dmw
 #  See Dave for details...
 #
 #  Revision 1.1  1997/08/23 06:27:19  dweinste
 #  Some trivial modules that I needed...
 #
 #
 ##

itcl_class LabelSurface {
    inherit Module
    constructor {config} {
        set name LabelSurface
        set_defaults
    }
    method set_defaults {} {
        global $this-numberf
	global $this-namef
	set $this-numberf 0
	set $this-namef ""
    }
    method ui {} {
        set w .ui$this
        if {[winfo exists $w]} {
            raise $w
            return;
        }

        toplevel $w
        wm minsize $w 200 50
        frame $w.f
        set n "$this-c needexecute "
	global $this-numberf
	global $this-namef
	frame $w.f.num
	label $w.f.num.l -text "Number:"
	entry $w.f.num.e -relief sunken -width 3 -textvariable $this-numberf
	frame $w.f.name
	label $w.f.name.l -text "Name:"
	entry $w.f.name.e -relief sunken -width 10 -textvariable $this-namef
	pack $w.f.num.l $w.f.num.e -side left
	pack $w.f.name.l $w.f.name.e -side left
	pack $w.f.num $w.f.name -padx 5 -side left -fill x
        pack $w.f -side top -expand yes
    }
}
