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
        global $this-numberg
	global $this-nameg
	set $this-numberf 0
	set $this-namef ""
	set $this-numberg 0
	set $this-nameg ""
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
	frame $w.g
	global $this-numberf
	global $this-namef
	frame $w.g.num
	label $w.g.num.l -text "Number:"
	entry $w.g.num.e -relief sunken -width 3 -textvariable $this-numberg
	frame $w.g.name
	label $w.g.name.l -text "Name:"
	entry $w.g.name.e -relief sunken -width 10 -textvariable $this-nameg
	pack $w.g.num.l $w.g.num.e -side left
	pack $w.g.name.l $w.g.name.e -side left
	pack $w.g.num $w.g.name -padx 5 -side left -fill x
        pack $w.f $w.g -side top -expand yes
    }
}
