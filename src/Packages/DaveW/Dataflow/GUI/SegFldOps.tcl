##
 #  SegFldOps.tcl: The segfld UI
 #
 #  Written by:
 #   David Weinstein
 #   Department of Computer Science
 #   University of Utah
 #   Aug 1996
 #
 #  Copyright (C) 1996 SCI Group
 # 
 ##

itcl_class SegFldOps {
    inherit Module
    method modname {} {
	set n $this
	if {[string first "::" "$n"] == 0} {
	    set n "[string range $n 2 end]"
	}
	return $n
    }
    constructor {config} {
        set name SegFldOps
        set_defaults
    }
    method set_defaults {} {
	global $this-itype
	global $this-meth
	global $this-annexSize
	global $this-sendCharFlag
	set $this-itype "char"
	set $this-meth "annex"
	set $this-annexSize 10
	set $this-sendCharFlag 0
    }
    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }

        toplevel $w
        wm minsize $w 350 150
        frame $w.f
        pack $w.f -padx 2 -pady 2 -side top -expand yes -fill both
        set n "$this-c needexecute "
	global $this-itype
	global $this-meth
	global $this-annexSize
	global $this-sendCharFlag
	make_labeled_radio $w.f.i "Input Port:" "" \
		left $this-itype \
		{{NewSegFld "newseg"} \
		{OldSegFld "oldseg"} \
		{SFRGChar "char"}}
	set $this-itype "char"
	make_labeled_radio $w.f.m "Method:" "" \
		left $this-meth \
		{{None "none"}
		{Annex "annex"}}
	set $this-meth "none"
	scale $w.f.s -label "Annex Scale:" -variable $this-annexSize \
		-orient horizontal -showvalue true -from 1 -to 400
	frame $w.f.b
	checkbutton $w.f.b.s -variable $this-sendCharFlag -text \
		"Send SFRGChar"
	button $w.f.b.p -text "Print" -command "$this-c print"
	button $w.f.b.a -text "Audit" -command "$this-c audit"
	button $w.f.b.e -text "Execute" -command "$this-c tcl_exec"
	button $w.f.b.c -text "Compress" -command "$this-c compress"
	pack $w.f.b.s $w.f.b.a $w.f.b.p $w.f.b.e $w.f.b.c -side left -fill x \
		-padx 4 -expand 1
	pack $w.f.i $w.f.m $w.f.s $w.f.b -side top -expand 1 -fill both
    }
}
