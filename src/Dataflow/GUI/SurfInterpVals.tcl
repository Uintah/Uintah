##
 #  SurfInterpVals.tcl: General surface interpolation module
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
 #  Revision 1.1  1999/07/27 16:58:30  mcq
 #  Initial commit
 #
 #  Revision 1.1.1.1  1999/04/24 23:12:35  dav
 #  Import sources
 #
 #  Revision 1.1  1999/01/04 05:32:33  dmw
 #  See Dave for details...
 #
 #  Revision 1.1  1997/08/23 06:27:19  dweinste
 #  Some trivial modules that I needed...
 #
 #
 ##

itcl_class PSECommon_Surface_SurfInterpVals {
    inherit Module
    constructor {config} {
        set name SurfInterpVals
        set_defaults
    }
    method set_defaults {} {
	global $this-method
	set $this-method volumeblur
	global $this-numnbrs
	set $this-numnbrs 5
	global $this-surfid
	set $this-surfid ""
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
		{{"S2->S1 N-nbr weighted volume blur" volblur} \
		{"S2->S1 Project" project} \
		{"S2->S1 ProjectNormal" projectNormal} \
		{"S1->S1 N-nbr voronoi surface blur" surfblur}}
	global $this-numnbrs
	scale $w.f.numnbrs -variable $this-numnbrs -orient horizontal \
		-from 1 -to 10 -showvalue true
	global $this-surfid
	frame $w.f.surfid
	label $w.f.surfid.l -text "SurfTree SurfId: "
	entry $w.f.surfid.e -relief sunken -width 10 -textvariable $this-surfid
	pack $w.f.surfid.l $w.f.surfid.e -side left
	
	pack $w.f.method $w.f.numnbrs $w.f.surfid -side top -fill x
        pack $w.f -side top -expand yes
    }
}
