
#  Radiosity.tcl
#  Written by:
#   David Weinstein
#   Department of Computer Science
#   University of Utah
#   March 1997
#  Copyright (C) 1997 SCI Group

catch {rename DaveW_CS684_Radiosity}
itcl_class DaveW_CS684_Radiosity {
    inherit Module
    constructor {config} {
	set name Radiosity
	set_defaults
    }
    
    method set_defaults {} {
	global $this-ns
	global $this-nl
	global $this-ni
	global $this-cscale
	global $this-raderr
	global $this-nrmls
	global $this-lnx
	global $this-vissamp
	global $this-ffsamp
	set $this-ns 10
	set $this-nl 1
	set $this-ni 4
	set $this-raderr 4
	set $this-cscale 30
	set $this-nrmls 0
	set $this-lnx 1
	set $this-vissamp 16
	set $this-ffsamp 16
    }
	
    method raiseLinks {} {
        set w .ui[modname].links
        if {[winfo exists $w]} {
            raise $w
        } else {
            toplevel $w
	    frame $w.f
	    button $w.f.higher -text "Up" -command "$this-c uplevel"
	    button $w.f.lower -text "Down" -command "$this-c downlevel"
	    pack $w.f.higher $w.f.lower -side left -fill x
	    pack $w.f -fill x -expand 1
	}
    }

    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w		    
	    return;
	}
	toplevel $w
	wm minsize $w 400 100
	frame $w.f
	global $this-nl
	global $this-ns
	global $this-ni
	global $this-cscale
	global $this-raderr
	global $this-nrmls
	global $this-lnx
	global $this-vissamp
	global $this-ffsamp
	scale $w.f.nl -label "Number of initial levels in mesh patches:" -variable $this-nl -from 1 -to 10 \
		-orient horizontal -showvalue true
	scale $w.f.ns -label "Number of solver iterations per refinement step" -variable $this-ns -from 1 -to 10 \
		-orient horizontal -showvalue true
	scale $w.f.ni -label "Number of refinement iterations per execute" -variable $this-ni -from 1 -to 10 \
		-orient horizontal -showvalue true
	scale $w.f.er -label "Radiance error" -variable $this-raderr -from 0.0000001 -to 0.1 \
		-digits 9 -resolution 0.0000001 -orient horizontal -showvalue true
	scale $w.f.cs -label "Image brightness scale" -variable $this-cscale -from 0.1 -to 100 \
		-digits 4 -resolution 0.1 -orient horizontal -showvalue true
	scale $w.f.vissamp -label "Number of visibility samples per link:" -variable $this-vissamp -from 10 -to 100 \
		-orient horizontal -showvalue true
	scale $w.f.ffsamp -label "Number of Monte Carlo form factor samples per link:" -variable $this-ffsamp -from 10 -to 100 \
		-orient horizontal -showvalue true
	button $w.f.r -text "Reset" -command "$this-c reset"
	button $w.f.e -text "Execute" -command "$this-c needexecute"
	button $w.f.d -text "Remove All Links" -command "$this-c remove"
	button $w.f.a -text "Add All Links" -command "$this-c add"
	frame $w.f.b -borderwidth 2 -relief raised
	checkbutton $w.f.b.l -text "Show Links" -variable $this-lnx -command "$this-c redraw"
	checkbutton $w.f.b.n -text "Show Normals" -variable $this-nrmls -command "$this-c redraw"
	pack $w.f.b.l $w.f.b.n -side left -expand 1 -fill x
	button $w.f.c -text "Check Form Factors" -command "$this-c checkff"
	button $w.f.re -text "Redraw" -command "$this-c redraw"
	frame $w.f.lev
	button $w.f.lev.higher -text "Up" -command "$this-c uplevel"
	button $w.f.lev.lower -text "Down" -command "$this-c downlevel"
	pack $w.f.lev.higher $w.f.lev.lower -side left -fill x -expand 1
	pack $w.f.nl $w.f.ns $w.f.ni $w.f.er $w.f.cs $w.f.vissamp $w.f.ffsamp $w.f.e $w.f.r $w.f.d $w.f.a $w.f.c $w.f.re $w.f.lev $w.f.b -side top -expand 1 -fill x
	pack $w.f -side top -expand 1 -fill x
    }
}
