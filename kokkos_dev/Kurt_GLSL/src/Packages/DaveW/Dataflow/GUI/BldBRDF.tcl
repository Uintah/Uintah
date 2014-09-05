
#  BldBRDF.tcl
#  Written by:
#   David Weinstein
#   Department of Computer Science
#   University of Utah
#   March 1997
#  Copyright (C) 1997 SCI Group

#source $sci_root/TCL/MaterialEditor.tcl

catch {rename DaveW_CS684_BldBRDF ""}

itcl_class DaveW_CS684_BldBRDF {
    inherit Module
    constructor {config} {
	set name BldBRDF
	set_defaults
    }
    
    method set_defaults {} {
	global $this-theta_expr
	global $this-phi_expr
	set $this-theta_expr "x"
	set $this-phi_expr "y"
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
	global $this-theta_expr
	global $this-phi_expr
# build two labels -- one for theta_expr and one for phi_expr
	pack $w.f -fill x -expand 1
    }
}
