##
 #  MatMat.tcl: Matrix - Matrix operations
 #
 #  Written by:
 #   David Weinstein
 #   Department of Computer Science
 #   University of Utah
 #   July 1999
 #
 #  Copyright (C) 1999 SCI Group
 # 
 #  Log Information:
 #
 ##

catch {rename MatMat ""}

itcl_class PSECommon_Matrix_MatMat {
    inherit Module
    constructor {config} {
        set name MatMat
        set_defaults
    }
    method set_defaults {} {	
	global $this-opTCL
        set $this-opTCL AtimesBinv
    }
    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }

        toplevel $w
        wm minsize $w 150 30
	set n "$this-c needexecute "

     	global $this-opTCL
	make_labeled_radio $w.f "Operation: " " " \
		top $this-opTCL \
		{{"A x B^(-1)" AtimesBinv} \
		{"A + B" AplusB}}
	
	pack $w.f -expand 1 -fill both
    }
}
