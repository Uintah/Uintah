##
 #  MatVec.tcl: Matrix - Matrix operations
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

catch {rename MatVec ""}

itcl_class PSECommon_Matrix_MatVec {
    inherit Module
    constructor {config} {
        set name MatVec
        set_defaults
    }
    method set_defaults {} {	
	global $this-opTCL
        set $this-opTCL AtimesB
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
		{{"A x b" AtimesB} \
		{"A^T x b" AtTimesB}}
	
	pack $w.f -expand 1 -fill both
    }
}
