##
 #  VecVec.tcl: Vector - Vector operations
 #
 #  Written by:
 #   David Weinstein
 #   Department of Computer Science
 #   University of Utah
 #   October 2000
 #
 #  Copyright (C) 2000 SCI Group
 # 
 #  Log Information:
 #
 ##

catch {rename VecVec ""}

itcl_class PSECommon_Matrix_VecVec {
    inherit Module
    constructor {config} {
        set name VecVec
        set_defaults
    }
    method set_defaults {} {	
	global $this-opTCL
        set $this-opTCL plus
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
		{{"a + b" plus} \
		{"a - b" minus} \
		{"cat (a  b)" cat}}
	
	pack $w.f -expand 1 -fill both
    }
}
