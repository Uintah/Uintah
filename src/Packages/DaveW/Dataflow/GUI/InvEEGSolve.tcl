##
 #  Coregister.tcl: The coregistration UI
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

catch {rename DaveW_EEG_InvEEGSolve ""}

itcl_class DaveW_EEG_InvEEGSolve {
    inherit Module
    constructor {config} {
        set name InvEEGSolve
        set_defaults
    }
    method modname {} {
	set n $this
	if {[string first "::" "$n"] == 0} {
	    set n "[string range $n 2 end]"
	}
	return $n
    }
    method set_defaults {} {
        global $this-status
        global $this-maxiter
	global $this-target_error
	global $this-iteration
	global $this-current_error
	set $this-status "ok"
	set $this-maxiter 1000
	set $this-target_error 0.001
	set $this-iteration 0
	set $this-current_error 100
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
        set n "$this-c needexecute "
        global $this-status
        global $this-maxiter
	global $this-target_error
	global $this-iteration
	global $this-current_error
	scale $w.f.mi -orient horizontal -label "Max Iters to Convergence: "\
		-variable $this-maxiter -showvalue true \
		-from 1 -to 10000
	scale $w.f.te -orient horizontal -label "Target Error: " \
		-variable $this-target_error -showvalue true \
		-from 0 -to .999 -resolution 0.001
	pack $w.f.mi $w.f.te -side top -expand 1 -fill x
	pack $w.f -expand 1 -fill x
    }
}
