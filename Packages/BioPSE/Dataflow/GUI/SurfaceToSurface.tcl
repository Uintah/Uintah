#
#  The contents of this file are subject to the University of Utah Public
#  License (the "License"); you may not use this file except in compliance
#  with the License.
#  
#  Software distributed under the License is distributed on an "AS IS"
#  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
#  License for the specific language governing rights and limitations under
#  the License.
#  
#  The Original Source Code is SCIRun, released March 12, 2001.
#  
#  The Original Source Code was developed by the University of Utah.
#  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
#  University of Utah. All Rights Reserved.
#

#
#  The contents of this file are subject to the University of Utah Public
#  License (the "License"); you may not use this file except in compliance
#  with the License.
#  
#  Software distributed under the License is distributed on an "AS IS"
#  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
#  License for the specific language governing rights and limitations under
#  the License.
#  
#  The Original Source Code is SCIRun, released March 12, 2001.
#  
#  The Original Source Code was developed by the University of Utah.
#  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
#  University of Utah. All Rights Reserved.
#

##
 #  SurfaceToSurface.tcl: Inverse surface to surface solution
 #  Written by:
 #   David Weinstein
 #   Department of Computer Science
 #   University of Utah
 #   Aug 1996
 #  Copyright (C) 1996 SCI Group
 ##

catch {rename BioPSE_Inverse_SurfaceToSurface ""}

itcl_class BioPSE_Inverse_SurfaceToSurface {
    inherit Module
    constructor {config} {
        set name SurfaceToSurface
        set_defaults
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
