##
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
 #  MatrixSend.tcl: Send matrix to a host:port
 #  Written by:
 #   Oleg
 #   Department of Computer Science
 #   University of Utah
 #   01Jan05
 ##

catch {rename MatlabInterface_DataIO_Matlab ""}

itcl_class MatlabInterface_DataIO_Matlab {
    inherit Module
    constructor {config} {
        set name Matlab
        set_defaults
    }
    method set_defaults {} {
        global $this-cmdTCL
	global $this-hpTCL
	set $this-cmdTCL ""
	set $this-hpTCL "127.0.0.1:5517"
    }
    method ui {} {
        set n "$this-c needexecute "
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }

	global $this-cmdTCL
        global $this-hpTCL
	
        toplevel $w
        wm minsize $w 100 30
        
        frame $w.f
        pack $w.f -side top -fill both -expand yes
	
        frame $w.f.hp
        frame $w.f.cmd
	pack $w.f.hp $w.f.cmd -side top -fill both -expand yes

	label $w.f.hp.l -text "host:port : "
	entry $w.f.hp.e -relief sunken -width 21 -textvariable $this-hpTCL
	pack $w.f.hp.l -side left -padx 5 -pady 5
	pack $w.f.hp.e -side left -fill x -expand yes -padx 5 -pady 5

	label $w.f.cmd.l -text "command : "
	entry $w.f.cmd.e -relief sunken -width 21 -textvariable $this-cmdTCL 
        pack $w.f.cmd.l -side left -padx 5 -pady 5
	pack $w.f.cmd.e -side left -fill x -expand yes -padx 5 -pady 5
    }
}
