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
 #  Written by:
 #   Oleg
 #   Department of Computer Science
 #   University of Utah
 #   01May25
 ##

catch {rename MatlabInterface_Math_Tikhonov ""}

itcl_class MatlabInterface_Math_Tikhonov {
    inherit Module
    constructor {config} {
        set name Tikhonov
        set_defaults
    }
    method set_defaults {} {
	global $this-hpTCL
	set $this-hpTCL ""
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
	global $this-hpTCL
	frame $w.f.hp
	label $w.f.hp.l -text "pars: "
	entry $w.f.hp.e -relief sunken -width 21 -textvariable $this-hpTCL
	pack $w.f.hp.l $w.f.hp.e -side left
	pack $w.f.hp -side top
        pack $w.f -side top -expand yes
    }
}
