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

catch {rename MatlabInterface_Math_Focusing ""}

itcl_class MatlabInterface_Math_Focusing {
    inherit Module
    constructor {config} {
        set name Focusing
        set_defaults
    }
    method set_defaults {} {
	global $this-noiseGUI
	global $this-fcsdgGUI
	set $this-noiseGUI "0.01"
	set $this-fcsdgGUI "5"
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
	global $this-noiseGUI
	global $this-fcsdgGUI
	frame $w.f.noise
	label $w.f.noise.l -text "Noise: "
	entry $w.f.noise.e -relief sunken -width 20 \
		-textvariable $this-noiseGUI
	pack $w.f.noise.l $w.f.noise.e -side left
	frame $w.f.fcsdg
	label $w.f.fcsdg.l -text "FCSDG: "
	entry $w.f.fcsdg.e -relief sunken -width 20 \
		-textvariable $this-fcsdgGUI
	pack $w.f.fcsdg.l $w.f.fcsdg.e -side left
	pack $w.f.noise $w.f.fcsdg -side top
        pack $w.f -side top -expand yes
    }
}
