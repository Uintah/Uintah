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

itcl_class SCIRun_Fields_SeedField {
    inherit Module
    constructor {config} {
        set name SeedField

        set_defaults
    }

    method set_defaults {} {
	global $this-random_seed
	global $this-number_dipoles
	set $this-random_seed 12345678
	set $this-number_dipoles 1
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }
        toplevel $w

	frame $w.f1
	frame $w.f2
	frame $w.f3

	pack $w.f1 $w.f2 $w.f3 -side top -expand yes -fill x

	label $w.f1.l1 -text "Random Seed"
	entry $w.f1.e1 -textvariable $this-random_seed
	pack $w.f1.l1 -side left -expand yes
	pack $w.f1.e1 -side right -expand yes

	label $w.f2.l1 -text "Number of Dipoles"
	entry $w.f2.e1 -textvariable $this-number_dipoles
	pack $w.f2.l1 -side left -expand yes
	pack $w.f2.e1 -side right -expand yes

	button $w.f3.execute -text "Execute" -command "$this-c execute"
	pack $w.f3.execute -side left -expand yes
    }
}


