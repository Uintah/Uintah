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
#    File   : EditTupleAxis.tcl
#    Author : David Weinstein
#    Date   : Mon Sep 22 09:46:23 2003

catch {rename Teem_NrrdData_EditTupleAxis ""}

itcl_class Teem_NrrdData_EditTupleAxis {
    inherit Module
    constructor {config} {
        set name EditTupleAxis
        set_defaults
    }
    method set_defaults {} {
        global $this-input-label
        set $this-input-label ""

        global $this-output-label
        set $this-output-label ""


    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }

        toplevel $w

        frame $w.f
	pack $w.f -padx 2 -pady 2 -side top -expand yes
	
	frame $w.f.options
	pack $w.f.options -side top -expand yes

        iwidgets::entryfield $w.f.options.input -labeltext "input tuple string:" -textvariable $this-input-label
        pack $w.f.options.input -side top -expand yes -fill x
        iwidgets::entryfield $w.f.options.output -labeltext "output tuple string:" -textvariable $this-output-label
        pack $w.f.options.output -side top -expand yes -fill x

	button $w.f.b -text "Execute" -command "$this-c needexecute"
	pack $w.f.b -side top -expand 1 -fill x
	pack $w.f -expand 1 -fill x
    }
}
