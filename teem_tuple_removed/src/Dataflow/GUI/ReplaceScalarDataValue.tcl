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

itcl_class SCIRun_FieldsData_ReplaceScalarDataValue {
    inherit Module
    constructor {config} {
        set name ReplaceScalarDataValue
        set_defaults
    }
    method set_defaults {} {
        global $this-function
	set $this-function "x+10"
    }
    method make_entry {w text v c} {
        frame $w
        label $w.l -text "$text"
        pack $w.l -side left
        global $v
        entry $w.e -textvariable $v
        bind $w.e <Return> $c
        pack $w.e -side right
    }
    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }

        toplevel $w
        wm minsize $w 170 20
        frame $w.f

	global $this-function
	make_entry $w.f.o "Old Value:" $this-oldvalue "$this-c needexecute"
	make_entry $w.f.n "New Value:" $this-newvalue "$this-c needexecute"
	pack $w.f.o $w.f.n -side left -expand 1 -fill x
	pack $w.f -side top -expand 1 -fill x
    }
}
