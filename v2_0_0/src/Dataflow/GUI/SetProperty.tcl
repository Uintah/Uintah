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
 #  SetProperty.tcl: The SetProperty UI
 #  Written by:
 #   David Weinstein
 #   Department of Computer Science
 #   University of Utah
 #   Jan 2000
 #  Copyright (C) 2000 SCI Group
 ##

catch {rename SCIRun_FieldsOther_SetProperty ""}

itcl_class SCIRun_FieldsOther_SetProperty {
    inherit Module
    constructor {config} {
        set name SetProperty
        set_defaults
    }
    method set_defaults {} {
        global $this-prop
        global $this-val
        global $this-meshprop
        set $this-prop units
        set $this-val cm
        set $this-meshprop 1
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
        wm minsize $w 200 80
        frame $w.f
        pack $w.f -padx 2 -pady 2 -side top -expand yes
	global $this-prop
	global $this-val
	global $this-meshprop
	make_entry $w.f.p "Property:" $this-prop "$this-c needexecute"
	make_entry $w.f.v "Value:" $this-val "$this-c needexecute"
	make_labeled_radio $w.f.m "Property belongs to:" "" \
		top $this-meshprop \
		{{"Field" 0} \
		{"Mesh" 1}}
	button $w.f.b -text "Execute" -command "$this-c needexecute"
	pack $w.f.p $w.f.v $w.f.m $w.f.b -side top -expand 1 -fill x
	pack $w.f -expand 1 -fill x
    }
}
