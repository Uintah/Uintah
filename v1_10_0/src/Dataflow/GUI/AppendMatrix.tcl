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

itcl_class SCIRun_Math_AppendMatrix {
    inherit Module
    constructor {config} {
        set name AppendMatrix
        set_defaults
    }
    method set_defaults {} {
        global $this-row
	set $this-row 0
        global $this-append
	set $this-append 0
        global $this-front
	set $this-front 0
        global $this-clear
	set $this-front 0
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
        wm minsize $w 150 20
        frame $w.f
        pack $w.f -padx 2 -pady 2 -side top -expand yes
        global $this-row
        make_labeled_radio $w.f.r "Rows/Columns" "" \
                top $this-row \
		{{"Row" 1} \
                {"Column" 0}}
        global $this-append
        make_labeled_radio $w.f.a "Append/Replace" "" \
                top $this-append \
		{{"Append" 1} \
                {"Replace" 0}}
        global $this-front
        make_labeled_radio $w.f.f "Prepend/Postpend" "" \
                top $this-front \
		{{"Prepend" 1} \
                {"Postpend" 0}}
	pack $w.f.r $w.f.a $w.f.f -side left -expand 1 -fill x

	button $w.clear -text "Clear Output" -command "$this-c clear"

	pack $w.f $w.clear -expand 1 -fill x
    }
}
