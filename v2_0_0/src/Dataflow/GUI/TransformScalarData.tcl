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

itcl_class SCIRun_FieldsData_TransformScalarData {
    inherit Module
    constructor {config} {
        set name TransformScalarData
        set_defaults
    }
    method set_defaults {} {
        global $this-method
        global $this-function
	global $this-imin
	global $this-imax
	global $this-omin
	global $this-omax
	set $this-method "function"
	set $this-function "x+10"
	set $this-imin 0
	set $this-imax 1
	set $this-omin 0
	set $this-omax 1
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

	global $this-method
	global $this-function
	global $this-imin
	global $this-imax
	global $this-omin
	global $this-omax

	set c "$this-c needexecute"

	frame $w.f.i -relief groove -borderwidth 2
	label $w.f.i.l1 -text "Original Min/Max         Min: "
	label $w.f.i.l2 -textvariable $this-imin
	label $w.f.i.l3 -text " Max: "
	label $w.f.i.l4 -textvariable $this-imax
	pack $w.f.i.l1 $w.f.i.l2 $w.f.i.l3 $w.f.i.l4 -side left -expand 1

	frame $w.f.f
	radiobutton $w.f.f.r -text "Function         " -anchor w -just left \
	    -variable $this-method -value "function"
	entry $w.f.f.e -textvariable $this-function
	bind $w.f.f.e <Return> $c
	pack $w.f.f.r $w.f.f.e -side left -fill x -expand 1

	frame $w.f.m
	radiobutton $w.f.m.r -text "New MinMax        " -anchor w -just left \
	    -variable $this-method -value "minmax"	
	label $w.f.m.l1 -text "Min: "
	entry $w.f.m.e1 -textvariable $this-omin -width 6
	bind $w.f.m.e1 <Return> $c
	label $w.f.m.l2 -text "Max: "
	entry $w.f.m.e2 -textvariable $this-omax -width 6
	bind $w.f.m.e2 <Return> $c
	pack $w.f.m.r $w.f.m.l1 $w.f.m.e1 $w.f.m.l2 $w.f.m.e2 -side left

	pack $w.f.i $w.f.f $w.f.m -side top -expand 1 -fill x
	pack $w.f -expand 1 -fill x
    }
}
