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


catch {rename BioPSE_Forward_EITAnalyticSolution ""}

itcl_class BioPSE_Forward_EITAnalyticSolution {
    inherit Module
    constructor {config} {
	set name EITAnalyticSolution
	set_defaults
    }
    method set_defaults {} {
	global $this-outerRadiusTCL
	global $this-innerRadiusTCL
	global $this-bodyGeomTCL
	set $this-outerRadiusTCL 0.150
	set $this-innerRadiusTCL 0.070
	set $this-bodyGeomTCL {"Concentric disks"}
    }
    method make_entry {w text v c} {
        frame $w
        label $w.l -text "$text"
        pack $w.l -side left
        entry $w.e -textvariable $v
        bind $w.e <Return> $c
        pack $w.e -side right
    }
    method ui {} {
	global $this-outerRadiusTCL
	global $this-innerRadiusTCL
	global $this-bodyGeomTCL

	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	
	make_labeled_radio $w.bodyGeom "Body Geometry:" "" top $this-bodyGeomTCL \
	    {{"Homogeneous disk"} {"Concentric disks"}}
	make_entry $w.outerRadius "Outer radius:" $this-outerRadiusTCL "$this-c needexecute"
	make_entry $w.innerRadius "Inner radius:" $this-innerRadiusTCL "$this-c needexecute"
	bind $w.outerRadius <Return> "$this-c needexecute"
	bind $w.innerRadius <Return> "$this-c needexecute"
	pack $w.bodyGeom $w.outerRadius $w.innerRadius -side top -fill x
    }
}
