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


catch {rename BioPSE_Forward_ApplyFEMCurrentSource ""}

itcl_class BioPSE_Forward_ApplyFEMCurrentSource {
    inherit Module
    constructor {config} {
	set name ApplyFEMCurrentSource
	set_defaults
    }
    method set_defaults {} {
	global $this-sourceNodeTCL
	global $this-sinkNodeTCL
	global $this-modeTCL
	set $this-sourceNodeTCL 0
	set $this-sinkNodeTCL 1
	set $this-modeTCL dipole
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
	global $this-sourceNodeTCL
	global $this-sinkNodeTCL
	global $this-modeTCL

	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	
	make_labeled_radio $w.mode "Source model:" "" left $this-modeTCL \
		{dipole sources-and-sinks}
	make_entry $w.source "Source electrode:" $this-sourceNodeTCL "$this-c needexecute"
	make_entry $w.sink "Sink electrode:" $this-sinkNodeTCL "$this-c needexecute"
	bind $w.source <Return> "$this-c needexecute"
	bind $w.sink <Return> "$this-c needexecute"
	pack $w.mode $w.source $w.sink -side top -fill x
    }
}
