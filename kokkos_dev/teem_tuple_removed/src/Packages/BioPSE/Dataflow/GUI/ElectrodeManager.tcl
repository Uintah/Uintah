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


catch {rename BioPSE_Forward_ElectrodeManager ""}

itcl_class BioPSE_Forward_ElectrodeManager {
    inherit Module
    constructor {config} {
        set name ElectrodeManager
        set_defaults
    }

    method set_defaults {} {
        global $this-modelTCL
	global $this-numElTCL
	global $this-lengthElTCL
	global $this-startNodeIdxTCL
        set $this-modelTCL 1
	set $this-numElTCL 32
	set $this-lengthElTCL 0.027
	set $this-startNodeIdxTCL 0
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
        global $this-modelTCL
	global $this-numElTCL
	global $this-lengthElTCL
	global $this-startNodeIdxTCL

        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }
        toplevel $w

	make_entry $w.model "0=Continuum, 1=Gap: " $this-modelTCL "$this-c needexecute"
	make_entry $w.numEl "Number of electrodes: " $this-numElTCL "$this-c needexecute"
	make_entry $w.lengthEl "Electrode length: " $this-lengthElTCL "$this-c needexecute"
	make_entry $w.startNodeIdx "Boundary node index for start of first electrode: " $this-startNodeIdxTCL "$this-c needexecute"
	bind $w.model <Return> "$this-c needexecute"
	bind $w.numEl <Return> "$this-c needexecute"
	bind $w.lengthEl <Return> "$this-c needexecute"
	bind $w.startNodeIdx <Return> "$this-c needexecute"
	pack $w.model $w.numEl $w.lengthEl $w.startNodeIdx -side top -fill x

    }
}


