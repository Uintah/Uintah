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

catch {rename BioPSE_Inverse_ConductivitySearch ""}

itcl_class BioPSE_Inverse_ConductivitySearch {
    inherit Module

    constructor {config} {
	set name ConductivitySearch
	set_defaults
    }

    method set_defaults {} {	
	global $this-refnodeTCL
	global $this-pinzeroTCL
	global $this-seedTCL
	set $this-refnodeTCL 0
	set $this-pinzeroTCL 1
	set $this-seedTCL 0
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
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	wm minsize $w 100 50
	frame $w.g
        button $w.g.go -text "Execute" -relief raised -command "$this-c exec"
        button $w.g.p -text "Pause" -relief raised -command "$this-c pause"
        button $w.g.np -text "Unpause" -relief raised -command "$this-c unpause"
	button $w.g.print -text "Print" -relief raised -command "$this-c print"
	button $w.g.stop -text "Stop" -relief raised -command "$this-c stop"
	pack $w.g.go $w.g.p $w.g.np $w.g.print $w.g.stop -side left -fill x -expand 1 -padx 3
	frame $w.ref
	global $this-pinzeroTCL
	checkbutton $w.ref.pinzero -text "Pin A Reference Node" -variable $this-pinzeroTCL
	global $this-refnodeTCL
	make_entry $w.ref.refnode "Reference node:" $this-refnodeTCL "$this-c needexecute"
	pack $w.ref.pinzero $w.ref.refnode -side top -fill x -expand 1
	frame $w.seed
	global $this-seedTCL
	make_entry $w.seed.seed "Random number seed:" $this-seedTCL "$this-c needexecute"
	pack $w.seed.seed -side top -fill x -expand 1
	pack $w.g $w.ref $w.seed -side top -fill x -expand 1
    }
}

