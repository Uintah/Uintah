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

catch {rename CardioWave_CreateModel_SetupFVMatrix ""}

itcl_class CardioWave_CreateModel_SetupFVMatrix {
    inherit Module

    constructor {config} {
	set name SetupFVMatrix
	set_defaults
    }

    method set_defaults {} {	
        global $this-sigx
	global $this-sigy
	global $this-sigz
	global $this-sprfile
	global $this-volumefile
        set $this-sigx 0
	set $this-sigy 0
	set $this-sigz 0
	set $this-sprfile "SPR"
	set $this-volumefile "VOLUME"
	
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
        wm minsize $w 150 20
        frame $w.f
        pack $w.f -padx 2 -pady 2 -side top -expand yes

  	frame $w.sigma
        global $this-sigx
	global $this-sigy
	global $this-sigz
	make_entry $w.sigma.x "Sigma X (mS/cm):" $this-sigx \
		"$this-c needexecute"
	make_entry $w.sigma.y "Sigma Y (mS/cm):" $this-sigy \
		"$this-c needexecute"
	make_entry $w.sigma.z "Sigma Z (mS/cm):" $this-sigz \
		"$this-c needexecute"
	pack $w.sigma.x $w.sigma.y $w.sigma.z -side left -fill x -expand 1


        global $this-sprfile
	make_entry $w.f.s "SPR file: " $this-sprfile \
		"$this-c needexecute"

        global $this-volumefile
	make_entry $w.f.v "Volume file: " $this-volumefile \
		"$this-c needexecute"		
	

	pack $w.f $w.sigma $w.f.s $w.f.v -side top -fill x -expand 1
   }
}