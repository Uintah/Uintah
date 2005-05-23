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

catch {rename CardioWave_CreateModel_SetupFVM2 ""}

itcl_class CardioWave_CreateModel_SetupFVM2 {
    inherit Module

    constructor {config} {
	set name SetupFVM2
	set_defaults
    }

    method set_defaults {} {	
        global $this-bathsig
	global $this-fibersig1
	global $this-fibersig2
	global $this-sprfile
	global $this-volumefile
	global $this-visfile
	global $this-idfile
        set $this-bathsig 20
	set $this-fibersig1 6
	set $this-fibersig2 1
	set $this-sprfile "SPR"
	set $this-volumefile "VOLUME"
	set $this-visfile "VIS"
	set $this-idfile "ID"
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

  	frame $w.bathsig
        global $this-bathsig
	global $this-fibersig1
	global $this-fibersig2
	global $this-sprfile
	global $this-volumefile
	global $this-visfile
	make_entry $w.bathsig.u "Bath Sigma (mS/cm):" $this-bathsig \
		"$this-c needexecute"
	pack $w.bathsig.u -side top -fill x -expand 1

  	frame $w.fibersig
	make_entry $w.fibersig.p "Fiber Primary Sigma (mS/cm):" \
		$this-fibersig1 "$this-c needexecute"
	make_entry $w.fibersig.s "Fiber Secondary Sigma (mS/cm):" \
		$this-fibersig2 "$this-c needexecute"
	pack $w.fibersig.p $w.fibersig.s -side top -fill x -expand 1

	frame $w.f

        global $this-sprfile
	make_entry $w.f.spr "SPR file: " $this-sprfile \
		"$this-c needexecute"

        global $this-volumefile
	make_entry $w.f.vol "Volume file: " $this-volumefile \
		"$this-c needexecute"		
	
        global $this-visfile
	make_entry $w.f.vis "Vis file: " $this-visfile \
		"$this-c needexecute"

        global $this-idfile
	make_entry $w.f.id "ID file: " $this-idfile \
		"$this-c needexecute"

	pack $w.f.spr $w.f.vol $w.f.vis $w.f.id -side top

	pack $w.bathsig $w.fibersig $w.f -side top
   }
}