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
        global $this-sigx1
	global $this-sigy1
	global $this-sigz1
        global $this-sigx2
	global $this-sigy2
	global $this-sigz2
	global $this-sprfile
	global $this-volumefile
	global $this-visfile
        set $this-sigx1 1
	set $this-sigy1 1
	set $this-sigz1 1
        set $this-sigx2 20
	set $this-sigy2 20
	set $this-sigz2 20
	set $this-sprfile "SPR"
	set $this-volumefile "VOLUME"
	set $this-visfile "VIS"
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

  	frame $w.sigma1
        global $this-sigx1
	global $this-sigy1
	global $this-sigz1
	make_entry $w.sigma1.x "Cell1-Sigma X (mS/cm):" $this-sigx1 \
		"$this-c needexecute"
	make_entry $w.sigma1.y "Cell1-Sigma Y (mS/cm):" $this-sigy1 \
		"$this-c needexecute"
	make_entry $w.sigma1.z "Cell1-Sigma Z (mS/cm):" $this-sigz1 \
		"$this-c needexecute"
	pack $w.sigma1.x $w.sigma1.y $w.sigma1.z -side left -fill x -expand 1

  	frame $w.sigma2
        global $this-sigx2
	global $this-sigy2
	global $this-sigz2
	make_entry $w.sigma2.x "Cell2-Sigma X (mS/cm):" $this-sigx2 \
		"$this-c needexecute"
	make_entry $w.sigma2.y "Cell2-Sigma Y (mS/cm):" $this-sigy2 \
		"$this-c needexecute"
	make_entry $w.sigma2.z "Cell2-Sigma Z (mS/cm):" $this-sigz2 \
		"$this-c needexecute"
	pack $w.sigma2.x $w.sigma2.y $w.sigma2.z -side left -fill x -expand 1


        global $this-sprfile
	make_entry $w.f.spr "SPR file: " $this-sprfile \
		"$this-c needexecute"

        global $this-volumefile
	make_entry $w.f.vol "Volume file: " $this-volumefile \
		"$this-c needexecute"		
	
        global $this-visfile
	make_entry $w.f.vis "Vis file: " $this-visfile \
		"$this-c needexecute"

	pack $w.f $w.sigma1 $w.sigma2 $w.f.spr $w.f.vol $w.f.vis -side top
   }
}