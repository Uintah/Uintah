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

catch {rename CardioWave_CreateModel_CreateSimpleMesh ""}

itcl_class CardioWave_CreateModel_CreateSimpleMesh {
    inherit Module

    constructor {config} {
	set name CreateSimpleMesh
	set_defaults
    }

    method set_defaults {} {	
        global $this-xdim
	global $this-ydim
	global $this-zdim
        global $this-dx
	global $this-dy
	global $this-dz
        global $this-fib1x
	global $this-fib1y
	global $this-fib1z
        global $this-fib2x
	global $this-fib2y
	global $this-fib2z
        set $this-xdim 1
	set $this-ydim 1
	set $this-zdim 1
        set $this-dx 0.01
	set $this-dy 0.01
	set $this-dz 0.01
        set $this-fib1x 1
	set $this-fib1y 0
	set $this-fib1z 0
        set $this-fib2x 0
	set $this-fib2y 1
	set $this-fib2z 0



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

  	frame $w.dims
        global $this-xdim
	global $this-ydim
	global $this-zdim
	make_entry $w.dims.x "Xdim:" $this-xdim \
		"$this-c needexecute"
	make_entry $w.dims.y "Ydim:" $this-ydim \
		"$this-c needexecute"
	make_entry $w.dims.z "Zdim:" $this-zdim \
		"$this-c needexecute"
	pack $w.dims.x $w.dims.y $w.dims.z -side left -fill x -expand 1

  	frame $w.step
        global $this-dx
	global $this-dy
	global $this-dz
	make_entry $w.step.x "dx (cm):" $this-dx \
		"$this-c needexecute"
	make_entry $w.step.y "dy (cm):" $this-dy \
		"$this-c needexecute"
	make_entry $w.step.z "dz (cm):" $this-dz \
		"$this-c needexecute"
	pack $w.step.x $w.step.y $w.step.z -side left -fill x -expand 1


  	frame $w.fiber1
        global $this-fib1x
	global $this-fib1y
	global $this-fib1z
	make_entry $w.fiber1.x "Fiber1 X:" $this-fib1x \
		"$this-c needexecute"
	make_entry $w.fiber1.y "Fiber1 Y:" $this-fib1y \
		"$this-c needexecute"
	make_entry $w.fiber1.z "Fiber1 Z:" $this-fib1z \
		"$this-c needexecute"
	pack $w.fiber1.x $w.fiber1.y $w.fiber1.z -side left -fill x -expand 1

  	frame $w.fiber2
        global $this-fib2x
	global $this-fib2y
	global $this-fib2z
	make_entry $w.fiber2.x "Fiber2 X:" $this-fib2x \
		"$this-c needexecute"
	make_entry $w.fiber2.y "Fiber2 Y:" $this-fib2y \
		"$this-c needexecute"
	make_entry $w.fiber2.z "Fiber2 Z:" $this-fib2z \
		"$this-c needexecute"
	pack $w.fiber2.x $w.fiber2.y $w.fiber2.z -side left -fill x -expand 1

	pack $w.f $w.dims $w.step $w.fiber1 $w.fiber2 -side top -fill x -expand 1





    }
}	
