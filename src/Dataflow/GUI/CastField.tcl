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

itcl_class SCIRun_Fields_CastField {
    inherit ModuleGui
    constructor {config} {
	set name CastField
	set_defaults
    }
    method set_defaults {} {
	global $this-haveMinMaxTCL
	global $this-haveOutVoxelTCL
	global $this-haveBBoxTCL
	global $this-NminTCL
	global $this-NmaxTCL
	global $this-CminTCL
	global $this-CmaxTCL
	global $this-minOutTCLX
	global $this-minOutTCLY
	global $this-minOutTCLZ
	global $this-maxOutTCLX
	global $this-maxOutTCLY
	global $this-maxOutTCLZ
	global $this-outVoxelTCL

	set $this-haveMinMaxTCL 0
	set $this-haveOutVoxelTCL 0
	set $this-haveBBoxTCL 0
	set $this-NminTCL 0
	set $this-NmaxTCL 0
	set $this-CminTCL 0
	set $this-CmaxTCL 0
	set $this-minOutTCLX 0
	set $this-minOutTCLY 0
	set $this-minOutTCLZ 0
	set $this-maxOutTCLX 0
	set $this-maxOutTCLY 0
	set $this-maxOutTCLZ 0
	set $this-outVoxelTCL 0
    }
    method ui {} {
	global $this-haveMinMaxTCL
	global $this-haveOutVoxelTCL
	global $this-haveBBoxTCL
	global $this-NminTCL
	global $this-NmaxTCL
	global $this-CminTCL
	global $this-CmaxTCL
	global $this-minOutTCLX
	global $this-minOutTCLY
	global $this-minOutTCLZ
	global $this-maxOutTCLX
	global $this-maxOutTCLY
	global $this-maxOutTCLZ
	global $this-outVoxelTCL

	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	frame $w.v -relief sunken -bd 2
	checkbutton $w.v.b -text "Change datatype?" \
		-variable $this-haveOutVoxelTCL
        make_labeled_radio $w.v.type "New datatype: " "" left \
		$this-outVoxelTCL {{double 0} {float 1} {int 2} \
		       { short 3 } {ushort 4} {char 5} {uchar 6}}
	pack $w.v.b $w.v.type -side top -fill x
	pack $w.v -side top -fill x -expand 1

	frame $w.b -relief sunken -bd 2
	checkbutton $w.b.b -text "Change bbox?" \
		-variable $this-haveBBoxTCL
	pack $w.b.b -side top -fill x
	frame $w.b.x
	label $w.b.x.text -text "Min/Max X:"
	entry $w.b.x.min -width 7 -relief sunken -bd 2 \
		-textvariable $this-minOutTCLX
	entry $w.b.x.max -width 7 -relief sunken -bd 2 \
		-textvariable $this-maxOutTCLX
	pack $w.b.x.text $w.b.x.min $w.b.x.max -side left \
		-padx 4 -fill x -expand 1
	pack $w.b.x -side top -fill x -expand 1
	frame $w.b.y
	label $w.b.y.text -text "Min/Max Y:"
	entry $w.b.y.min -width 7 -relief sunken -bd 2 \
		-textvariable $this-minOutTCLY
	entry $w.b.y.max -width 7 -relief sunken -bd 2 \
		-textvariable $this-maxOutTCLY
	pack $w.b.y.text $w.b.y.min $w.b.y.max -side left \
		-padx 4 -fill x -expand 1
	pack $w.b.y -side top -fill x -expand 1
	frame $w.b.z
	label $w.b.z.text -text "Min/Max Z:"
	entry $w.b.z.min -width 7 -relief sunken -bd 2 \
		-textvariable $this-minOutTCLZ
	entry $w.b.z.max -width 7 -relief sunken -bd 2 \
		-textvariable $this-maxOutTCLZ
	pack $w.b.z.text $w.b.z.min $w.b.z.max -side left \
		-padx 4 -fill x -expand 1
	pack $w.b.z -side top -fill x -expand 1
	pack $w.b -side top -fill x -expand 1

	frame $w.m -relief sunken -bd 2
	checkbutton $w.m.m -text "Change data values?" \
		-variable $this-haveMinMaxTCL
	pack $w.m.m -side top -fill x
	frame $w.m.n
	label $w.m.n.text -text "New min/max span:"
	entry $w.m.n.min -width 7 -relief sunken -bd 2 \
		-textvariable $this-NminTCL
	entry $w.m.n.max -width 7 -relief sunken -bd 2 \
		-textvariable $this-NmaxTCL
	pack $w.m.n.text $w.m.n.min $w.m.n.max -side left \
		-padx 4 -fill x -expand 1
	pack $w.m.n -side top -fill x -expand 1
	frame $w.m.c
	label $w.m.c.text -text "Crop min/max span:"
	entry $w.m.c.min -width 7 -relief sunken -bd 2 \
		-textvariable $this-CminTCL
	entry $w.m.c.max -width 7 -relief sunken -bd 2 \
		-textvariable $this-CmaxTCL
	pack $w.m.c.text $w.m.c.min $w.m.c.max -side left \
		-padx 4 -fill x -expand 1
	pack $w.m.c -side top -fill x -expand 1
	pack $w.m -side top -fill x -expand 1
	button $w.ex -text "Execute" -command "$this-c needexecute"
	pack $w.ex -side top -fill both -expand 1
    }
}
