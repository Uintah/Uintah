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

itcl_class SCIRun_FieldsCreate_ClipLattice {
    inherit Module
    constructor {config} {
        set name ClipLattice
        set_defaults
    }
    method set_defaults {} {
	global $this-use-text-bbox
	global $this-text-min-x
	global $this-text-min-y
	global $this-text-min-z
	global $this-text-max-x
	global $this-text-max-y
	global $this-text-max-z
	set $this-use-text-bbox 0
	set $this-text-min-x 0
	set $this-text-min-y 0
	set $this-text-min-z 0
	set $this-text-max-x 1
	set $this-text-max-y 1
	set $this-text-max-z 1
    }
    method labelentry { win text var } {
	frame $win 
	label $win.l -text $text -anchor w
	entry $win.e -width 10 -just left -textvariable $var
	pack $win.l $win.e -padx 5 -side left
	pack $win -side top -padx 5
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
	global $this-text-min-x
	global $this-text-min-y
	global $this-text-min-z
	global $this-text-max-x
	global $this-text-max-y
	global $this-text-max-z
	global $this-use-text-bbox
	frame $w.f.min -relief sunken -borderwidth 2
	label $w.f.min.l -text "Min Clip Point"
	pack $w.f.min.l -side top
	labelentry $w.f.min.x "X:" $this-text-min-x
	labelentry $w.f.min.y "Y:" $this-text-min-y
	labelentry $w.f.min.z "Z:" $this-text-min-z
	pack $w.f.min.x $w.f.min.y $w.f.min.z -side top -fill x -pady 3 -expand 1
	frame $w.f.max -relief sunken -borderwidth 2
	label $w.f.max.l -text "Max Clip Point"
	pack $w.f.max.l -side top
	labelentry $w.f.max.x "X:" $this-text-max-x
	labelentry $w.f.max.y "Y:" $this-text-max-y
	labelentry $w.f.max.z "Z:" $this-text-max-z
	pack $w.f.max.x $w.f.max.y $w.f.max.z -side top -fill x -pady 3 -expand 1
	checkbutton $w.f.c -text "Use This BBox" -variable $this-use-text-bbox
	pack $w.f.min $w.f.max $w.f.c -side top -fill x -padx 3 -pady 3 -expand 1
    }
}
