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

itcl_class SCIRun_Math_MatrixSelectVector {
    inherit Module 

    constructor {config} {
        set name MatrixSelectVector

        global $this-row_or_col
        global $this-selectable_min
        global $this-selectable_max
        global $this-selectable_inc
	global $this-selectable_units
        global $this-range_min
        global $this-range_max

        set_defaults
    }
    method set_defaults {} {    
        set $this-row_or_col         row
        set $this-selectable_min     0
        set $this-selectable_max     100
        set $this-selectable_inc     1
	set $this-selectable_units   Units
        set $this-range_min          0
        set $this-range_max          0
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }
        toplevel $w

        frame $w.f
        scale $w.f.r -variable $this-range_min \
		-label "Start" \
                -showvalue true -orient horizontal
        scale $w.f.c -variable $this-range_max \
		-label "End" \
                -showvalue true -orient horizontal
        frame $w.f.ff
        radiobutton $w.f.ff.r -text "Row" -variable $this-row_or_col \
		-value row -command "$this-c needexecute"
        radiobutton $w.f.ff.c -text "Column" -variable $this-row_or_col \
		-value col -command "$this-c needexecute"
        pack $w.f.ff.r $w.f.ff.c -side left -fill x -expand 1
        frame $w.f.b
        button $w.f.b.go -text "Execute" -command "$this-c needexecute"
        pack $w.f.b.go -side left -fill x -expand 1
        pack $w.f.r $w.f.c $w.f.ff $w.f.b -side top -fill x -expand yes
        pack $w.f

	update
    }

    method update {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            #puts "updating!"

            $w.f.r config -from [set $this-selectable_min]
            $w.f.r config -to   [set $this-selectable_max]
            $w.f.r config -label [concat "Start " [set $this-selectable_units]]
            $w.f.c config -from [set $this-selectable_min]
            $w.f.c config -to   [set $this-selectable_max]
            $w.f.c config -label [concat "End " [set $this-selectable_units]]

	    pack $w.f.r $w.f.c $w.f.ff $w.f.b -side top -fill x -expand yes
        }
    }
}
