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
    inherit ModuleGui 

    constructor {config} {
        set name MatrixSelectVector
        set_defaults
    }
    method set_defaults {} {    
        global $this-col
        global $this-col_max
        global $this-row
        global $this-row_max
        global $this-row_or_col;
        global $this-animate;
        set $this-col 0
        set $this-col_max 100
        set $this-row 0
        set $this-row_max 100
        set $this-row_or_col row
        set $this-animate 0
    }
    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }

        toplevel $w
        wm minsize $w 150 30
        set n "$this-c needexecute "

        global $this-col
        global $this-col_max
        global $this-row
        global $this-row_max
        global $this-row_or_col
        global $this-animate

        frame $w.f
        scale $w.f.r -variable $this-row \
                -from 0 -to [set $this-row_max] \
                -label "Row #" \
                -showvalue true -orient horizontal
        scale $w.f.c -variable $this-col \
                -from 0 -to [set $this-col_max] \
                -label "Column #" \
                -showvalue true -orient horizontal
        frame $w.f.ff
        radiobutton $w.f.ff.r -text "Row" -variable $this-row_or_col -value row
        radiobutton $w.f.ff.c -text "Column" -variable $this-row_or_col -value col
        pack $w.f.ff.r $w.f.ff.c -side left -fill x -expand 1
        frame $w.f.b
        button $w.f.b.go -text "Execute" -command $n 
        checkbutton $w.f.b.a -text "Animate" -variable $this-animate
        button $w.f.b.stop -text "Stop" -command "$this-c stop" 
        pack $w.f.b.go $w.f.b.a $w.f.b.stop -side left -fill x -expand 1
        pack $w.f.r $w.f.c $w.f.ff $w.f.b -side top -fill x -expand yes
        pack $w.f
    }

    method update {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            global $this-col_max
            global $this-row_max
            $w.f.r config -to [set $this-row_max]
            $w.f.c config -to [set $this-col_max]
            puts "updating!"
        }
    }
}
