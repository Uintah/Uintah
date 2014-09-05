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
#    File   : UnuJoin.tcl
#    Author : Martin Cole
#    Date   : Thu Jan 16 09:44:07 2003

itcl_class Teem_Unu_UnuJoin {
    inherit Module
    constructor {config} {
        set name UnuJoin
        set_defaults
    }

    method set_defaults {} {
	global $this-dim
	global $this-join-axis
	global $this-incr-dim

	set $this-dim 0
	set $this-join-axis 0
	set $this-incr-dim 0
    }
   
    # do not allow spaces in the label
    method valid_string {ind str} {
	set char "a"
	
	set char [string index $str $ind]
	if {$ind >= 0 && [string equal $char " "]} {
	    return 0
	}
	return 1
    }


    method axis_radio {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
	    if {[winfo exists $w.f.rfr.radio]} { destroy $w.f.rfr.radio }

	    set choices [list]

	    for {set i 0} {$i < [set $this-dim]} {incr i} {
		if {$i == 0} {
		    lappend choices [list "Tuple Axis" $i]
		} else {
		    set lab "Axis $i"
		    lappend choices [list $lab $i]
		}
	    }

	    make_labeled_radio $w.f.rfr.radio \
		"Join Axis"  "" top $this-join-axis $choices		
	    pack $w.f.rfr.radio -fill both -expand 1 -side top
        }	
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }
        toplevel $w

	frame $w.f -borderwidth 2
	pack $w.f -side top -e y -f both -padx 5 -pady 5

	#frame to pack and repack radio button in
	frame $w.f.rfr -relief groove -borderwidth 2

	axis_radio

	checkbutton $w.f.incrdim \
		-text "Increment Dimension" \
		-variable $this-incr-dim
	
	pack $w.f.rfr $w.f.incrdim -fill both -expand 1 -side top

	button $w.execute -text "Ok" -command "destroy $w"
	pack $w.execute -side top -e n -f both
    }
}


