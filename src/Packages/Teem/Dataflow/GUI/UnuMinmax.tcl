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
#    File   : UnuMinmax.tcl
#    Author : Darby Van Uitert
#    Date   : April 2004

itcl_class Teem_Unu_UnuMinmax {
    inherit Module
    constructor {config} {
        set name UnuMinmax
        set_defaults
    }

    method set_defaults {} {
	global $this-nrrds
	global $this-old_nrrds

	set $this-nrrds 0
	set $this-old_nrrds 0
    }

    method init_axes {mins maxs} {
	set w .ui[modname]

	#if {[winfo exists $w.f.mmf]} {
	    #destroy $w.f.mmf.t
	#}
	# clear old ones using old_dim
	for {set i 0} {$i < [set $this-old_nrrds]} {incr i} {
	    if {[winfo exists $w.f.mmf.a$i]} {
		destroy $w.f.mmf.a$i
	    }
	}

	set $this-old_nrrds [set $this-nrrds]
	# create new ones
	for {set i 0} {$i < [set $this-nrrds]} {incr i} {
	    create_min_max_info $w.f.mmf.a$i $i [lindex $mins $i] [lindex $maxs $i]
	}
    }
    
    method create_min_max_info {w which min max} {
	iwidgets::labeledframe $w -labeltext "Nrrd $which" \
	    -labelpos nw
	pack $w -side top -anchor nw -pady 3

	set n [$w childsite]
	
	frame $n.f 
	pack $n.f -side top -anchor nw
	
	label $n.f.minl -text "Min: $min"
	label $n.f.maxl -text "Max: $max"
	pack $n.f.minl $n.f.maxl -side left  -padx 8
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }
        toplevel $w

        wm minsize $w 150 80
        frame $w.f
	pack $w.f -padx 2 -pady 2 -side top -expand yes
	
	frame $w.f.mmf
	pack $w.f.mmf -padx 2 -pady 2 -side top -expand yes
	
	if {[set $this-old_nrrds] == 0} {
	    label $w.f.mmf.t -text "Need to Execute to know the number of Nrrds."
	    pack $w.f.mmf.t
	} 

        makeSciButtonPanel $w $w $this
	moveToCursor $w

	pack $w.f -expand 1 -fill x
    }
}


