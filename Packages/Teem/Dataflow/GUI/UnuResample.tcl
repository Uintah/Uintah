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

##
 #  UnuResample.tcl: The UnuResample UI
 #  Written by:
 #   David Weinstein
 #   Department of Computer Science
 #   University of Utah
 #   Jan 2000
 #  Copyright (C) 2000 SCI Group
 ##

catch {rename Teem_Unu_UnuResample ""}

itcl_class Teem_Unu_UnuResample {
    inherit Module
    constructor {config} {
        set name UnuResample
        set_defaults
    }
    method set_defaults {} {
        global $this-filtertype
	global $this-sigma
	global $this-extent
	global $this-dim

        set $this-filtertype gaussian
	set $this-sigma 1
	set $this-extent 6
	set $this-dim 0
    }

    # never resample the tuple axis (axis 0) so no interface for axis 0
    method make_min_max {} {
	set w .ui[modname]
        if {[winfo exists $w]} {
 	    if {[winfo exists $w.f.f.axesf.t]} {
		destroy $w.f.f.axesf.t
	    }
	    for {set i 1} {$i < [set $this-dim]} {incr i} {
		#puts $i
		if {! [winfo exists $w.f.f.axesf.a$i]} {
		    make_entry $w.f.f.axesf.a$i "Axis$i :" $this-resampAxis$i
		    pack $w.f.f.axesf.a$i -side top -expand 1 -fill x
		}
	    }
	}
    }
    
    # never resample the tuple axis (axis 0)
    method init_axes {} {
	for {set i 0} {$i < [set $this-dim]} {incr i} {
	    #puts "init_axes----$i"

	    if { [catch { set t [set $this-resampAxis$i] } ] } {
		set $this-resampAxis$i "x1"
		#puts "made minAxis$i"
	    }
	}
	make_min_max
    }
 
    method clear_axes {} {
	set w .ui[modname]
        if {[winfo exists $w]} {
	    
	    if {[winfo exists $w.f.f.axesf.t]} {
		destroy $w.f.f.axesf.t
	    }
	    for {set i 1} {$i < [set $this-dim]} {incr i} {
		#puts $i
		if {[winfo exists $w.f.f.axesf.a$i]} {
		    destroy $w.f.f.axesf.a$i
		}
		unset $this-resampAxis$i
	    }
	}
    }

    method make_entry {w text v} {
        frame $w
        label $w.l -text "$text"
        pack $w.l -side left
        global $v
        entry $w.e -textvariable $v
        pack $w.e -side right
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }

        toplevel $w
        wm minsize $w 200 80
        frame $w.f
        pack $w.f -padx 2 -pady 2 -side top -expand yes
	global $this-filtertype
	make_labeled_radio $w.f.t "Filter Type:" "" \
		top $this-filtertype \
		{{"Box" box} \
		{"Tent" tent} \
		{"Cubic (Catmull-Rom)" cubicCR} \
		{"Cubic (B-spline)" cubicBS} \
		{"Quartic" quartic} \
		{"Gaussian" gaussian}}
	global $this-sigma
	make_entry $w.f.s "   Guassian sigma:" $this-sigma 
	global $this-extent
	make_entry $w.f.e "   Guassian extent:" $this-extent 
	frame $w.f.f
	label $w.f.f.l -text "Number of samples (e.g. `128')\nor, if preceded by an x,\nthe resampling ratio\n(e.g. `x0.5' -> half as many samples)"

	frame $w.f.f.axesf

	if {[set $this-dim] == 0} {
	    label $w.f.f.axesf.t -text "Need Execute to know the number of Axes."
	    pack $w.f.f.axesf.t
	} else {
	    init_axes 
	}

	pack $w.f.f.l $w.f.f.axesf -side top -expand 1 -fill x
	button $w.f.b -text "Execute" -command "$this-c needexecute"
	pack $w.f.t $w.f.s $w.f.e $w.f.f $w.f.b -side top -expand 1 -fill x
	pack $w.f -expand 1 -fill x
    }
}
