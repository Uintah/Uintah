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
 #  NrrdResample.tcl: The NrrdResample UI
 #  Written by:
 #   David Weinstein
 #   Department of Computer Science
 #   University of Utah
 #   Jan 2000
 #  Copyright (C) 2000 SCI Group
 ##

catch {rename Teem_Filters_NrrdResample ""}

itcl_class Teem_Filters_NrrdResample {
    inherit Module
    constructor {config} {
        set name NrrdResample
        set_defaults
    }
    method set_defaults {} {
        global $this-filtertype
        global $this-resampAxis1
        global $this-resampAxis2
        global $this-resampAxis3
	global $this-sigma
	global $this-extent
        set $this-filtertype gaussian
        set $this-resampAxis1 x1
        set $this-resampAxis2 x1
        set $this-resampAxis3 x1
	set $this-sigma 1
	set $this-extent 6
    }
    method make_entry {w text v c} {
        frame $w
        label $w.l -text "$text"
        pack $w.l -side left
        global $v
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
	make_entry $w.f.s "   Guassian sigma:" $this-sigma "$this-c needexecute"
	global $this-extent
	make_entry $w.f.e "   Guassian extent:" $this-extent "$this-c needexecute"
	frame $w.f.f
	label $w.f.f.l -text "Number of samples (e.g. `128')\nor, if preceded by an x,\nthe resampling ratio\n(e.g. `x0.5' -> half as many samples)"
	global $this-resampAxis1
	make_entry $w.f.f.fi "Axis1:" $this-resampAxis1 "$this-c needexecute"
	global $this-resampAxis2
	make_entry $w.f.f.fj "Axis2:" $this-resampAxis2 "$this-c needexecute"
	global $this-resampAxis3
	make_entry $w.f.f.fk "Axis3:" $this-resampAxis3 "$this-c needexecute"
	pack $w.f.f.l $w.f.f.fi $w.f.f.fj $w.f.f.fk -side top -expand 1 -fill x
	button $w.f.b -text "Execute" -command "$this-c needexecute"
	pack $w.f.t $w.f.s $w.f.e $w.f.f $w.f.b -side top -expand 1 -fill x
	pack $w.f -expand 1 -fill x
    }
}
