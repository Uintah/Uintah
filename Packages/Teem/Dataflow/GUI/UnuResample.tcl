#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  License for the specific language governing rights and limitations under
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
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

catch {rename Teem_UnuNtoZ_UnuResample ""}

itcl_class Teem_UnuNtoZ_UnuResample {
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

    method make_min_max {} {
	set w .ui[modname]
        if {[winfo exists $w]} {
 	    if {[winfo exists $w.f.f.axesf.t]} {
		destroy $w.f.f.axesf.t
	    }
	    for {set i 0} {$i < [set $this-dim]} {incr i} {
		#puts $i
		if {! [winfo exists $w.f.f.axesf.a$i]} {
		    make_entry $w.f.f.axesf.a$i "Axis$i :" $this-resampAxis$i
		    pack $w.f.f.axesf.a$i -side top -expand 1 -fill x
		}
	    }
	}
    }
    
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
	    for {set i 0} {$i < [set $this-dim]} {incr i} {
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
            return
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
	    label $w.f.f.axesf.t -text "Need to Execute to know the number of Axes."
	    pack $w.f.f.axesf.t
	} else {
	    init_axes 
	}

	pack $w.f.f.l $w.f.f.axesf -side top -expand 1 -fill x

	pack $w.f.t $w.f.s $w.f.e $w.f.f -side top -expand 1 -fill x

	pack $w.f -expand 1 -fill x

	makeSciButtonPanel $w $w $this
	moveToCursor $w
    }
}
