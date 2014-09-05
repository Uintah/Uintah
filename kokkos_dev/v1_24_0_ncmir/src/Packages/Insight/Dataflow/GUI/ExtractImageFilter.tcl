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

#    File   : ExtractImageFilter.tcl
#    Author : Darby Van Uitert
#    Date   : January 2004

itcl_class Insight_Filters_ExtractImageFilter {
    inherit Module
    constructor {config} {
        set name ExtractImageFilter
        set_defaults
    }

    method set_defaults {} {
	global $this-num-dims
	global $this-reset
	global $this-minDim0
	global $this-minDim1
	global $this-minDim2
	global $this-minDim3
	global $this-maxDim0
	global $this-maxDim1
	global $this-maxDim2
	global $this-maxDim3
	global $this-absmaxDim0
	global $this-absmaxDim1
	global $this-absmaxDim2
	global $this-absmaxDim3
	global $this-uis

	set $this-num-dims 0
	set $this-reset 0
	set $this-minDim0 0
	set $this-minDim1 0
	set $this-minDim2 0
	set $this-minDim3 0
	set $this-maxDim0 1024
	set $this-maxDim1 1024
	set $this-maxDim2 1024
	set $this-maxDim3 1024
	set $this-absmaxDim0 0
	set $this-absmaxDim1 0
	set $this-absmaxDim2 0
	set $this-absmaxDim3 0
	set $this-uis 4
    }

    method reset_vals {} {
	set w .ui[modname]
	for {set i 0} {$i < [set $this-num-dims]} {incr i} {
	    set $this-minDim$i 0
	    set $this-maxDim$i 1024
	}
    }

    method make_min_max {i} {
	set w .ui[modname]
        if {[winfo exists $w]} {
  
	    if {[winfo exists $w.f.mmf.t]} {
		destroy $w.f.mmf.t
	    }
	    if {! [winfo exists $w.f.mmf.a$i]} {
		if {![info exists $this-absmaxDim$i]} {
		    set $this-minDim$i 0
		    set $this-maxDim$i 1024
		    set $this-absmaxDim$i 0
		}
		iwidgets::labeledframe $w.f.mmf.a$i -labeltext "Axis $i" \
		    -labelpos nw
		pack $w.f.mmf.a$i -side top -anchor nw -padx 3

		set fr [$w.f.mmf.a$i childsite]
		iwidgets::entryfield $fr.min -labeltext "Min:" \
		    -labelpos w -textvariable $this-minDim$i
		Tooltip $fr.min "Lower corner of the bounding box."

		iwidgets::entryfield $fr.max -labeltext "Max:" \
		    -labelpos w -textvariable $this-maxDim$i
		Tooltip $fr.max "Higher corner of the bounding box."

		pack $fr.min $fr.max -side left
	    }
	}
    }

   method clear_axis {i} {
	set w .ui[modname]
        if {[winfo exists $w]} {
	    if {[winfo exists $w.f.mmf.t]} {
		destroy $w.f.mmf.t
	    }
	    if {[winfo exists $w.f.mmf.a$i]} {
		destroy $w.f.mmf.a$i
	    }
	    unset $this-minDim$i
	    unset $this-maxDim$i
	    unset $this-absmaxDim$i

	    set $this-reset 1
	}
    }

    method update_sizes {} {
	set w .ui[modname]

	if {[winfo exists $w]} {
	    set sizes ""
	    for {set i 0} {$i < [expr [set $this-uis] - 1]} {incr i} {
		set sizes [concat $sizes [expr [set $this-absmaxDim$i] + 1] ", "]
	    }

	    set last [expr [set $this-uis] - 1]
	    set sizes [concat $sizes [expr [set $this-absmaxDim$last] + 1]]
	    $w.f.sizes configure -text "Samples: $sizes"
	}
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }

        toplevel $w
        wm minsize $w 150 80
        frame $w.f
	pack $w.f -padx 2 -pady 2 -side top -expand yes

	# Indicate number of samples
	label $w.f.sizes -text "Samples: "
	pack $w.f.sizes -side top -anchor nw
	$this update_sizes
	
	frame $w.f.mmf
	pack $w.f.mmf -padx 2 -pady 2 -side top -expand yes

	# Add crop axes
	for {set i 0} {$i < [set $this-uis]} {incr i} {
	    make_min_max $i
	}

	# add buttons to increment/decrement resample axes
	frame $w.f.buttons
	pack $w.f.buttons -side top -anchor n

	button $w.f.buttons.add -text "Add Axis" -command "$this-c add_axis"
	button $w.f.buttons.remove -text "Remove Axis" -command "$this-c remove_axis"

	pack $w.f.buttons.add $w.f.buttons.remove -side left -anchor nw -padx 5 -pady 4

        makeSciButtonPanel $w $w $this
	moveToCursor $w

	pack $w.f -expand 1 -fill x
    }
}


