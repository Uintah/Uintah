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

#    File   : UnuCrop.tcl
#    Author : Martin Cole
#    Date   : Tue Mar 18 08:46:53 2003

catch {rename Teem_UnuAtoM_UnuCrop ""}

itcl_class Teem_UnuAtoM_UnuCrop {
    inherit Module
    constructor {config} {
        set name UnuCrop
        set_defaults
    }
    method set_defaults {} {
	global $this-num-axes
	global $this-reset
	global $this-uis
	global $this-minAxis0
	global $this-minAxis1
	global $this-minAxis2
	global $this-minAxis3
	global $this-maxAxis0
	global $this-maxAxis1
	global $this-maxAxis2
	global $this-maxAxis3
	global $this-absmaxAxis0
	global $this-absmaxAxis1
	global $this-absmaxAxis2
	global $this-absmaxAxis3
	global $this-reset_data
	global $this-digits_only


	set $this-num-axes 0
	set $this-reset 0
	set $this-uis 4
	set $this-minAxis0 0
	set $this-minAxis1 0
	set $this-minAxis2 0
	set $this-minAxis3 0
	set $this-maxAxis0 M
	set $this-maxAxis1 M
	set $this-maxAxis2 M
	set $this-maxAxis3 M
	set $this-absmaxAxis0 0
	set $this-absmaxAxis1 0
	set $this-absmaxAxis2 0
	set $this-absmaxAxis3 0
	set $this-reset_data 1
	set $this-digits_only 0
    }

    method reset_vals {} {
	set w .ui[modname]
	for {set i 0} {$i < [set $this-num-axes]} {incr i} {
	    set $this-minAxis$i 0
	    set $this-maxAxis$i M
	}
    }


    method make_min_max {i} {
	set w .ui[modname]
        if {[winfo exists $w]} {
  
	    if {[winfo exists $w.f.mmf.t]} {
		destroy $w.f.mmf.t
	    }
	    if {! [winfo exists $w.f.mmf.a$i]} {
		if {![info exists $this-absmaxAxis$i]} {
		    set $this-minAxis$i 0
		    set $this-maxAxis$i M
		    set $this-absmaxAxis$i 0
		}
		iwidgets::labeledframe $w.f.mmf.a$i -labeltext "Axis $i" \
		    -labelpos nw
		pack $w.f.mmf.a$i -side top -anchor nw -padx 3

		set fr [$w.f.mmf.a$i childsite]
		iwidgets::entryfield $fr.min -labeltext "Min:" \
		    -labelpos w -textvariable $this-minAxis$i
		Tooltip $fr.min "Lower corner of the bounding box.\nValue can either be an integer\nor in the form M, M+<int>, or M-<int>\nwhere M is equal to the number of\nsamples minus 1 for Axis $i."

		iwidgets::entryfield $fr.max -labeltext "Max:" \
		    -labelpos w -textvariable $this-maxAxis$i
		Tooltip $fr.max "Higher corner of the bounding box.\nValue can either be an integer\nor in the form M, M+<int>, M-<int>,\n or m+<int> where M is equal to the\nnumber of samples minus 1 for Axis $i\nand m+<int> specifies an index relative\nto the minimum."

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
	    unset $this-minAxis$i
	    unset $this-maxAxis$i
	    unset $this-absmaxAxis$i

	    set $this-reset 1
	}
    }

    method update_sizes {} {
	set w .ui[modname]

	if {[winfo exists $w]} {
	    set sizes ""
	    for {set i 0} {$i < [expr [set $this-uis] - 1]} {incr i} {
		set sizes [concat $sizes [set $this-absmaxAxis$i] ", "]
	    }

	    set last [expr [set $this-uis] - 1]
	    set sizes [concat $sizes [set $this-absmaxAxis$last]]
	    $w.f.sizes configure -text "Samples: $sizes"
	}
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
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

	button $w.f.buttons.add -text "Add Crop Axis" -command "$this-c add_axis"
	button $w.f.buttons.remove -text "Remove Crop Axis" -command "$this-c remove_axis"

	pack $w.f.buttons.add $w.f.buttons.remove -side left -anchor nw -padx 5 -pady 4

	checkbutton $w.f.reset -text "Reset min and max for new data" \
	    -variable $this-reset_data
	pack $w.f.reset -side top -anchor n

        makeSciButtonPanel $w $w $this
	moveToCursor $w

	pack $w.f -expand 1 -fill x
    }
}
