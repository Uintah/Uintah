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

	set $this-num-dims 0
	set $this-reset 0
    }

    method set_max_vals {} {
	set w .ui[modname]

	if {[winfo exists $w]} {
	    for {set i 0} {$i < [set $this-num-dims]} {incr i} {

		set ax $this-absmaxDim$i
		set_scale_max_value $w.f.mmf.a$i [set $ax]
		if {[set $this-maxDim$i] == -1 || [set $this-reset]} {
		    set $this-maxDim$i [set $this-absmaxDim$i]
		}
	    }
	    set $this-reset 0
	}
    }

    method init_dims {} {
	for {set i 0} {$i < [set $this-num-dims]} {incr i} {

	    if { [catch { set t [set $this-minDim$i] } ] } {
		set $this-minDim$i 0
	    }
	    if { [catch { set t [set $this-maxDim$i]}] } {
		set $this-maxDim$i -1
	    }
	    if { [catch { set t [set $this-absmaxDim$i]}] } {
		set $this-absmaxDim$i 0
	    }
	}
	make_min_max
    }

    method make_min_max {} {
	set w .ui[modname]
        if {[winfo exists $w]} {
  
	    if {[winfo exists $w.f.mmf.t]} {
		destroy $w.f.mmf.t
	    }
	    for {set i 0} {$i < [set $this-num-dims]} {incr i} {
		if {! [winfo exists $w.f.mmf.a$i]} {
		    min_max_widget $w.f.mmf.a$i "Dim $i" \
			$this-minDim$i $this-maxDim$i $this-absmaxDim$i 
		}
	    }
	}
    }

   method clear_dims {} {
	set w .ui[modname]
        if {[winfo exists $w]} {
  
	    if {[winfo exists $w.f.mmf.t]} {
		destroy $w.f.mmf.t
	    }
	    for {set i 0} {$i < [set $this-num-dims]} {incr i} {
		if {[winfo exists $w.f.mmf.a$i]} {
		    destroy $w.f.mmf.a$i
		}
		unset $this-minDim$i
		unset $this-maxDim$i
		unset $this-absmaxDim$i
	    }
	    set $this-reset 1
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
	
	frame $w.f.mmf
	pack $w.f.mmf -padx 2 -pady 2 -side top -expand yes
	
	
	if {[set $this-num-dims] == 0} {
	    label $w.f.mmf.t -text "Need to Execute to know the number of Dimensions."
	    pack $w.f.mmf.t
	} else {
	    init_dims 
	}

	button $w.f.b -text "Execute" -command "$this-c needexecute"
	pack $w.f.b -side top -expand 1 -fill x
	pack $w.f -expand 1 -fill x
    }
}


