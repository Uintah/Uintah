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


# GUI for FieldFrenet module
# by Allen Sanderson
# April 2005

# This GUI interface is for selecting an axis and index for sub sampling a
# topologically structured field

itcl_class SCIRun_FieldsOther_FieldFrenet {
    inherit Module
    constructor {config} {
        set name FieldFrenet
        set_defaults
    }

    method set_defaults {} {

	global $this-direction
	global $this-dims
	global $this-axis

	set $this-direction 0
	set $this-axis 2
	set $this-dims 3

	trace variable $this-dims w "$this update_setsize_callback"
    }

    method ui {} {

	global $this-direction
	global $this-axis
	global $this-dims

        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }

        toplevel $w

	frame $w.direction

	label $w.direction.label -text "Direction" -width 9 -anchor w -just left

	radiobutton $w.direction.tangent -text "Tangent" -width 8 \
		-anchor w -just left -variable $this-direction -value 0

	radiobutton $w.direction.normal -text "Normal" -width 7 \
		-anchor w -just left -variable $this-direction -value 1

	radiobutton $w.direction.binormal -text "Binormal" -width 9 \
		-anchor w -just left -variable $this-direction -value 2

	pack $w.direction.label $w.direction.tangent $w.direction.normal \
	    $w.direction.binormal  -side left


	frame $w.axis

	label $w.axis.label -text "Axis" -width 5 -anchor w -just left

	for {set i 0} {$i < 3} {incr i 1} {
	    if { $i == 0 } {
		set index i
	    } elseif { $i == 1 } {
		set index j
	    } elseif { $i == 2 } {
		set index k
	    }

	    radiobutton $w.axis.$index -text "$index" -width 2 \
		-anchor w -just left -variable $this-axis -value $i
	}

	if { [set $this-dims] == 3 } {
	    pack $w.axis.label $w.axis.i $w.axis.j $w.axis.k -side left -padx 10
	} elseif { [set $this-dims] == 2 } {
	    pack $w.axis.label $w.axis.i $w.axis.j -side left -padx 10
	} elseif { [set $this-dims] == 1 } {
	    pack $w.axis.label $w.axis.i -side left -padx 10
	}

	pack $w.direction $w.axis -side top
	
	makeSciButtonPanel $w $w $this
	moveToCursor $w
    }

    method update_setsize_callback { name1 name2 op } {
	global $this-dims

	set_size [set $this-dims]
    }

    method set_size { dims } {
	global $this-dims
	global $this-axis

	set $this-dims $dims

	if { [set $this-axis] >= [set $this-dims] } {
	    set $this-axis [expr [set $this-dims]-1]
	}

	set w .ui[modname]

	if {[winfo exists $w]} {

	    pack forget $w.axis.label
	    pack forget $w.axis.i
	    pack forget $w.axis.k
	    pack forget $w.axis.j

	    if { [set $this-dims] == 3 } {
		pack $w.axis.label $w.axis.i $w.axis.j $w.axis.k -side left -padx 10
	    } elseif { [set $this-dims] == 2 } {
		pack $w.axis.label $w.axis.i $w.axis.j -side left -padx 10
	    } elseif { [set $this-dims] == 1 } {
		pack $w.axis.label $w.axis.i -side left -padx 10
	    }
	}
    }
}
