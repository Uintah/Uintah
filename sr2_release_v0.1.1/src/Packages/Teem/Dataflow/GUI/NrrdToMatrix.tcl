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
 #  NrrdToMatrix.tcl
 #  Written by:
 #   Darby Van Uitert
 #   April 2004
 ##

itcl_class Teem_Converters_NrrdToMatrix {
    inherit Module
    constructor {config} {
        set name NrrdToMatrix
        set_defaults
    }

    method set_defaults {} {
	global $this-cols
	global $this-entry
	global $this-which

	set $this-cols {-1}
	set $this-entry 3
	set $this-which 0

	trace variable $this-entry w "$this entry_changed"
    }

    method entry_changed {name1 name2 op} {
	update_which
    }

    method update_which {} {
	if {[set $this-which] == 1} {
	    # user specifies columns
	    set $this-cols [set $this-entry]
	} else {
	    # auto
	    set $this-cols {-1}
	}
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }

        toplevel $w

	frame $w.cols
	pack $w.cols -side top -anchor nw -padx 3 -pady 3
	radiobutton $w.cols.auto -text "Auto" \
	    -variable $this-which \
	    -value 0 \
	    -command "$this update_which"
	radiobutton $w.cols.spec -text "Columns (for Sparse Row Matrix): " \
	    -variable $this-which \
	    -value 1 \
	    -command "$this update_which"
	entry $w.cols.e -textvariable $this-entry 
	pack $w.cols.auto $w.cols.spec $w.cols.e -side left -anchor nw -padx 3 -pady 3

	makeSciButtonPanel $w $w $this
	moveToCursor $w
    }
}


