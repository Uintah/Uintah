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

#    File   : TendSlice.tcl
#    Author : Darby Van Uitert
#    Date   : April 2004

itcl_class Teem_Tend_TendSlice {
    inherit Module
    constructor {config} {
        set name TendSlice
        set_defaults
    }

    method set_defaults {} {
	global $this-axis
	set $this-axis 1

	global $this-position
	set $this-position 0

	global $this-dimension
	set $this-dimension 3
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w

        frame $w.f
	pack $w.f -padx 2 -pady 2 -side top -expand yes
	
	frame $w.f.options
	pack $w.f.options -side top -expand yes

	iwidgets::entryfield $w.f.options.axis \
	    -labeltext "Axis:" \
	    -textvariable $this-axis
        pack $w.f.options.axis -side top -expand yes -fill x
	

        iwidgets::entryfield $w.f.options.position \
	    -labeltext "Position:" \
	    -textvariable $this-position
        pack $w.f.options.position -side top -expand yes -fill x

	iwidgets::labeledframe $w.f.options.dimension \
	    -labeltext "Dimension" \
	    -labelpos nw
	pack $w.f.options.dimension -side top -expand yes -fill x
	
	set dimension [$w.f.options.dimension childsite]
	radiobutton $dimension.1 \
	    -text "2 dimensional" \
	    -variable $this-dimension \
	    -value 2

	radiobutton $dimension.2 \
	    -text "3 dimensional" \
	    -variable $this-dimension \
	    -value 3

	pack $dimension.1 $dimension.2 \
	    -side top -anchor nw -padx 3 -pady 3

	makeSciButtonPanel $w.f $w $this
	moveToCursor $w

	pack $w.f -expand 1 -fill x
    }
}