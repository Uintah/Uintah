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

#    File   : UnuProject.tcl
#    Author : Martin Cole
#    Date   : Mon Sep  8 09:46:23 2003

catch {rename Teem_UnuNtoZ_UnuProject ""}

itcl_class Teem_UnuNtoZ_UnuProject {
    inherit Module
    constructor {config} {
        set name UnuProject
        set_defaults
    }
    method set_defaults {} {
        global $this-axis
        set $this-axis 0

        global $this-measure
        set $this-measure 2


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

        iwidgets::entryfield $w.f.options.axis -labeltext "axis:" -textvariable $this-axis
        pack $w.f.options.axis -side top -expand yes -fill x

	make_labeled_radio $w.f.options.measure \
	    "Projection Measure"  "" top  $this-measure \
	    {{"Minimum" 1} \
		 {"Maximum" 2} \
		 {"Mean" 3} \
		 {"Median" 4} \
		 {"Mode" 5} \
		 {"Product" 6} \
		 {"Sum" 7} \
		 {"L1" 8} \
		 {"L2" 9} \
		 {"L-infinity" 10} \
		 {"Variance" 11} \
		 {"Standard Deviation" 12}}
		 
        pack $w.f.options.measure -side top -expand yes -fill x
	pack $w.f -expand 1 -fill x

	makeSciButtonPanel $w.f $w $this
	moveToCursor $w

	pack $w.f -expand 1 -fill x
    }
}
