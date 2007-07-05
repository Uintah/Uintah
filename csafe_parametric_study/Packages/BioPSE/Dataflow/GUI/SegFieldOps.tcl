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


itcl_class BioPSE_Modeling_SegFieldOps {
    inherit Module
    constructor {config} {
        set name SegFieldOps
        set_defaults
    }

    method set_defaults {} {
	global $this-min_comp_size
	set $this-min_comp_size 100
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }
        toplevel $w

	frame $w.s
	scale $w.s.s -orient horizontal -from 1 -to 1000 \
	    -showvalue true -variable "$this-min_comp_size"
	pack $w.s.s -side top -fill x -expand 1
	
	frame $w.b
	button $w.b.e -text "Execute" -command "$this-c needexecute"
	button $w.b.a -text "Audit" -command "$this-c audit"
	button $w.b.p -text "Print" -command "$this-c print"
	button $w.b.k -text "Absorb" -command "$this-c absorb"
	button $w.b.r -text "Reset" -command "$this-c reset"
	pack $w.b.e $w.b.a $w.b.p $w.b.p $w.b.k $w.b.r -side left -fill x -expand 1 -padx 3
	pack $w.s $w.b -side top -fill x -expand 1
    }
}
