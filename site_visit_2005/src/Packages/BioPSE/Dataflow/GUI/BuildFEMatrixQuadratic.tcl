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
 #  BuildFEMatrix.tcl
 #
 #  Written by:
 #   David Weinstein
 #   Department of Computer Science
 #   University of Utah
 #   Aug 1996
 #
 #  Copyright (C) 1996 SCI Group
 #
 ##

itcl_class BioPSE_Forward_BuildFEMatrixQuadratic {
    inherit Module
    constructor {config} {
        set name BuildFEMatrixQuadratic
        set_defaults
    }
    method set_defaults {} {
        global $this-BCFlag
	global $this-UseCondTCL
	global $this-refnodeTCL
        set $this-BCFlag "none"
	set $this-UseCondTCL 1
	set $this-refnodeTCL 0
    }
    method make_entry {w text v c} {
        frame $w
        label $w.l -text "$text"
        pack $w.l -side left
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
        wm minsize $w 150 20
        frame $w.f
        pack $w.f -padx 2 -pady 2 -side top -expand yes
        global $this-BCFlag
        make_labeled_radio $w.f.r "Boundary Conditions:" "" \
                left $this-BCFlag \
                {{"None" none} \
                {"Apply Dirichlet" DirSub} \
                {"Ground Average DC" AverageGround} \
                {"Use Reference Node" PinZero}}
	global $this-refnodeTCL
	make_entry $w.f.pinned "Reference node:" $this-refnodeTCL {}
	global $this-UseCondTCL
	checkbutton $w.f.b -text "Use Conductivities" -variable $this-UseCondTCL
	pack $w.f.r $w.f.pinned $w.f.b -side top -expand 1 -fill x
	pack $w.f -expand 1 -fill x
    }
}
