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
 #  SetupFEMatrix.tcl
 #  Written by:
 #   David Weinstein
 #   Department of Computer Science
 #   University of Utah
 #   Aug 1996, March 2001
 #  Copyright (C) 1996 SCI Group
 ##

catch {rename BioPSE_Forward_SetupFEMatrix ""}

itcl_class BioPSE_Forward_SetupFEMatrix {
    inherit Module
    constructor {config} {
        set name SetupFEMatrix
        set_defaults
    }
    method set_defaults {} {
	global $this-UseCondTCL
	global $this-UseBasisTCL
	global $this-nprocs
	set $this-UseCondTCL 1
	set $this-UseBasisTCL 0
	set $this-nprocs "auto"
    }
    method ui {} {
        set w .ui[modname]

        if {[winfo exists $w]} {
            return
        }
	global $this-UseCondTCL
	global $this-UseBasisTCL
	global $this-nprocs

        toplevel $w

	frame $w.np
	label $w.np.l -text "Number of Threads"
	entry $w.np.e -width 5 -textvariable $this-nprocs -justify center
	pack $w.np.l $w.np.e -side left

	checkbutton $w.c -text "Use Conductivities" \
	    -variable $this-UseCondTCL
	checkbutton $w.b -text "Use Conductivity Basis Matrices" \
	    -variable $this-UseBasisTCL
	pack $w.np $w.c $w.b -side top -anchor w -padx 4 -pady 2

	makeSciButtonPanel $w $w $this
	moveToCursor $w
    }
}
