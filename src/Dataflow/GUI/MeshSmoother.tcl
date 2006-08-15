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
 #  MeshSmoother.tcl: The MeshSmoother UI
 #  Written by:
 #   Jason Shepherd
 #   Department of Computer Science
 #   University of Utah
 #   March 2006
 #  Copyright (C) 2006 SCI Group
 ##

catch {rename SCIRun_FieldsGeometry_MeshSmoother ""}

itcl_class SCIRun_FieldsGeometry_MeshSmoother {
    inherit Module

    constructor {config} {
        set name MeshSmoother
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }

        toplevel $w
        wm minsize $w 150 80

	frame $w.bound1
	label $w.bound1.t1 -text "Smooth Boundary?"
	pack $w.bound1.t1
	pack $w.bound1

	frame $w.bound
	radiobutton $w.bound.smoothboundaryon -text "On" \
	    -variable $this-smoothboundary -value "On"
	radiobutton $w.bound.smoothboundaryoff -text "Off" \
	    -variable $this-smoothboundary -value "Off"
	pack $w.bound.smoothboundaryon $w.bound.smoothboundaryoff \
	    -side left -anchor nw -padx 3
	pack $w.bound -side top

	frame $w.sep
	label $w.sep.t1
 	label $w.sep.t2 -text "Smoothing Scheme"
	pack $w.sep.t1 $w.sep.t2
	pack $w.sep

	frame $w.style
	radiobutton $w.style.smartlaplacian -text "Smart Laplacian" \
	    -variable $this-smoothscheme -value "SmartLaplacian"

	radiobutton $w.style.shapeimprovement -text "Shape Improvement" \
	    -variable $this-smoothscheme -value "ShapeImprovement"

	radiobutton $w.style.none -text "None" \
	    -variable $this-smoothscheme -value "None"


	pack $w.style.smartlaplacian $w.style.shapeimprovement \
	    -side left -anchor nw -padx 3
	pack $w.style

        frame $w.f
	frame $w.fb
        pack $w.f $w.fb -padx 2 -pady 2 -side top -expand yes

        makeSciButtonPanel $w $w $this
	moveToCursor $w
    }
}
