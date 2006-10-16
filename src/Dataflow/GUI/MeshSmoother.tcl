#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  
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
        #wm minsize $w 80 130

	frame $w.bound -relief groove -borderwidth 2
	label $w.bound.t1 -text "Smooth Boundary"
	pack $w.bound.t1 

	radiobutton $w.bound.smoothboundaryon -text "On" \
	    -variable $this-smoothboundary -value "On"
	radiobutton $w.bound.smoothboundaryoff -text "Off" \
	    -variable $this-smoothboundary -value "Off"
	pack $w.bound.smoothboundaryon $w.bound.smoothboundaryoff \
	    -side left -anchor n

	frame $w.style -relief groove -borderwidth 2
 	label $w.style.t1 -text "Smoothing Scheme"
	pack $w.style.t1

	radiobutton $w.style.none -text "None" \
	    -variable $this-smoothscheme -value "None"
	radiobutton $w.style.laplacian -text "Laplacian" \
	    -variable $this-smoothscheme -value "Laplacian"
	radiobutton $w.style.smartlaplacian -text "Smart Laplacian" \
	    -variable $this-smoothscheme -value "SmartLaplacian"
	radiobutton $w.style.shapeimprovement -text "Shape Improvement" \
	    -variable $this-smoothscheme -value "ShapeImprovement"
	pack $w.style.none $w.style.laplacian $w.style.smartlaplacian $w.style.shapeimprovement \
	    -side top -anchor w

        pack $w.bound $w.style -side top -e y -f both -padx 5 -pady 5
        
        makeSciButtonPanel $w $w $this
	moveToCursor $w
    }
}
