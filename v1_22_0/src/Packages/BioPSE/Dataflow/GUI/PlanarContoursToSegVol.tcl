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
 #  PlanarContoursToSegVol.tcl: Turn a contour set into a seg field
 #
 #  Written by:
 #   David Weinstein
 #   School of Computing
 #   University of Utah
 #   March 2003
 #
 #  Copyright (C) 2003 SCI Institute
 # 
 ##

catch {rename BioPSE_Modeling_PlanarContoursToSegVol ""}

itcl_class DaveW_FEM_PlanarContoursToSegVol {
    inherit Module
    constructor {config} {
        set name PlanarContoursToSegVol
        set_defaults
    }
    method set_defaults {} {
	global $this-nxTCL
	set $this-nxTCL "16"
	global $this-nyTCL
	set $this-nyTCL "16"
    }
    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }

        toplevel $w
        wm minsize $w 100 30
        frame $w.f
        set n "$this-c needexecute "
	global $this-nxTCL
	global $this-nyTCL
	frame $w.f.nx
	label $w.f.nx.l -text "NX: "
	entry $w.f.nx.e -relief sunken -width 4 -textvariable $this-nxTCL
	pack $w.f.nx.l $w.f.nx.e -side left
	frame $w.f.ny
	label $w.f.ny.l -text "NY: "
	entry $w.f.ny.e -relief sunken -width 4 -textvariable $this-nyTCL
	pack $w.f.ny.l $w.f.ny.e -side left
	pack $w.f.nx $w.f.ny -side top
        pack $w.f -side top -expand yes
    }
}
