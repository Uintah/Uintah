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
 #  ExtractSingleSurface.tcl: Extract a single component from a separating 
 #                            surface
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

catch {rename BioPSE_Modeling_ExtractSingleSurface ""}

itcl_class BioPSE_Modeling_ExtractSingleSurface {
    inherit Module
    constructor {config} {
        set name ExtractSingleSurface
        set_defaults
    }
    method set_defaults {} {
	global $this-surfid
	global $this-data
	set $this-surfid "0"
	set $this-data "material"
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
	global $this-surfid
	frame $w.f.surf
	label $w.f.surf.l -text "SurfId: "
	entry $w.f.surf.e -relief sunken -width 10 -textvariable $this-surfid
	bind $w.f.surf.e <Return> $n
	pack $w.f.surf.l $w.f.surf.e -side left
	pack $w.f.surf -side left -fill x
	frame $w.data
	label $w.data.label -text "Data values based on:"
	radiobutton $w.data.material -text "Material" \
		-variable $this-data -value material
	radiobutton $w.data.cindex -text "Component index" \
		-variable $this-data -value cindex
	radiobutton $w.data.size -text "Size (voxels)" \
		-variable $this-data -value size
	pack $w.data.label $w.data.material $w.data.cindex $w.data.size \
	    -side top -anchor w
        pack $w.f $w.data -side top -expand yes
    }
}
