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


itcl_class BioPSE_Modeling_SegFieldToLatVol {
    inherit Module
    constructor {config} {
        set name SegFieldToLatVol
        set_defaults
    }

    method set_defaults {} {
	global $this-lat_vol_data
	set $this-lat_vol_data componentMatl
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }
        toplevel $w

	frame $w.f
	label $w.f.l -text "Data values for output field"
	radiobutton $w.f.sv -text "Material" \
	    -variable $this-lat_vol_data -value componentMatl
	radiobutton $w.f.ci -text "Component Index" \
	    -variable $this-lat_vol_data -value componentIdx
	radiobutton $w.f.cs -text "Component Size" \
	    -variable $this-lat_vol_data -value componentSize
	pack $w.f.l -side top
	pack $w.f.sv $w.f.ci $w.f.cs -side top -anchor w
	pack $w.f -side top
    }
}
