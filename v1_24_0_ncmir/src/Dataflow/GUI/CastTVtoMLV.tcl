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


itcl_class SCIRun_FieldsGeometry_CastTVtoMLV {
    inherit Module
    constructor {config} {
        set name CastTVtoMLV
	global $this-nx
	global $this-ny
	global $this-nz
        set_defaults
    }

    method set_defaults {} {
	set $this-nx 8
	set $this-ny 8
	set $this-nz 8
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w
	
	frame $w.x
	frame $w.y
	frame $w.z

	pack $w.x $w.y $w.z -side top -e y -f both -padx 5 -pady 5
	
	label $w.x.label -text "nx: "
	entry $w.x.entry -textvariable $this-nx
	bind $w.x.entry <Return> "$this-c needexecute"
	pack $w.x.label $w.x.entry -side left

	label $w.y.label -text "ny: "
	entry $w.y.entry -textvariable $this-ny
	bind $w.y.entry <Return> "$this-c needexecute"
	pack $w.y.label $w.y.entry -side left

	label $w.z.label -text "nz: "
	entry $w.z.entry -textvariable $this-nz
	bind $w.z.entry <Return> "$this-c needexecute"
	pack $w.z.label $w.z.entry -side left

	makeSciButtonPanel $w $w $this
	moveToCursor $w
    }
}


