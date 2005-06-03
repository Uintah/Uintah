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


itcl_class SCIRun_FieldsGeometry_MapDataToMeshCoord {
    inherit Module
    constructor {config} {
        set name MapDataToMeshCoord
        set_defaults
    }
    method set_defaults {} {
        global $this-coord
	set $this-coord 2
    }
    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }

        toplevel $w
        wm minsize $w 170 20
        frame $w.f

	global $this-coord
	frame $w.l
	label $w.l.label -text "Coordinate to replace with data: "
	pack $w.l.label -side left
	frame $w.b
	radiobutton $w.b.x -text "X    " -variable $this-coord -value 0
	radiobutton $w.b.y -text "Y    " -variable $this-coord -value 1
	radiobutton $w.b.z -text "Z    " -variable $this-coord -value 2
	radiobutton $w.b.n -text "Push surface nodes by (normal x data)" \
	    -variable $this-coord -value 3
	pack $w.b.x $w.b.y $w.b.z $w.b.n -side left -expand 1 -fill x
	pack $w.l $w.b -side top -fill both -expand 1

	makeSciButtonPanel $w $w $this
	moveToCursor $w
    }
}
