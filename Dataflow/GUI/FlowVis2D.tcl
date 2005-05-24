#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2005 Scientific Computing and Imaging Institute,
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


catch {rename FlowVis2D ""}

itcl_class SCIRun_Visualization_FlowVis2D {
    inherit Module
    constructor {config} {
	set name FlowVis2D
	set_defaults
    }
    method set_defaults {} {
	global $this-vis_type
        global $this-tex_x
        global $this-tex_y
        global $this-auto_tex
        set $this-vis_type 0
        set $this-auto_tex 1
        set $this-tex_x 1
        set $this-tex_y 1
    }

    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    return
	}
	toplevel $w
	frame $w.f 
	pack $w.f -padx 2 -pady 2 -fill x
	
	set n "$this-c needexecute"
	set s "$this state"

        radiobutton $w.f.lic -text LIC -value 0 \
            -variable $this-vis_type
        radiobutton $w.f.ibfv -text IBFV -value 1 \
            -variable $this-vis_type
        radiobutton $w.f.lea -text LEA -value 2 \
            -variable $this-vis_type
        checkbutton $w.f.at -text "Auto Tex?" -variable $this-auto_tex \
            -command $n
        scale $w.f.x  -from 0 -to 1 -command $n \
            -resolution 0.01 -variable $this-tex_x -orient horizontal
        scale $w.f.y  -from 0 -to 1 -command $n \
            -resolution 0.01 -variable $this-tex_y -orient horizontal

        pack  $w.f.lic $w.f.ibfv $w.f.lea $w.f.at $w.f.x $w.f.y -side top

	makeSciButtonPanel $w $w $this
	moveToCursor $w
    }
}