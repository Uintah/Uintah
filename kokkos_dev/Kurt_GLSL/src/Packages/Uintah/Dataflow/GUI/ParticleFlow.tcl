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


itcl_class Uintah_Visualization_ParticleFlow {
    inherit Module
    constructor {config} {
	set name ParticleFlow
	set_defaults
    }
    method set_defaults {} {
	global $this-animate
        global $this-time
        set $this-animate 0
        set $this-time 0.002
    }

    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    return
	}
	toplevel $w
        frame $w.f
	frame $w.f1 
        frame $w.f2
        
        pack $w.f -side top
	pack $w.f1 $w.f2 -in $w.f -padx 2 -pady 2 -fill x -side left
	
	set n "$this-c needexecute"

        checkbutton $w.f1.animate -text Animate -variable $this-animate
        label $w.f2.l -text "Particle life time increment"
        scale $w.f2.scale -to 0.01 -from 0.0002 -orient horizontal \
            -variable $this-time -resolution 0.0002
        pack  $w.f1.animate 
        pack $w.f2.l $w.f2.scale  -side top

	makeSciButtonPanel $w $w $this
	moveToCursor $w

    }
}





