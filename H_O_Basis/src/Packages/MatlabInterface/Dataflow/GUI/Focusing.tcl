##
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

 #  Written by:
 #   Oleg
 #   Department of Computer Science
 #   University of Utah
 #   01May25
 ##

catch {rename MatlabInterface_Math_Focusing ""}

itcl_class MatlabInterface_Math_Focusing {
    inherit Module
    constructor {config} {
        set name Focusing
        set_defaults
    }
    method set_defaults {} {
	global $this-noiseGUI
	global $this-fcsdgGUI
	set $this-noiseGUI "0.01"
	set $this-fcsdgGUI "5"
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
	global $this-noiseGUI
	global $this-fcsdgGUI
	frame $w.f.noise
	label $w.f.noise.l -text "Noise: "
	entry $w.f.noise.e -relief sunken -width 20 \
		-textvariable $this-noiseGUI
	pack $w.f.noise.l $w.f.noise.e -side left
	frame $w.f.fcsdg
	label $w.f.fcsdg.l -text "FCSDG: "
	entry $w.f.fcsdg.e -relief sunken -width 20 \
		-textvariable $this-fcsdgGUI
	pack $w.f.fcsdg.l $w.f.fcsdg.e -side left
	pack $w.f.noise $w.f.fcsdg -side top
        pack $w.f -side top -expand yes
    }
}
