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

catch {rename MatlabInterface_Math_Tikhonov ""}

itcl_class MatlabInterface_Math_Tikhonov {
    inherit Module
    constructor {config} {
        set name Tikhonov
        set_defaults
    }
    method set_defaults {} {
	global $this-hpTCL
	set $this-hpTCL ""
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
	global $this-hpTCL
	frame $w.f.hp
	label $w.f.hp.l -text "pars: "
	entry $w.f.hp.e -relief sunken -width 21 -textvariable $this-hpTCL
	pack $w.f.hp.l $w.f.hp.e -side left
	pack $w.f.hp -side top
        pack $w.f -side top -expand yes
    }
}
