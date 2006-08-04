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


#  CriticalPointWidget.tcl
#  Written by:
#   James Purciful
#   Department of Computer Science
#   University of Utah
#   Apr. 1995
#  Copyright (C) 1995 SCI Group


catch {rename CriticalPointWidget ""}

itcl_class CriticalPointWidget {
    inherit BaseWidget
    constructor {config} {
	BaseWidget::constructor
	set name CriticalPointWidget
    }

    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	#wm minsize $w 100 100

	Dialbox $w.dialbox "CriticalPointWidget - Translate"
	$w.dialbox unbounded_dial 0 "Translate X" 0.0 1.0 \
		"$this-c translate x" "$this-c dialdone"
	$w.dialbox unbounded_dial 2 "Translate Y" 0.0 1.0 \
		"$this-c translate y" "$this-c dialdone"
	$w.dialbox unbounded_dial 4 "Translate Z" 0.0 1.0 \
		"$this-c translate z" "$this-c dialdone"

	frame $w.f
	Base_ui $w.f
	#button $w.f.dials -text "Dialbox" -command "$w.dialbox connect"
	#pack $w.f.dials -pady 5
	pack $w.f
    }
}
