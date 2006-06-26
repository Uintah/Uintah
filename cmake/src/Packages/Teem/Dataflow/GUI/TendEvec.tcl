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

#    File   : TendEvec.tcl
#    Author : Martin Cole
#    Date   : Mon Sep  8 09:46:23 2003

catch {rename Teem_Tend_TendEvec ""}

itcl_class Teem_Tend_TendEvec {
    inherit Module
    constructor {config} {
        set name TendEvec
        set_defaults
    }
    method set_defaults {} {
	global $this-major
	set $this-major 1

	global $this-medium
	set $this-medium 0

	global $this-minor
	set $this-minor 0

        global $this-threshold
        set $this-threshold 0.5
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }

        toplevel $w

        frame $w.f
	pack $w.f -padx 2 -pady 2 -side top -expand yes
	
	frame $w.f.options
	pack $w.f.options -side top -expand yes

	checkbutton $w.f.options.major -text "Major" -variable $this-major
        pack $w.f.options.major -side top -expand yes -fill x

	checkbutton $w.f.options.medium -text "Medium" -variable $this-medium
        pack $w.f.options.medium -side top -expand yes -fill x

	checkbutton $w.f.options.minor -text "Minor" -variable $this-minor
        pack $w.f.options.minor -side top -expand yes -fill x

        iwidgets::entryfield $w.f.options.threshold -labeltext "Threshold:" \
	    -textvariable $this-threshold
        pack $w.f.options.threshold -side top -expand yes -fill x

	makeSciButtonPanel $w $w $this
	moveToCursor $w

	pack $w.f -expand 1 -fill x
    }
}
