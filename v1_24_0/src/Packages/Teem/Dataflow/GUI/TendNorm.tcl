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

#    File   : TendEstim.tcl
#    Author : Martin Cole
#    Date   : Mon Sep  8 09:46:23 2003

catch {rename Teem_Tend_TendNorm ""}

itcl_class Teem_Tend_TendNorm {
    inherit Module
    constructor {config} {
        set name TendNorm
        set_defaults
    }
    method set_defaults {} {
        global $this-major-weight
        set $this-major-weight 1.0

        global $this-medium-weight
        set $this-medium-weight 1.0

        global $this-minor-weight
        set $this-minor-weight 1.0

        global $this-amount
        set $this-amount 1.0

        global $this-target
        set $this-target 1.0
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

        iwidgets::entryfield $w.f.options.major -labeltext "Major weight:" \
	    -textvariable $this-major-weight
        pack $w.f.options.major -side top -expand yes -fill x

        iwidgets::entryfield $w.f.options.medium -labeltext "Medium weight:" \
	    -textvariable $this-medium-weight
        pack $w.f.options.medium -side top -expand yes -fill x

        iwidgets::entryfield $w.f.options.minor -labeltext "Minor weight:" \
	    -textvariable $this-minor-weight
        pack $w.f.options.minor -side top -expand yes -fill x

        iwidgets::entryfield $w.f.options.amount -labeltext "Amount:" \
	    -textvariable $this-amount
	pack $w.f.options.amount -side top -expand yes -fill x

        iwidgets::entryfield $w.f.options.target -labeltext "Target:" \
	    -textvariable $this-target
        pack $w.f.options.target -side top -expand yes -fill x

	makeSciButtonPanel $w $w $this
	moveToCursor $w

	pack $w.f -expand 1 -fill x
    }
}
