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

#    File   : TendSten.tcl
#    Author : Darby Van Uitert
#    Date   : April 2004

itcl_class Teem_Tend_TendSten {
    inherit Module
    constructor {config} {
        set name TendSten
        set_defaults
    }

    method set_defaults {} {
	global $this-diffscale
	set $this-diffscale 1

	global $this-intscale
	set $this-intscale 2

	global $this-factor
	set $this-factor 1
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w

        frame $w.f
	pack $w.f -padx 2 -pady 2 -side top -expand yes
	
	frame $w.f.options
	pack $w.f.options -side top -expand yes

        iwidgets::entryfield $w.f.options.diffscale \
	    -labeltext "Difference Scale:" \
	    -textvariable $this-diffscale
        pack $w.f.options.diffscale -side top -expand yes -fill x

        iwidgets::entryfield $w.f.options.intscale \
	    -labeltext "Integration Scale:" \
	    -textvariable $this-intscale
        pack $w.f.options.intscale -side top -expand yes -fill x

        iwidgets::entryfield $w.f.options.factor \
	    -labeltext "Downsample Factor:" \
	    -textvariable $this-factor
        pack $w.f.options.factor -side top -expand yes -fill x

	makeSciButtonPanel $w.f $w $this
	moveToCursor $w

	pack $w.f -expand 1 -fill x
    }
}

