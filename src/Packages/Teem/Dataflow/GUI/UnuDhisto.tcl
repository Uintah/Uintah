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

#    File   : UnuDhisto.tcl
#    Author : Martin Cole
#    Date   : Mon Sep  8 09:46:23 2003

catch {rename Teem_UnuAtoM_UnuDhisto ""}

itcl_class Teem_UnuAtoM_UnuDhisto {
    inherit Module
    constructor {config} {
        set name UnuDhisto
        set_defaults
    }
    method set_defaults {} {
        global $this-height
        set $this-height 0

        global $this-log
        set $this-log 1

	global $this-max
	set $this-max 0

	global $this-usemax
	set $this-usemax 1
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

        iwidgets::entryfield $w.f.options.height \
	    -labeltext "Height:" -textvariable $this-height
        pack $w.f.options.height -side top -expand yes -fill x

        checkbutton $w.f.options.log \
	    -text "Show log-scaled histogram:" -variable $this-log
        pack $w.f.options.log -side top -anchor nw

	frame $w.f.options.max -relief groove -borderwidth 2
	pack $w.f.options.max -side top -expand yes -fill x

        iwidgets::entryfield $w.f.options.max.v \
	    -labeltext "Max Number of Hits:" -textvariable $this-max
        pack $w.f.options.max.v -side top -anchor nw

        checkbutton $w.f.options.max.usemax \
	    -text "Use Max Number of Hits:" -variable $this-usemax
        pack $w.f.options.max.usemax -side top -anchor nw

	makeSciButtonPanel $w.f $w $this
	moveToCursor $w

	pack $w.f -expand 1 -fill x
    }
}
