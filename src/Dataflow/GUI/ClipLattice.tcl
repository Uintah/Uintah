#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  
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


itcl_class SCIRun_FieldsCreate_ClipLattice {
    inherit Module

    constructor {config} {
        set name ClipLattice
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }

        toplevel $w
        wm minsize $w 150 20
        frame $w.f
        pack $w.f -padx 2 -pady 2 -side top -expand yes
	global $this-text-min-x
	global $this-text-min-y
	global $this-text-min-z
	global $this-text-max-x
	global $this-text-max-y
	global $this-text-max-z
	global $this-use-text-bbox
	frame $w.f.min -relief sunken -borderwidth 2
	label $w.f.min.l -text "Min Clip Point"
	pack $w.f.min.l -side top
	labelentry $w.f.min.x "X:" $this-text-min-x
	labelentry $w.f.min.y "Y:" $this-text-min-y
	labelentry $w.f.min.z "Z:" $this-text-min-z
	pack $w.f.min.x $w.f.min.y $w.f.min.z -side top -fill x -pady 3 -expand 1
	frame $w.f.max -relief sunken -borderwidth 2
	label $w.f.max.l -text "Max Clip Point"
	pack $w.f.max.l -side top
	labelentry $w.f.max.x "X:" $this-text-max-x
	labelentry $w.f.max.y "Y:" $this-text-max-y
	labelentry $w.f.max.z "Z:" $this-text-max-z
	pack $w.f.max.x $w.f.max.y $w.f.max.z -side top -fill x -pady 3 -expand 1

	checkbutton $w.c -text "Use This BBox" -variable $this-use-text-bbox

	pack $w.f.min $w.f.max -side left -fill x -padx 3 -pady 3 -expand 1
        pack $w.c

	makeSciButtonPanel $w $w $this
	moveToCursor $w
    }
}
