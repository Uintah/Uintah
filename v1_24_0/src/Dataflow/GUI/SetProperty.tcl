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


##
 #  SetProperty.tcl: The SetProperty UI
 #  Written by:
 #   David Weinstein
 #   Department of Computer Science
 #   University of Utah
 #   Jan 2000
 #  Copyright (C) 2000 SCI Group
 ##

catch {rename SCIRun_FieldsOther_SetProperty ""}

itcl_class SCIRun_FieldsOther_SetProperty {
    inherit Module
    constructor {config} {
        set name SetProperty
        set_defaults
    }
    method set_defaults {} {
        global $this-prop
        global $this-val
        global $this-meshprop
        set $this-prop units
        set $this-val cm
        set $this-meshprop 1
    }
    method make_entry {w text v c} {
        frame $w
        label $w.l -text "$text"
        pack $w.l -side left
        global $v
        entry $w.e -textvariable $v
        bind $w.e <Return> $c
        pack $w.e -side right
    }
    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }

        toplevel $w
        wm minsize $w 200 80
        frame $w.f
        pack $w.f -padx 2 -pady 2 -side top -expand yes
	global $this-prop
	global $this-val
	global $this-meshprop
	make_entry $w.f.p "Property:" $this-prop "$this-c needexecute"
	make_entry $w.f.v "Value:" $this-val "$this-c needexecute"
	make_labeled_radio $w.f.m "Property belongs to:" "" \
		top $this-meshprop \
		{{"Field" 0} \
		{"Mesh" 1}}

	pack $w.f.p $w.f.v $w.f.m -side top -expand yes -fill x
	pack $w.f -expand 1 -fill x

	makeSciButtonPanel $w $w $this
	moveToCursor $w
    }
}
