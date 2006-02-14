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


itcl_class SCIRun_Math_AppendMatrix {
    inherit Module
    constructor {config} {
        set name AppendMatrix
        set_defaults
    }
    method set_defaults {} {
        global $this-row
	set $this-row 0
        global $this-append
	set $this-append 0
        global $this-front
	set $this-front 0
        global $this-clear
	set $this-front 0
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
            return
        }

        toplevel $w
        wm minsize $w 150 20
        frame $w.f
        pack $w.f -padx 2 -pady 2 -side top -expand yes
        global $this-row
        make_labeled_radio $w.f.r "Rows/Columns" "" \
                top $this-row \
		{{"Row" 1} \
                {"Column" 0}}
        global $this-append
        make_labeled_radio $w.f.a "Append/Replace" "" \
                top $this-append \
		{{"Append" 1} \
                {"Replace" 0}}
        global $this-front
        make_labeled_radio $w.f.f "Prepend/Postpend" "" \
                top $this-front \
		{{"Prepend" 1} \
                {"Postpend" 0}}
	pack $w.f.r $w.f.a $w.f.f -side left -expand 1 -fill x

	makeSciButtonPanel $w $w $this "\"Clear Output\" \"$this-c clear\" \"\""
	moveToCursor $w
    }
}
