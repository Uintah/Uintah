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
 #  UnuConvert.tcl: The UnuConvert UI
 #  Written by:
 #   David Weinstein
 #   Department of Computer Science
 #   University of Utah
 #   Jan 2000
 #  Copyright (C) 2000 SCI Group
 ##

catch {rename Teem_UnuAtoM_UnuConvert ""}

itcl_class Teem_UnuAtoM_UnuConvert {
    inherit Module
    constructor {config} {
        set name UnuConvert
        set_defaults
    }
    method set_defaults {} {
        global $this-type
	set $this-type 5
    }
    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }

        toplevel $w
        wm minsize $w 100 230
        frame $w.f
        pack $w.f -padx 2 -pady 2 -side top -expand yes
	global $this-type
	make_labeled_radio $w.f.t "New Type:" "" \
		top $this-type \
		{{char 1} \
		{uchar 2} \
		{short 3} \
		{ushort 4} \
		{int 5} \
		{uint 6} \
		{float 9} \
		{double 10}}

	makeSciButtonPanel $w $w $this
	moveToCursor $w

	pack $w.f.t -side top -expand 1 -fill x
	pack $w.f -expand 1 -fill x
    }
}
