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


itcl_class SCIRun_FieldsCreate_ClipByFunction {
    inherit Module
    constructor {config} {
        set name ClipByFunction
        set_defaults
    }

    method set_defaults {} {
        global $this-clipfunction
	global $this-clipmode
	global $this-u0
	global $this-u1
	global $this-u2
	global $this-u3
	global $this-u4
	global $this-u5
	set $this-clipfunction "x < 0"
	set $this-clipmode "cell"
	set $this-u0 0.0
	set $this-u1 0.0
	set $this-u2 0.0
	set $this-u3 0.0
	set $this-u4 0.0
	set $this-u5 0.0
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }

        toplevel $w

	set c "$this-c needexecute"

	frame $w.location -relief groove -borderwidth 2
	label $w.location.label -text "Location To Test"
        Tooltip $w.location "The function will be evaluated at these location(s) to determine which elements are preserved."
	radiobutton $w.location.cell -text "Element Center" \
	    -variable $this-clipmode -value cell -command $c
	radiobutton $w.location.nodeone -text "One Node" \
	    -variable $this-clipmode -value onenode -command $c
	radiobutton $w.location.nodeall -text "All Nodes" \
	    -variable $this-clipmode -value allnodes -command $c

	pack $w.location.label -side top -expand yes -fill both
	pack $w.location.cell $w.location.nodeone $w.location.nodeall \
	    -side top -anchor w

	frame $w.function -borderwidth 2
	label $w.function.l -text "F(x, y, z, v)"
	entry $w.function.e -width 20 -textvariable $this-clipfunction
	bind $w.function.e <Return> $c
	pack $w.function.l -side left
	pack $w.function.e -side left -fill x -expand 1

	pack $w.location $w.function -side top -fill x -expand 1 \
	    -padx 5 -pady 5

	makeSciButtonPanel $w $w $this
	moveToCursor $w
    }
}
