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

#    File   : UnuUnquantize.tcl
#    Author : Darby Van Uitert
#    Date   : April 2004

itcl_class Teem_UnuNtoZ_UnuUnquantize {
    inherit Module
    constructor {config} {
        set name UnuUnquantize
        set_defaults
    }

    method set_defaults {} {
	global $this-min
	global $this-useinputmin
	global $this-max
	global $this-useinputmax
	global $this-double

	set $this-min {0.0}
	set $this-useinputmin 1
	set $this-max {0.0}
	set $this-useinputmax 1
	set $this-double 0
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

	frame $w.f.options.min -relief groove -borderwidth 2
	pack $w.f.options.min -side top -expand yes -fill x

        iwidgets::entryfield $w.f.options.min.v -labeltext "Min:" \
	    -textvariable $this-min
        pack $w.f.options.min.v -side top -expand yes -fill x

        checkbutton $w.f.options.min.useinputmin \
	    -text "Use Input's Min" \
	    -variable $this-useinputmin
        pack $w.f.options.min.useinputmin -side top -anchor nw 

	frame $w.f.options.max -relief groove -borderwidth 2
	pack $w.f.options.max -side top -expand yes -fill x

        iwidgets::entryfield $w.f.options.max.v -labeltext "Max:" \
	    -textvariable $this-max
        pack $w.f.options.max.v -side top -expand yes -fill x

        checkbutton $w.f.options.max.useinputmax \
	    -text "Use Input's Max" \
	    -variable $this-useinputmax
        pack $w.f.options.max.useinputmax -side top -anchor nw 


	checkbutton $w.f.options.double -text "Use double for output type" \
	    -variable $this-double
	pack $w.f.options.double -side top -expand yes -fill x

	makeSciButtonPanel $w.f $w $this
	moveToCursor $w

	pack $w.f -expand 1 -fill x
    }
}


