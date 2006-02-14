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
 #  UnuQuantize.tcl: The UnuQuantize UI
 #  Written by:
 #   David Weinstein
 #   Department of Computer Science
 #   University of Utah
 #   Jan 2000
 #  Copyright (C) 2000 SCI Group
 ##

catch {rename Teem_UnuNtoZ_UnuQuantize ""}

itcl_class Teem_UnuNtoZ_UnuQuantize {
    inherit Module
    constructor {config} {
        set name UnuQuantize
        set_defaults
    }
    method set_defaults {} {
        global $this-minf
        global $this-maxf
        global $this-nbits
	global $this-useinputmin
	global $this-useinputmax
	global $this-realmin
	global $this-realmax

        set $this-minf 0
        set $this-maxf 255
	set $this-nbits 32
	set $this-useinputmin 1
	set $this-useinputmax 1
	set $this-realmin "unknown"
	set $this-realmax "unknown"
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
	
        toplevel $w
        wm minsize $w 200 80
        frame $w.f
        pack $w.f -padx 2 -pady 2 -side top -expand yes


 	frame $w.f.vals 
 	pack $w.f.vals -side top -anchor nw -expand yes -fill x

	frame $w.f.vals.min
 	pack $w.f.vals.min -side left -anchor nw -expand yes -fill x \
	    -padx 4

        label $w.f.vals.min.l -text "Min:" 
	entry $w.f.vals.min.v -textvariable $this-realmin \
	    -width 10 -state disabled -relief flat -foreground "#663399"
        pack $w.f.vals.min.l $w.f.vals.min.v -side left -expand yes -fill x

	frame $w.f.vals.max
 	pack $w.f.vals.max -side left -anchor nw -expand yes -fill x \
	    -padx 4

        label $w.f.vals.max.l -text "Max:" 
	entry $w.f.vals.max.v -textvariable $this-realmax \
	    -width 10 -state disabled -relief flat -foreground "#663399"
        pack $w.f.vals.max.l $w.f.vals.max.v -side left -expand yes -fill x

	frame $w.f.min -relief groove -borderwidth 2
	pack $w.f.min -side top -expand yes -fill x

        iwidgets::entryfield $w.f.min.v -labeltext "Min:" \
	    -textvariable $this-minf
        pack $w.f.min.v -side top -expand yes -fill x

        checkbutton $w.f.min.useinputmin \
	    -text "Use lowest value of input nrrd as min" \
	    -variable $this-useinputmin
        pack $w.f.min.useinputmin -side top -expand yes -fill x


	frame $w.f.max -relief groove -borderwidth 2
	pack $w.f.max -side top -expand yes -fill x

        iwidgets::entryfield $w.f.max.v -labeltext "Max:" \
	    -textvariable $this-maxf
        pack $w.f.max.v -side top -expand yes -fill x

        checkbutton $w.f.max.useinputmax \
	    -text "Use highest value of input nrrd as max" \
	    -variable $this-useinputmax
        pack $w.f.max.useinputmax -side top -expand yes -fill x

 	make_labeled_radio $w.f.nbits "Number of bits:" "" \
 	 		left $this-nbits \
 			{{8 8} \
 			{16 16} \
 			{32 32}}
	pack $w.f.nbits -side top -expand 1 -fill x

	makeSciButtonPanel $w $w $this
	moveToCursor $w

	pack $w.f -expand 1 -fill x
    }
}
