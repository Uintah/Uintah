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

#    File   : TendSatin.tcl
#    Author : Darby Van Uitert
#    Date   : April 2004

itcl_class Teem_Tend_TendSatin {
    inherit Module
    constructor {config} {
        set name TendSatin
        set_defaults
    }

    method set_defaults {} {
	global $this-torus
	set $this-torus 0

	global $this-anisotropy
	set $this-anisotropy 1.0

	global $this-maxca1
	set $this-maxca1 1.0

	global $this-minca1
	set $this-minca1 0.0

	global $this-boundary
	set $this-boundary 0.05
	
	global $this-thickness
	set $this-thickness {0.3}

	global $this-size
	set $this-size {32}
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

	checkbutton $w.f.options.torus \
	    -text "Generate Torus Dataset" \
	    -variable $this-torus
	pack $w.f.options.torus -side top -anchor nw -padx 3 -pady 3

	iwidgets::entryfield $w.f.options.anisotropy \
	    -labeltext "Anisotropy Parameter:" \
	    -textvariable $this-anisotropy
        pack $w.f.options.anisotropy -side top -expand yes -fill x
	

        iwidgets::entryfield $w.f.options.maxca1 \
	    -labeltext "Max Anisotropy:" \
	    -textvariable $this-maxca1
        pack $w.f.options.maxca1 -side top -expand yes -fill x

        iwidgets::entryfield $w.f.options.minca1 \
	    -labeltext "Min Anisotropy:" \
	    -textvariable $this-minca1
        pack $w.f.options.minca1 -side top -expand yes -fill x

        iwidgets::entryfield $w.f.options.boundary \
	    -labeltext "Boundary:" \
	    -textvariable $this-boundary
        pack $w.f.options.boundary -side top -expand yes -fill x

        iwidgets::entryfield $w.f.options.thickness \
	    -labeltext "Thickness:" \
	    -textvariable $this-thickness
        pack $w.f.options.thickness -side top -expand yes -fill x

        iwidgets::entryfield $w.f.options.size \
	    -labeltext "Size:" \
	    -textvariable $this-size
        pack $w.f.options.size -side top -expand yes -fill x

	makeSciButtonPanel $w.f $w $this
	moveToCursor $w

	pack $w.f -expand 1 -fill x
    }
}
