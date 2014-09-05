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

#    File   : TendEvq.tcl
#    Author : Darby Van Uitert
#    Date   : April 2004

itcl_class Teem_Tend_TendEvq {
    inherit Module
    constructor {config} {
        set name TendEvq
        set_defaults
    }

    method set_defaults {} {
	global $this-index
	set $this-index 0

	global $this-anisotropy
	set $this-anisotropy "cl1"

	global $this-ns
	set $this-ns 0

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

	iwidgets::labeledframe $w.f.options.index \
	    -labeltext "Evec Index" \
	    -labelpos nw
	pack $w.f.options.index -side top -expand yes -fill x
	
	set index [$w.f.options.index childsite]
	radiobutton $index.1 \
	    -text "Largest Eigenvalue" \
	    -variable $this-index \
	    -value 0

	radiobutton $index.2 \
	    -text "Middle Eigenvalue" \
	    -variable $this-index \
	    -value 1

	radiobutton $index.3 \
	    -text "Smallest Eigenvalue" \
	    -variable $this-index \
	    -value 2

	pack $index.1 $index.2 \
	    $index.3 -side top -anchor nw -padx 3 -pady 3

	iwidgets::optionmenu $w.f.options.anisotropy \
	    -labeltext "Anisotropy Metric to Plot:" \
	    -labelpos w \
	    -command "$this update_anisotropy $w.f.options.anisotropy"
	$w.f.options.anisotropy insert end cl1 cl2 cp1 cp2 \
	    ca1 ca2 cs1 cs2 ct1 ct2 ra fa vf tr
	pack $w.f.options.anisotropy -side top -anchor nw -padx 3 -pady 3

        checkbutton $w.f.options.ns \
	    -text "Don't attenuate the color by Anisotropy:" \
	    -variable $this-ns
        pack $w.f.options.ns -side top -anchor nw

	makeSciButtonPanel $w.f $w $this
	moveToCursor $w

	pack $w.f -expand 1 -fill x
    }

    method update_anisotropy {menu} {
	global $this-anisotropy
	set which [$menu get]
	set $this-anisotropy $which
    }

    method set_anisotropy { name1 name2 op } {
	set w .ui[modname]
	set menu $w.f.options.anisotropy
	if {[winfo exists $menu]} {
	    $menu select [set $this-anisotropy]
	}
    }
}


