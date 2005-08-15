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

#    File   : UnuAxinfo.tcl
#    Author : Darby Van Uitert
#    Date   : January 2004


itcl_class Teem_UnuAtoM_UnuAxinfo {
    inherit Module
    constructor {config} {
        set name UnuAxinfo
        set_defaults
    }
    
    method set_defaults {} {
	# the width of the first column of the data display
	global $this-firstwidth
	global $this-axis
	global $this-label
	global $this-kind
	global $this-min
	global $this-max
	global $this-spacing
	global $this-reset
	
	set $this-firstwidth 12
	set $this-axis 0
	set $this-label "---"
	set $this-kind "nrrdKindUnknown"
	set $this-spacing 1.0
	set $this-min 0
	set $this-max 1.0
	set $this-reset 0
    }
    	
    
    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w

	makelabelentry $w.ax "Axis" $this-axis
	Tooltip $w.ax "Axis to modify\n(integer index)."

	makelabelentry $w.l "Label" $this-label
	Tooltip $w.l "Label to associate\nwith this axis"

	make_kind_optionmenu $w.k $this-kind

	makelabelentry $w.sp "Spacing" $this-spacing
	Tooltip $w.sp "Change spacing between\nsamples. This should be\nexpressed as a double."

	makelabelentry $w.mn "Min" $this-min
	Tooltip $w.mn "Change the minimum value\n. This should be expressed\nas a double."

	makelabelentry $w.mx "Max" $this-max
	Tooltip $w.mx "Change the maximum value\n. This should be expressed\nas a double."
	
	pack $w.ax $w.l $w.sp $w.mn $w.mx  -side top
	
	makeSciButtonPanel $w $w $this
	moveToCursor $w
    }
    
    
    method labelpair { win text1 text2 } {
	global $text2
	
	frame $win 
	pack $win -side top -padx 5 -pady 1
	label $win.l1 -text $text1 -width [set $this-firstwidth] \
	    -anchor w -just left
	label $win.colon  -text ":" -width 2 -anchor w -just left 
	label $win.l2 -textvar $text2 -width 40 -anchor w -just left \
	    -fore darkred
	pack $win.l1 $win.colon -side left -anchor nw
	pack $win.l2 -side left -fill x -expand 1 -anchor nw
    } 
    
    method makelabelentry { win l var} {
	global $var
	
	frame $win
	pack $win -side left -padx 5 -pady 1 -fill x
	label $win.l -text "$l" -width [set $this-firstwidth] \
	    -anchor w -just left
	label $win.colon -text ":" -width 2 -anchor w -just left
	entry $win.e -textvar $var \
	    -foreground darkred
	
	pack $win.l $win.colon -side left
	pack $win.e -side left -fill x -expand 1
    } 
    
    method make_kind_optionmenu { win var} {
	global $var

	iwidgets::optionmenu $win -labeltext "Kind:" \
	    -labelpos w -command "$this update_kind $win $var"
	$win insert end nrrdKindUnknown nrrdKindDomain nrrdKindScalar \
	    nrrdKind3Color nrrdKind3Vector nrrdKind3Normal \
	    nrrdKind3DSymMatrix nrrdKind3DMaskedSymMatrix nrrdKind3DMatrix \
	    nrrdKindList nrrdKindStub
	if {[info exists $var] && [set $var] != ""} {
	    $win select [set $var]
	} else {
	    $win select nrrdKindUnknown
	}
	pack $win -side top -anchor nw -padx 3 -pady 3

	trace variable $var w "$this update_kind_menu"
    }
    
    method update_kind {w var} {
	set which [$w get]
	
	set $var $which
    }

    method update_kind_menu {name1 name2 op} {
	set window .ui[modname]
	
	if {[winfo exists $window]} {
	    set op $window.k
	    $op select [set $this-kind]
	}
    }

}
