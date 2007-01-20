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

#    File   : UnuAxinfo.tcl
#    Author : Darby Van Uitert
#    Date   : January 2004


itcl_class Teem_UnuAtoM_UnuAxinfo {
    inherit Module

    constructor {config} {
        set name UnuAxinfo

	setGlobal $this-firstwidth 15
	setGlobal $this-spaceDir "0,0,0"
	setGlobal $this-use_spaceDir 0

	trace variable $this-kind   w "$this update_kind_menu"
	trace variable $this-center w "$this update_center_menu"
    }
    
    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w

	make_labelentry $w.ax "Axis" $this-axis 0
	Tooltip $w.ax "Axis to modify\n(integer index)."

	make_labelentry $w.l "Label" $this-label $this-use_label
	Tooltip $w.l "Label to associate\nwith this axis"
	Tooltip $w.l.ch "Use new label for output NRRD"

	make_optionmenu $w.k "Kind" $this-kind $this-use_kind nrrdKindUnknown \
	    "nrrdKindUnknown nrrdKindDomain nrrdKindScalar nrrdKind3Color nrrdKind3Vector nrrdKind3Normal nrrdKind3DSymMatrix nrrdKind3DMaskedSymMatrix nrrdKind3DMatrix nrrdKindList nrrdKindStub"
	Tooltip $w.k "Kind to associate\nwith this axis"
	Tooltip $w.k.ch "Use new axis kind for output NRRD"

	make_labelentry $w.sp "Spacing" $this-spacing $this-use_spacing
	Tooltip $w.sp "Change spacing between\nsamples. This should be\nexpressed as a double."
	Tooltip $w.sp.ch "Use new spacing for output NRRD"

	make_labelentry $w.mn "Min" $this-min $this-use_min
	Tooltip $w.mn "Change the minimum value\n. This should be expressed\nas a double."
	Tooltip $w.mn.ch "Use new min for output NRRD"

	make_labelentry $w.mx "Max" $this-max $this-use_max
	Tooltip $w.mx "Change the maximum value\n. This should be expressed\nas a double."
	Tooltip $w.mx.ch "Use new max for output NRRD"

	make_optionmenu $w.c "Kind" $this-center $this-use_center nrrdCenterUnknown \
	    "nrrdCenterUnknown nrrdCenterNode nrrdCenterCell"
	Tooltip $w.c "Center to associate\nwith this axis"
	Tooltip $w.c.ch "Use new axis center for output NRRD"

	make_labelentry $w.sd "Space Direction" $this-spaceDir $this-use_spaceDir
	Tooltip $w.sd "Change the spaceDirection vector.  This should be values separated by commas. NOTE: cannot set spaceDirection and min/max/spacing."
	Tooltip $w.sd.ch "Use spaceDirection vector for output NRRD.  NOTE: cannot set spaceDirection and min/max/spacing."
	
	pack $w.ax $w.l $w.k $w.sp $w.mn $w.mx $w.c $w.sd -side top
	
	makeSciButtonPanel $w $w $this
	moveToCursor $w
    }
    
    method make_labelentry {win l var use_var} {
	global $var
	
	frame $win
	pack $win -side left -padx 5 -pady 1 -fill x

	if { $use_var != 0 } {
	    checkbutton $win.ch -variable $use_var
	} else {
	    label $win.ch -text "   " -width 3 -anchor w -just left
	}

	label $win.l -text "$l" -width [set $this-firstwidth] \
	    -anchor w -just left
	label $win.colon -text ":" -width 2 -anchor w -just left
	entry $win.e -textvar $var \
	    -foreground darkred
	
	pack $win.ch $win.l $win.colon -side left
	pack $win.e -side left -fill x -expand 1
    } 
    
    method make_optionmenu {win l var use_var intial_val values} {
	global $var

#	puts stderr $values

	frame $win
	pack $win -side left -padx 5 -pady 1 -fill x

 	checkbutton $win.ch -variable $use_var

	label $win.l -text "$l" -width [set $this-firstwidth] \
	    -anchor w -just left

	label $win.colon -text ":" -width 2 -anchor w -just left

	iwidgets::optionmenu $win.om \
	    -command "$this update_optionmenu $win.om $var"

 	foreach value $values {
 	    $win.om insert end $value
 	}

	if {[info exists $var] && [set $var] != ""} {
	    $win.om select [set $var]
	} else {
	    $win.om select $intial_val
	}

	pack $win.ch $win.l $win.colon -side left
	pack $win.om -side left -fill x -expand 1
    }

    # Method called when $this-kind value changes.
    # This helps when loading a saved network to sync
    # the optionmenu.
    method update_kind_menu {name1 name2 op} {
	set window .ui[modname]
	
	if {[winfo exists $window]} {
	    set op $window.k.om
	    $op select [set $this-kind]
	}
    }

    # Method called when $this-center value changes.
    # This helps when loading a saved network to sync
    # the optionmenu.
    method update_center_menu {name1 name2 op} {
	set window .ui[modname]
	
	if {[winfo exists $window]} {
	    set op $window.c.om
	    $op select [set $this-center]
	}
    }

    # Method called when optionmenu changes to update the variable
    method update_optionmenu {w var} {
	global $var
	set $var [$w get]
    }

}
