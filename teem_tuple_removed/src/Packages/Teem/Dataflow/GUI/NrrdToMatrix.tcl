#
#  The contents of this file are subject to the University of Utah Public
#  License (the "License"); you may not use this file except in compliance
#  with the License.
#  
#  Software distributed under the License is distributed on an "AS IS"
#  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
#  License for the specific language governing rights and limitations under
#  the License.
#  
#  The Original Source Code is SCIRun, released March 12, 2001.
#  
#  The Original Source Code was developed by the University of Utah.
#  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
#  University of Utah. All Rights Reserved.
#

##
 #  NrrdToMatrix.tcl
 #  Written by:
 #   Darby Van Uitert
 #   April 2004
 ##

itcl_class Teem_DataIO_NrrdToMatrix {
    inherit Module
    constructor {config} {
        set name NrrdToMatrix
        set_defaults
    }

    method set_defaults {} {
	global $this-cols
	global $this-entry
	global $this-which

	set $this-cols {-1}
	set $this-entry 3
	set $this-which 0

	trace variable $this-entry w "$this entry_changed"
    }

    method entry_changed {name1 name2 op} {
	update_which
    }

    method update_which {} {
	if {[set $this-which] == 1} {
	    # user specifies columns
	    set $this-cols [set $this-entry]
	} else {
	    # auto
	    set $this-cols {-1}
	}
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }

        toplevel $w

	frame $w.cols
	pack $w.cols -side top -anchor nw -padx 3 -pady 3
	radiobutton $w.cols.auto -text "Auto" \
	    -variable $this-which \
	    -value 0 \
	    -command "$this update_which"
	radiobutton $w.cols.spec -text "Columns (for Sparse Row Matrix): " \
	    -variable $this-which \
	    -value 1 \
	    -command "$this update_which"
	entry $w.cols.e -textvariable $this-entry 
	pack $w.cols.auto $w.cols.spec $w.cols.e -side left -anchor nw -padx 3 -pady 3

	makeSciButtonPanel $w $w $this
	moveToCursor $w
    }
}


