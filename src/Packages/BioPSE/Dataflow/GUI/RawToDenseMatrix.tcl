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
#    File   : RawToDenseMatrix.tcl
#    Author : Martin Cole
#    Date   : Tue Jan 15 09:12:29 2002

itcl_class BioPSE_DataIO_RawToDenseMatrix {
    inherit Module
    constructor {config} {
        set name RawToDenseMatrix
        set_defaults
    }

    method set_defaults {} {
	global $this-filename
	global $this-pots
	global $this-units
	global $this-min
	global $this-max
	set $this-filename "none"
	set $this-pots "none"
	set $this-units "s"
	set $this-min 0.0
	set $this-max 1.0
    }

    method ui {} {
	global $this-filename

        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }
        toplevel $w

	frame $w.p1
	label $w.p1.units-l -text "Units :" -anchor w -just left
	entry $w.p1.units -width 14 -just left -textvariable $this-units
	pack $w.p1.units-l $w.p1.units -side left

	frame $w.p2
	label $w.p2.min-l -text "Start Time:" -anchor w -just left
	entry $w.p2.min -width 14 -just left -textvariable $this-min
	pack $w.p2.min-l $w.p2.min -side left

	frame $w.p3
	label $w.p3.max-l -text "End Time:" -anchor w -just left
	entry $w.p3.max -width 14 -just left -textvariable $this-max
	pack $w.p3.max-l $w.p3.max -side left


	button $w.sel -text "Open New File" -command "$this file_selection"
	pack $w.p1 $w.p2 $w.p3 $w.sel -side top
    }
    
    method file_selection {} {
	global $this-filename

        set w [format "%s-fb" .ui[modname]]
        if {[winfo exists $w]} {
            raise $w
            return
        }
	toplevel $w
	set initdir ""
	
	#######################################################
	# to be modified for particular reader

	# extansion to append if no extension supplied by user
	set defext ".pts"
	set title "Open CVRTI points file"
	
	# file types to appers in filter box
	set types {
	    {{CVRTI Points}     {.pts}      }
	    {{All Files} {.*}   }
	}
	
	######################################################
	
	makeOpenFilebox \
		-parent $w \
		-filevar $this-filename \
		-command "$this working_files from-gui; destroy $w" \
		-cancel "destroy $w" \
		-title $title \
		-filetypes $types \
		-initialdir $initdir \
		-defaultextension $defext

    }

    method working_files {full} {
	global $this-pots
	global $this-filename
	if {$full == "from-gui"} {
	    set full [set $this-filename]
	}
	set dir [regexp -inline {/.*/} $full]
	set pots [lsort [glob $dir*.pot]]

	for {set i 0} {$i < [llength $pots]} {incr i} {
	    $this-c add-pot [lindex $pots $i]
	}
	$this-c needexecute
    }
}


