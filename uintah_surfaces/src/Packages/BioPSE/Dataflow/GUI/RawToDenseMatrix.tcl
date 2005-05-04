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
            return
        }
        toplevel $w

	frame $w.p1
	label $w.p1.units-l -text "Units:" -anchor w -just left
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

	button $w.sel -text " Open New File " -command "$this file_selection"
	pack $w.p1 $w.p2 $w.p3 $w.sel -side top -padx 4 -pady 4 -anchor e

	makeSciButtonPanel $w $w $this
	moveToCursor $w
    }
    
    method file_selection {} {
	global $this-filename

        set w [format "%s-fb" .ui[modname]]
        if {[winfo exists $w]} {
	    if { [winfo ismapped $w] == 1} {
		raise $w
	    } else {
		wm deiconify $w
	    }
            return
        }
	toplevel $w -class TkFDialog
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
		-command "$this working_files from-gui; wm withdraw $w" \
		-cancel "wm withdraw $w" \
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


