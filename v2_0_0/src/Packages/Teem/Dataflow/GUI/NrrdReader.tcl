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

# GUI for NrrdReader module
# by Samsonov Alexei
# December 2000

catch {rename Teem_DataIO_NrrdReader ""}

itcl_class Teem_DataIO_NrrdReader {
    inherit Module
    constructor {config} {
	set name NrrdReader
	set_defaults
    }

    method set_defaults {} {
	global $this-filename
	global $this-label
	global $this-type
	global $this-add
	global $this-axis
	global $this-sel

	set $this-filename ""
	set $this-label unknown
	set $this-type Scalar
	set $this-add 0
	set $this-axis ""
	set $this-sel ""
    }

    method make_file_open_box {} {
	global env
	global $this-filename

	set w [format "%s-fb" .ui[modname]]

	if {[winfo exists $w]} {
	    set child [lindex [winfo children $w] 0]

	    # $w withdrawn by $child's procedures
	    raise $child
	    return;
	}

	#toplevel $w
	set initdir ""
	
	# place to put preferred data directory
	# it's used if $this-filename is empty
	
	if {[info exists env(SCIRUN_DATA)]} {
	    set initdir $env(SCIRUN_DATA)
	} elseif {[info exists env(SCI_DATA)]} {
	    set initdir $env(SCI_DATA)
	} elseif {[info exists env(PSE_DATA)]} {
	    set initdir $env(PSE_DATA)
	}
	
	#######################################################
	# to be modified for particular reader

	# extansion to append if no extension supplied by user
	set defext ".fld"
	set title "Open nrrd file"
	
	# file types to appers in filter box
	set types {
            {{Nrrd Header File}     {.nhdr}    }
            {{Nrrd File}     {.nrrd}    }
	    {{NrrdData File}     {.nd}      }
	    {{All Files} {.*}   }
	}
	
	######################################################
	
	return \
	    [makeOpenFilebox \
		 -parent . \
		 -filevar $this-filename \
		 -command "set $this-axis \"\";$this-c read_nrrd;destroy " \
		 -cancel "destroy " \
		 -title $title \
		 -filetypes $types \
		 -initialdir $initdir \
		 -defaultextension $defext]
    }

    method update_type {om} {
	global $this-type
	set $this-type [$om get]
    }


    # set the axis variable
    method set_axis {w} {
	if {[get_selection $w] != ""} {
	    set $this-axis [get_selection $w]
	}
    }

    method clear_axis_info {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    delete_all_axes $w.rb
	}
    }

    method add_axis_info {id label center size spacing min max} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    add_axis $w.rb "axis$id" "Axis $id\nLabel: $label\nCenter: $center\nSize $size\nSpacing: $spacing\nMin: $min\nMax: $max"
	}
	# set the saved axis...
	if {[set $this-axis] == "axis$id"} {
	    if {[winfo exists $w.rb]} {
		select_axis $w.rb "axis$id"
	    }
	    set $this-axis "axis$id"
	}
    }

    method ui {} {
	global env
	set w .ui[modname]

	if {[winfo exists $w]} {
	    set child [lindex [winfo children $w] 0]

	    # $w withdrawn by $child's procedures
	    raise $child
	    return;
	}

	toplevel $w

	# read a nrrd
	iwidgets::labeledframe $w.f \
		-labeltext "Nrrd Reader Info"
	set f [$w.f childsite]

	iwidgets::entryfield $f.fname -labeltext "File:" \
	    -textvariable $this-filename

	button $f.sel -text "Browse" \
	    -command "$this make_file_open_box"

	pack $f.fname $f.sel -side top -fill x -expand yes -padx 4 -pady 4
	pack $w.f -fill x -expand yes -side top

	# axis info and selection
	make_axis_info_sel_box $w.rb "$this set_axis $w.rb"


	iwidgets::labeledframe $w.add \
		-labeltext "Tuple Axis Options"
	set add [$w.add childsite]

	text $add.instr -wrap word -width 50 -height 4 -relief flat 
	$add.instr insert end "Either Select which axis should be the tuple axis above, or select the add tuple axis option below. The label and type will be used to name the tuple axis."

	checkbutton $add.cb\
		-text "Add Tuple Axis" \
		-variable $this-add

	pack $add.instr $add.cb -fill x -expand yes -side top
	

	# set axis label and type
	iwidgets::labeledframe $w.f1 \
		-labelpos nw -labeltext "Set Tuple Axis Info"

	set f1 [$w.f1 childsite]

	iwidgets::entryfield $f1.lab -labeltext "Label:" \
	    -textvariable $this-label

	iwidgets::optionmenu $f1.type -labeltext "Type:" \
		-labelpos w -command "$this update_type $f1.type"
	$f1.type insert end Scalar Vector Tensor
	$f1.type select [set $this-type]

	pack $f1.lab $f1.type -fill x -expand yes -side top -padx 4 -pady 2

	pack $w.add $w.f1 -fill x -expand yes -side top

	makeSciButtonPanel $w $w $this
	moveToCursor $w

	if {[set $this-filename] != ""} {
	    $this-c read_nrrd
	}

    }
}
