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

#   File: UnuMake.tcl
#   Author: Darby Van Uitert
#   Date: February 2004

package require Iwidgets 3.0 

itcl_class Teem_Unu_UnuMake {
    inherit Module

    constructor {config} {
        set name UnuMake
        set_defaults
    }

    method set_defaults {} {
	global $this-filename
	global $this-header_filename
	global $this-header_filetype
	global $this-label
	global $this-type
	global $this-axis
	global $this-sel
	global $this-write_header
	global $this-data_type
	global $this-samples
	global $this-spacing
	global $this-labels
	global $this-content
	global $this-line_skip
	global $this-byte_skip
	global $this-endian
	global $this-encoding

	set $this-filename ""
	set $this-header_filename ""
	set $this-header_filetype ASCII
	set $this-label unknown
	set $this-type Scalar
	set $this-axis ""
	set $this-sel ""
	set $this-write_header 0
	set $this-data_type "unsigned char"
	set $this-samples ""
	set $this-spacing ""
	set $this-labels ""
	set $this-content ""
	set $this-line_skip 0
	set $this-byte_skip 0
	set $this-endian "little"
	set $this-encoding "Raw"

	trace variable $this-data_type w "$this set_data_type"
	trace variable $this-encoding w "$this set_encoding"
    }

    # Method for browsing to the data file
    method make_file_open_box {} {
	global env
	global $this-filename

	set w [format "%s-fb" .ui[modname]]

	if {[winfo exists $w]} {
	    if { [winfo ismapped $w] == 1} {
		raise $w
	    } else {
		wm deiconify $w
	    }
	    return $w
	}

	toplevel $w -class TkFDialog
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
	set defext ".raw"
	set title "Specify raw file"
	
	# file types to appers in filter box
	set types {
	    {{Raw File}      {.raw}           }
	    {{Hex File}      {.hex}           }
	    {{Zipped File}      {.zip .gzip}           }
	    {{All Files}          {.*}            }
	}
	
	######################################################

	makeOpenFilebox \
	    -parent $w \
	    -filevar $this-filename \
	    -command "set $this-axis \"\"; wm withdraw $w" \
	    -cancel "wm withdraw $w" \
	    -title $title \
	    -filetypes $types \
	    -initialdir $initdir \
	    -defaultextension $defext

	moveToCursor $w
	wm deiconify $w

	return $w
    }

    # Method for browsing to a header file in
    # the case of writing out the header.
    method make_file_save_box {} {
	global env
	global $this-header_filename
	global $this-header_filetype
	
	set w [format "%s-hfb" .ui[modname]]
	
	if {[winfo exists $w]} {
	    if { [winfo ismapped $w] == 1} {
		raise $w
	    } else {
		wm deiconify $w
	    }
	    return $w
	}
	toplevel $w -class TkFDialog
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
	set defext ".nhdr"
	
	# name to appear initially
	set defname "MyNrrd"
	set title "Save nrrd header"
	
	# file types to appers in filter box
	set types {
	    {{Nrrd Header}     {.nhdr}      }
	    {{All Files}       {.*}   }
	}
	
	######################################################
	
	makeSaveFilebox \
	    -parent $w \
	    -filevar $this-header_filename \
	    -command "wm withdraw $w" \
	    -cancel "wm withdraw $w" \
	    -title $title \
	    -filetypes $types \
	    -initialfile $defname \
	    -initialdir $initdir \
	    -defaultextension $defext \
	    -formatvar $this-header_filetype

	moveToCursor $w
	wm deiconify $w
	
	return $w
    }

    # Method called to update type
    method update_type {om} {
	global $this-type
	set $this-type [$om get]
    }
    
    # Method called when encoding changed
    method update_encoding {om} {
 	global $this-encoding
 	set which [$om get]
	 set $this-encoding $which
    }

    # Method called when $this-encoding value changes.
    # This helps when loading a saved network to sync
    # the optionmenu.
    method set_encoding { name1 name2 op } {
	set window .ui[modname]
	if {[winfo exists $window]} {
	    set opt [$window.inf childsite]
	    $opt.c.enc select [set $this-encoding]
	}
    }


    # Set the axis variable
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

    # Method called when optionmenu for data
    # type changes
    method update_data_type {menu} {
	global $this-data_type
	set which [$menu get]
	set $this-data_type $which
    }

    # Method called when $this-data_type value changes.
    # This helps when loading a saved network to sync
    # the optionmenu.
    method set_data_type { name1 name2 op } {
	set window .ui[modname]
	if {[winfo exists $window]} {
	    set opt [$window.inf childsite]
	    $opt.dtype select [set $this-data_type]
	}
    }

    # Method to create the data file info box
    method make_data_info_box {w} {
	global $this-data_type
	global $this-write_header

	iwidgets::labeledframe $w \
	    -labelpos nw -labeltext "Header Information"
	set inf [$w childsite]

	frame $inf.h
	pack $inf.h -side top -anchor nw

	checkbutton $inf.h.whead \
	    -text "Write header" \
	    -variable $this-write_header
	Tooltip $inf.h.whead "Write out the header."

	button $inf.h.browse -text " Specify Header File " \
	    -command "$this make_file_save_box"
	pack $inf.h.whead  $inf.h.browse -anchor nw -side left -padx 4 -pady 4
	Tooltip $inf.h.browse "Specify the header file to write out."

	iwidgets::optionmenu $inf.dtype -labeltext "Data Type:" \
	    -labelpos w -command "$this update_data_type $inf.dtype"
	$inf.dtype insert end char "unsigned char" short "unsigned short" \
	    int "unsigned int" "long long" "unsigned long long" float double
	$inf.dtype select [set $this-data_type]
	pack $inf.dtype -anchor nw -side top
	Tooltip $inf.dtype "Specify the type of data\n(e.g. uchar, int, float, double, etc.)"

	frame $inf.a
	pack $inf.a -side top -anchor nw -fill x -expand 1

	iwidgets::entryfield $inf.a.samples -labeltext "Samples:" \
	    -textvariable $this-samples 
	Tooltip $inf.a.samples "Number of samples along each\naxis (and an implicit indicator \nof the dimension of the nrrd).\nThese value should be represented as\n1 or more ints separated by spaces."
	pack $inf.a.samples -side left -anchor nw -padx 3 -pady 3 -fill x -expand 1

	iwidgets::entryfield $inf.a.spacing -labeltext "Spacing:" \
	    -textvariable $this-spacing 
	Tooltip $inf.a.spacing "Spacing between samples on each axis.\nUse nan for any non-spatial axes (e.g.\nspacing between red, green, and blue\nalong axis 0 of interleaved RGB image\ndata). These values should be represented\nas 1 or more doubles separated by spaces."
	pack $inf.a.spacing -side left -anchor nw -padx 3 -pady 3 -fill x -expand 1

	iwidgets::entryfield $inf.labels -labeltext "Axes Labels:" \
	    -textvariable $this-labels 
	Tooltip $inf.labels "Short string labels for each of the axes.\nThese should be represented as 1 or more\nstrings separated by spaces."
	pack $inf.labels -side top -anchor nw -fill x -expand 1 -padx 3 -pady 3

	iwidgets::entryfield $inf.content -labeltext "Content:  " \
	    -textvariable $this-content
	Tooltip $inf.content "Specifies the content string of the nrrd,\nwhich is built upon by many nrrd function\nto record a history of operations. This is\nrepresented as a single string."
	pack $inf.content -side top -anchor nw -fill x -expand 1 -padx 3 -pady 3

	frame $inf.b
	pack $inf.b -side top -anchor nw -fill x -expand 1

	iwidgets::entryfield $inf.b.lines -labeltext "Line Skip:" \
	    -textvariable $this-line_skip
	Tooltip $inf.b.lines "Number of ascii lines to skip before\nreading data. This should be an integer."
	pack $inf.b.lines -side left -anchor nw -padx 3 -pady 3

	iwidgets::entryfield $inf.b.bytes -labeltext "Byte Skip:" \
	    -textvariable $this-byte_skip
	Tooltip $inf.b.bytes "Number of bytes to skip (after skipping\nascii lines, if any) before reading data.\nA value of -1 indicates a binary header of\nunknown length in raw-encoded data. This\nvalue should be represented as an int."
	pack $inf.b.bytes -side left -anchor nw -padx 3 -pady 3

	frame $inf.c
	pack $inf.c -side top -anchor nw -fill x -expand 1

	iwidgets::optionmenu $inf.c.enc -labeltext "Encoding:" \
	    -labelpos w -command "$this update_encoding $inf.c.enc"
	$inf.c.enc insert end Raw ASCII Hex Gzip Bzip2
	pack $inf.c.enc -side left -anchor nw -padx 3

	Tooltip $inf.c.enc "Output file format. Possibilities include:\n  raw : raw encoding\n  ascii : ascii values, one scanline per\n       line of text, values within line\n       are delimited by space, tab, or comma\n  hex : two hex digits per byte"


	label $inf.c.label -text "Endianness:"
	radiobutton $inf.c.big -text "Big" \
	    -variable $this-endian -value big
	radiobutton $inf.c.little -text "Little" \
	    -variable $this-endian -value little

	pack $inf.c.label $inf.c.big $inf.c.little \
	    -side left -anchor nw -padx 3 -pady 3

	Tooltip $inf.c.little "Endianness of data; relevent\nfor any data with value\nrepresentation bigger than 8 bits\nwith a non-ascii encoding. Defaults\nto endianness of this machine."
	Tooltip $inf.c.big "Endianness of data; relevent\nfor any data with value\nrepresentation bigger than 8 bits\nwith a non-ascii encoding. Defaults\nto endianness of this machine."

	button $inf.c.read -text " Generate " -command "$this-c generate_nrrd"
	pack $inf.c.read -side right -anchor ne -padx 3 -pady 3

	Tooltip $inf.c.read "Press to read in the data file\nand build a nrrd with the\nattributes given in the UI."

	pack $w -fill x -expand yes -side top
    }


    method ui {} {
	global env
	set w .ui[modname]

	if {[winfo exists $w]} {
	    set child [lindex [winfo children $w] 0]

	    # $w withdrawn by $child's procedures
	    raise $child
	    return
	}

	toplevel $w

	# read a nrrd
	iwidgets::labeledframe $w.f \
		-labeltext "File Reader Info"
	set f [$w.f childsite]

	iwidgets::entryfield $f.fname -labeltext "File:" \
	    -textvariable $this-filename

	button $f.sel -text "Browse" \
	    -command "$this make_file_open_box" -width 50
	Tooltip $f.sel "Select a data file to\nmake into a nrrd."

	pack $f.fname $f.sel -side top -fill x -expand yes -padx 4 -pady 4
	pack $w.f -fill x -expand yes -side top
	
	# header information
	make_data_info_box $w.inf

	# axis info and selection
	make_axis_info_sel_box $w.rb "$this set_axis $w.rb"


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

	pack $w.f1 -fill x -expand yes -side top

	makeSciButtonPanel $w $w $this
	moveToCursor $w

    }
}


