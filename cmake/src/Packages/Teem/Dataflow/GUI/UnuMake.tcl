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


#   File: UnuMake.tcl
#   Author: Darby Van Uitert
#   Date: February 2004

package require Iwidgets 3.0 

itcl_class Teem_UnuAtoM_UnuMake {
    inherit Module

    constructor {config} {
        set name UnuMake
        set_defaults
    }

    method set_defaults {} {
	global $this-filename
	global $this-header_filename
	global $this-header_filetype
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
	global $this-key1
	global $this-key2
	global $this-key3
	global $this-val1
	global $this-val2
	global $this-val3
	global $this-kind

	set $this-filename ""
	set $this-header_filename ""
	set $this-header_filetype ASCII
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
	set $this-key1 ""
	set $this-key2 ""
	set $this-key3 ""
	set $this-val1 ""
	set $this-val2 ""
	set $this-val3 ""
	set $this-kind "nrrdKindUnknown"

	trace variable $this-data_type w "$this set_data_type"
	trace variable $this-encoding w "$this set_encoding"
    }

    # Method for browsing to the data file
    method make_file_open_box {} {
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
	
	# place to put preferred data directory
	# it's used if $this-filename is empty
	set initdir [netedit getenv SCIRUN_DATA]
	
	#######################################################
	# to be modified for particular reader

	# extansion to append if no extension supplied by user
	set defext ".raw"
	set title "Specify raw file"
	
	# file types to appers in filter box
	set types {
	    {{Raw File}      {.raw}           }
	    {{ASCII Text}      {.txt}           }
	    {{Hex File}      {.hex}           }
	    {{Zipped File}      {.zip .gzip}           }
	    {{All Files}          {.*}            }
	}
	
	######################################################

	makeOpenFilebox \
	    -parent $w \
	    -filevar $this-filename \
	    -command "wm withdraw $w" \
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
	
	# place to put preferred data directory
	# it's used if $this-filename is empty	
	set initdir [netedit getenv SCIRUN_DATA]
	
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
	    $opt.a.enc select [set $this-encoding]
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
	    $opt.a.dtype select [set $this-data_type]
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

	iwidgets::optionmenu $inf.kind -labeltext "nrrdKind of First Axis:" \
	    -labelpos w -command "$this update_kind $inf.kind"
	$inf.kind insert end nrrdKindUnknown nrrdKindDomain nrrdKindScalar \
	    nrrdKind3Color nrrdKind3Vector nrrdKind3Normal \
	    nrrdKind3DSymMatrix nrrdKind3DMaskedSymMatrix nrrdKind3DMatrix \
	    nrrdKindList nrrdKindStub
	pack $inf.kind -side top -anchor nw -padx 3 -pady 3
	$inf.kind select nrrdKindUnknown

	frame $inf.a
	pack $inf.a -side top -anchor nw -fill x -expand 1

	iwidgets::optionmenu $inf.a.dtype -labeltext "Data Type:" \
	    -labelpos w -command "$this update_data_type $inf.a.dtype"
	$inf.a.dtype insert end char "unsigned char" short "unsigned short" \
	    int "unsigned int" "long long" "unsigned long long" float double
	$inf.a.dtype select [set $this-data_type]
	pack $inf.a.dtype -anchor nw -side left
	Tooltip $inf.a.dtype "Specify the type of data\n(e.g. uchar, int, float, double, etc.)"

	iwidgets::optionmenu $inf.a.enc -labeltext "Encoding:" \
	    -labelpos w -command "$this update_encoding $inf.a.enc"
	$inf.a.enc insert end Raw ASCII Hex Gzip Bzip2
	pack $inf.a.enc -side left -anchor nw -padx 3
	$inf.a.enc select [set $this-encoding]

	Tooltip $inf.a.enc "Output file format. Possibilities include:\n  raw : raw encoding\n  ascii : ascii values, one scanline per\n       line of text, values within line\n       are delimited by space, tab, or comma\n  hex : two hex digits per byte"

	frame $inf.d
	pack $inf.d -side top -anchor nw -fill x -expand 1

	label $inf.d.label -text "Endianness:"
	radiobutton $inf.d.big -text "Big" \
	    -variable $this-endian -value big
	radiobutton $inf.d.little -text "Little" \
	    -variable $this-endian -value little

	pack $inf.d.label $inf.d.big $inf.d.little \
	    -side left -anchor nw -padx 3 -pady 3

	Tooltip $inf.d.little "Endianness of data; relevent\nfor any data with value\nrepresentation bigger than 8 bits\nwith a non-ascii encoding. Defaults\nto endianness of this machine."
	Tooltip $inf.d.big "Endianness of data; relevent\nfor any data with value\nrepresentation bigger than 8 bits\nwith a non-ascii encoding. Defaults\nto endianness of this machine."




	frame $inf.b
	pack $inf.b -side top -anchor nw -fill x -expand 1

	iwidgets::entryfield $inf.b.samples -labeltext "Samples:" \
	    -textvariable $this-samples 
	Tooltip $inf.b.samples "Number of samples along each\naxis (and an implicit indicator \nof the dimension of the nrrd).\nThese value should be represented as\n1 or more ints separated by spaces."
	pack $inf.b.samples -side left -anchor nw -padx 3 -pady 3 -fill x -expand 1

	iwidgets::entryfield $inf.b.spacing -labeltext "Spacing:" \
	    -textvariable $this-spacing 
	Tooltip $inf.b.spacing "Spacing between samples on each axis.\nUse nan for any non-spatial axes (e.g.\nspacing between red, green, and blue\nalong axis 0 of interleaved RGB image\ndata). These values should be represented\nas 1 or more doubles separated by spaces."
	pack $inf.b.spacing -side left -anchor nw -padx 3 -pady 3 -fill x -expand 1

	iwidgets::entryfield $inf.labels -labeltext "Axes Labels:" \
	    -textvariable $this-labels 
	Tooltip $inf.labels "Short string labels for each of the axes.\nThese should be represented as 1 or more\nstrings separated by spaces."
	pack $inf.labels -side top -anchor nw -fill x -expand 1 -padx 3 -pady 3

	iwidgets::entryfield $inf.content -labeltext "Content:  " \
	    -textvariable $this-content
	Tooltip $inf.content "Specifies the content string of the nrrd,\nwhich is built upon by many nrrd function\nto record a history of operations. This is\nrepresented as a single string."
	pack $inf.content -side top -anchor nw -fill x -expand 1 -padx 3 -pady 3

	frame $inf.c
	pack $inf.c -side top -anchor nw -fill x -expand 1

	iwidgets::entryfield $inf.c.lines -labeltext "Line Skip:" \
	    -textvariable $this-line_skip
	Tooltip $inf.c.lines "Number of ascii lines to skip before\nreading data. This should be an integer."
	pack $inf.c.lines -side left -anchor nw -padx 3 -pady 3

	iwidgets::entryfield $inf.c.bytes -labeltext "Byte Skip:" \
	    -textvariable $this-byte_skip
	Tooltip $inf.c.bytes "Number of bytes to skip (after skipping\nascii lines, if any) before reading data.\nA value of -1 indicates a binary header of\nunknown length in raw-encoded data. This\nvalue should be represented as an int."
	pack $inf.c.bytes -side left -anchor nw -padx 3 -pady 3

	frame $inf.e
	pack $inf.e -side left -anchor nw -fill x -expand 1

	iwidgets::labeledframe $inf.e.keys \
	    -labelpos nw -labeltext "Key/Value Pairs"
	set kv [$inf.e.keys childsite]

	Tooltip $kv "Key/Value string pairs to be stored\nin the nrrd. Each key must be a\nsingle string and separate more than\none value by spaces."

	pack $inf.e.keys -side top -anchor n

	label $kv.keys -text "Key:" 
	label $kv.vals -text "Values:"
	grid $kv.keys -row 0 -col 0 
	grid $kv.vals -row 0 -col 1

	entry $kv.key1 -textvar $this-key1
	entry $kv.val1 -textvar $this-val1
	grid $kv.key1 -row 1 -col 0
	grid $kv.val1 -row 1 -col 1

	entry $kv.key2 -textvar $this-key2
	entry $kv.val2 -textvar $this-val2
	grid $kv.key2 -row 2 -col 0
	grid $kv.val2 -row 2 -col 1


	entry $kv.key3 -textvar $this-key3
	entry $kv.val3 -textvar $this-val3
	grid $kv.key3 -row 3 -col 0
	grid $kv.val3 -row 3 -col 1
	
	pack $w -fill x -expand yes -side top
    }

    method update_kind {op} {
	set which [$op get]
	set $this-kind $which
    }


    method ui {} {
	set w .ui[modname]

	if {[winfo exists $w]} {
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

	makeSciButtonPanel $w $w $this
	moveToCursor $w
    }
}


