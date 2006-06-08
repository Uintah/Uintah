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


# GUI for SliceReader module
# by Darby Van Uitert
# April 2005

catch {rename Insight_DataIO_SliceReader ""}

itcl_class Insight_DataIO_SliceReader {
    inherit Module
    constructor {config} {
	set name SliceReader
	set_defaults
    }

    method set_defaults {} {
	global $this-filename
	global $this-p_type
	global $this-size_0
	global $this-size_1
	global $this-size_2
	global $this-spacing_0
	global $this-spacing_1
	global $this-spacing_2
	global $this-slice
	global $this-cast_output

	set $this-filename ""
	set $this-p_type "Unknown"
	set $this-size_0 0
	set $this-size_1 0
	set $this-size_2 0
	set $this-spacing_0 1.0
	set $this-spacing_1 1.0
	set $this-spacing_2 1.0
	set $this-slice 0
	set $this-cast_output 0
    }

    method ui {} {
	global env
	global $this-filename
	
	set w .ui[modname]
	
	if {[winfo exists $w]} {
	    return
	}

	toplevel $w
	
	frame $w.file -relief groove -borderwidth 2
	label $w.file.l -text "Analyze File:"
	entry $w.file.e -textvariable $this-filename
	button $w.file.b -text "Browse" \
	    -command "$this choose_file"
	pack $w.file.l $w.file.e $w.file.b -side left -padx 3

	frame $w.samples -relief groove -borderwidth 2
	label $w.samples.l1 -text "Samples: ("
	entry $w.samples.x -relief flat -state disabled \
	    -textvariable $this-size_0 -width 4
	label $w.samples.l2 -text ","
	entry $w.samples.y -relief flat -state disabled \
	    -textvariable $this-size_1 -width 4
	label $w.samples.l3 -text ","
	entry $w.samples.z -relief flat -state disabled \
	    -textvariable $this-size_2 -width 4
	label $w.samples.l4 -text ")"
	pack $w.samples.l1 $w.samples.x $w.samples.l2 \
	    $w.samples.y $w.samples.l3 $w.samples.z \
	    $w.samples.l4 -side left -padx 0 
	
	frame $w.slice -relief groove -borderwidth 2
	label $w.slice.l -text "Slice:"
	scale $w.slice.s -variable $this-slice \
	    -from 0 -to 1000 -width 15 \
	    -showvalue false -length 150 \
	    -orient horizontal
	entry $w.slice.e -textvariable $this-slice -width 4
	button $w.slice.b1 -text "Read Slice" \
	    -command "$this-c needexecute"
	button $w.slice.b2 -text "Read Next Slice" \
	    -command "$this read_next_slice"
	pack $w.slice.l $w.slice.s $w.slice.e $w.slice.b1 \
	    $w.slice.b2 -side left -padx 2
	
	checkbutton $w.cast -text "Cast output to float" \
	    -variable $this-cast_output

	pack $w.file $w.samples $w.slice $w.cast -side top \
	    -anchor nw -pady 3

	makeSciButtonPanel $w $w $this
	moveToCursor $w	
    }

    method configure_slice_slider {max} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    $w.slice.s configure -from 0 -to $max
	}
    }

    method read_next_slice {} {
	global $this-slice
	set $this-slice [expr [set $this-slice] + 1]
	$this-c needexecute
    }

    method choose_file {} {
	set w .ui[modname]fb

	if {[winfo exists $w]} {
	    SciRaise $w
	    return $w
	}

	toplevel $w -class TkFDialog
	
	# place to put preferred data directory
	# it's used if $this-filename is empty
	set initdir [netedit getenv SCIRUN_DATA]
	
	#######################################################
	# to be modified for particular reader
	
	# extansion to append if no extension supplied by user
	set defext ".hdr"
	set title "Open Analyze File"
	
	# file types to appers in filter box
	set types {
            {{Analyze Files}         {.hdr .img}   }
	    {{All Files}          {.*}            }
	}
	
	######################################################
	
	makeOpenFilebox \
	    -parent $w \
	    -filevar $this-filename \
	    -setcmd "wm withdraw $w" \
	    -command "$this-c needexecute; wm withdraw $w" \
	    -cancel "wm withdraw $w" \
	    -title $title \
	    -filetypes $types \
	    -initialdir $initdir \
	    -defaultextension $defext

	moveToCursor $w
	wm deiconify $w

	return $w
    }
}
