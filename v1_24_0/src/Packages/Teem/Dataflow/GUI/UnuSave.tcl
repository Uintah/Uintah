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

#    File   : UnuSave.tcl
#    Author : Darby Van Uitert
#    Date   : April 2004

itcl_class Teem_UnuNtoZ_UnuSave {
    inherit Module
    constructor {config} {
        set name UnuSave
        set_defaults
    }

    method set_defaults {} {
	global $this-format
	set $this-format "nrrd"

	global $this-encoding
	set $this-encoding "raw"

	global $this-endian
	set $this-endian "little"

	global $this-filename
	set $this-filename ""

	global $this-filetype
	set $this-filetype Binary
    }

    method create_filebox {} {
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
	set defext ".nrrd"
	
	# name to appear initially
	set defname "MyNrrd"
	set title "Unu Save"

	# file types to appers in filter box
	set types {
	    {{Nrrd}     {.nrrd}      }
	    {{pnm}     {.pnm .ppm .pgm}      }
	    {{Text}     {.txt}      }
	    {{PNG}     {.png}      }
	    {{VTK}     {.vtk}      }
	    {{EPS}     {.eps}      }
	    {{All Files}       {.*}   }
	}
	
	######################################################
	
	makeSaveFilebox \
		-parent $w \
		-filevar $this-filename \
		-command "wm withdraw $w" \
	        -commandname Save \
		-cancel "wm withdraw $w" \
		-title $title \
		-filetypes $types \
	        -initialfile $defname \
		-initialdir $initdir \
		-defaultextension $defext \
		-formatvar $this-filetype 
		#-splitvar $this-split 

	moveToCursor $w
	wm deiconify $w	
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

        iwidgets::entryfield $w.f.options.format -labeltext "Format:" \
	    -textvariable $this-format
        pack $w.f.options.format -side top -expand yes -fill x

        iwidgets::entryfield $w.f.options.encoding -labeltext "Encoding:" \
	    -textvariable $this-encoding
        pack $w.f.options.encoding -side top -expand yes -fill x

        iwidgets::entryfield $w.f.options.endian -labeltext "Endian:" \
	    -textvariable $this-endian
        pack $w.f.options.endian -side top -expand yes -fill x

	iwidgets::entryfield $w.f.options.filename -labeltext "Filename:" \
	    -textvariable $this-filename
        pack $w.f.options.filename -side top -expand yes -fill x

	button $w.f.options.filenameb -text "Browse" \
	    -command "$this create_filebox"  
	pack $w.f.options.filenameb -side top -expand yes -fill x

	makeSciButtonPanel $w.f $w $this
	moveToCursor $w

	pack $w.f -expand 1 -fill x
    }
}


