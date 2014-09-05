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

itcl_class Insight_DataIO_ColorImageReaderUChar3D {
    inherit Module
    constructor {config} {
        set name ColorImageReaderUChar3D
        set_defaults
    }

    method set_defaults {} {
    }

    method ui {} {
	global env

        set w .ui[modname]
        if {[winfo exists $w]} {
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

	set defext ".mhd"
	set title "Open image file"
	
	# file types to appers in filter box
	set types {
	    {{Meta Image}        {.mhd .mha} }
	    {{All Files}       {.*}    }
	}

	makeOpenFilebox \
		-parent $w \
		-filevar $this-FileName \
		-command "$this-c needexecute; wm withdraw $w" \
		-cancel "wm withdraw $w" \
		-title "Open Image File" \
                -filetypes $types \
		-initialdir $initdir \
		-defaultextension $defext
    }
}


