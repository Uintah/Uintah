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

# NrrdWriter.tcl
# by Samsonov Alexei
# October 2000 

catch {rename Teem_DataIO_NrrdWriter ""}

itcl_class Teem_DataIO_NrrdWriter {
    inherit Module
    constructor {config} {
	set name NrrdWriter
	set_defaults
    }
    method set_defaults {} {
	global $this-filetype
	set $this-filetype Binary
	# set $this-split 0
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
	set defext ".nd"
	
	# name to appear initially
	set defname "MyNrrd"
	set title "Save nrrd file"

	# file types to appers in filter box
	set types {
	    {{Nrrd Data (and separate .nrrd)}     {.nd}      }
	    {{All Files}       {.*}   }
	}
	
	######################################################
	
	makeSaveFilebox \
		-parent . \
		-filevar $this-filename \
		-command "$this-c needexecute; destroy " \
		-cancel "destroy " \
		-title $title \
		-filetypes $types \
	        -initialfile $defname \
		-initialdir $initdir \
		-defaultextension $defext \
		-formatvar $this-filetype 
		#-splitvar $this-split
    }
}
