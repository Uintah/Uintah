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

itcl_class Insight_DataIO_ImageFileWriter {
    inherit Module
    constructor {config} {
        set name ImageFileWriter
        set_defaults
    }

    method set_defaults {} {
	global $this-filetype
	set $this-filetype Binary
	set $this-split 0
    }

    method ui {} {
        set w .ui[modname]
	if {[winfo exists $w]} {
	    set child [lindex [winfo children $w] 0]
	    
	    # $w withdrawn by $child's procedures
	    raise $child
	    return;
        }

	############
	set types {
	    {{Meta Image}        {.mhd} }
	    {{PNG Image}        {.png} }
	    {{All Files}       {.*} }
	}
	set defname "MyImage.mhd"
	set defext ".mhd"
	############
        #toplevel $w
	makeSaveFilebox \
	    -parent . \
	    -filevar $this-FileName \
	    -command "$this-c needexecute; destroy " \
	    -cancel "destroy " \
	    -title "Save Image File" \
	    -filetypes $types \
	    -initialfile $defname \
	    -defaultextension $defext \
	    -formatvar $this-filetype \
	    -splitvar $this-split
    }
}


