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

# GUI for Fusion_DataIO_MDSPlusDataReader module
# by Allen R. Sanderson
# March 2002

itcl_class Fusion_DataIO_MDSPlusDataReader {
    inherit Module
    constructor {config} {
        set name MDSPlusDataReader
        set_defaults
    }

    method set_defaults {} {
	global $this-serverName
	global $this-treeName
	global $this-shotNumber

	set $this-serverName "atlas.gat.com"
	set $this-treeName "NIMROD"
	set $this-shotNumber "10089"
    }

    method ui {} {
	global $this-serverName
	global $this-treeName
	global $this-shotNumber

        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }

        toplevel $w

	labelEntry $w.server "Server" $this-serverName
	labelEntry $w.tree   "Tree"   $this-treeName
	labelEntry $w.shot   "Shot"   $this-shotNumber

	button $w.button -text "Download" -command "$this-c needexecute"

	pack $w.button -padx 10 -pady 10
    }

    method labelEntry { win text1 text2 } {
	frame $win 
	pack $win -side top -padx 5 -pady 5
	label $win.l -text $text1  -width 6 -anchor w -just left
	label $win.c  -text ":" -width 1 -anchor w -just left 
	entry $win.e -text $text2 -width 20 -just left -fore darkred
	pack $win.l $win.c -side left
	pack $win.e -padx 5 -side left
    }
}







