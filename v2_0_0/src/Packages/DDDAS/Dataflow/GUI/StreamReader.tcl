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

itcl_class DDDAS_DataIO_StreamReader {
    inherit Module
    constructor {config} {
        set name StreamReader
        set_defaults
    }

    method set_defaults {} {
	global $this-hostname
	global $this-port
	global $this-file-read
	global $this-file-write
        global $this-stop-sr
	set $this-hostname "arthur.ccs.uky.edu"
	set $this-port 8000
	set $this-file-read "test_file.mp3"
	set $this-file-write "sample.txt"
	set $this-stop-sr 0
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }
        toplevel $w

	frame $w.row5
	frame $w.row6
	frame $w.row7
	frame $w.row8
	frame $w.row4
	frame $w.row9
	frame $w.which -relief groove -borderwidth 2

        pack $w.row5 $w.row6 $w.row7 $w.row8  $w.which $w.row4 $w.row9 \
             -side top -e y -f both -padx 5 -pady 5
	
	label $w.row5.hostname_label -text "Hostname   "
	entry $w.row5.hostname -textvariable $this-hostname
	label $w.row6.port_label -text "Port Number   "
	entry $w.row6.port -textvariable $this-port
	label $w.row7.file-read_label -text "File to Read   "
	entry $w.row7.file-read -textvariable $this-file-read
	label $w.row8.file-write_label -text "Save As   "
	entry $w.row8.file-write -textvariable $this-file-write

	pack $w.row5.hostname_label $w.row5.hostname -side left
	pack $w.row6.port_label $w.row6.port -side left
	pack $w.row7.file-read_label $w.row7.file-read -side left	
        pack $w.row8.file-write_label $w.row8.file-write -side left

	button $w.row4.execute -text "Execute" -command "set $this-stop-sr 0; $this-c needexecute"
	pack $w.row4.execute -side top -e n -f both

	button $w.row9.stop-sr_button -text "Stop" -command "set $this-stop-sr 1"
	entry $w.row9.stop-sr -textvariable $this-stop-sr
	pack $w.row9.stop-sr_button -side top -e n -f both

    }
}


