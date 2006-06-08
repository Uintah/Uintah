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

itcl_class DDDAS_DataIO_NetConnector {
    inherit Module
    constructor {config} {
        set name NetConnector
        set_defaults
    }

    method set_defaults {} {
	global $this-cliserv
        global $this-stop
        global $this-test
	set $this-cliserv "Server"
	set $this-stop 0
        set $this-test 1
    }

    method ui {} {

        set_defaults

        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }
        toplevel $w

	frame $w.row5
	frame $w.row6
	frame $w.row4
	frame $w.row9
	frame $w.which -relief groove -borderwidth 2

        pack $w.row5 $w.row6 $w.which $w.row4 $w.row9 \
             -side top -e y -f both -padx 5 -pady 5

        # Client/Server selection 	
	radiobutton $w.row5.client_radiobutton -text "Client" \
		-variable $this-cliserv -value "Client" 
	radiobutton $w.row5.server_radiobutton -text "Server" \
		-variable $this-cliserv -value "Server" 

        # Test selection
	label $w.row6.test_label -text "Test number  " 
	entry $w.row6.test -textvariable $this-test -width 5

	pack $w.row5.client_radiobutton $w.row5.server_radiobutton -side left
	pack $w.row6.test_label $w.row6.test -side left

	button $w.row4.execute -text "Execute" -command "set $this-stop 0; $this-c needexecute"
	pack $w.row4.execute -side top -e n -f both

	button $w.row9.stop_button -text "Stop" -command "set $this-stop 1"
	entry $w.row9.stop -textvariable $this-stop
	pack $w.row9.stop_button -side top -e n -f both

    }
}


