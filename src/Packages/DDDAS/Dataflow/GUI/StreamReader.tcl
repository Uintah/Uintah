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
	global $this-brokerip
	global $this-brokerport
	global $this-groupname
	global $this-listenport
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
	frame $w.row7
	frame $w.row8
	frame $w.row4
	frame $w.row9
	frame $w.which -relief groove -borderwidth 2

        pack $w.row5 $w.row6 $w.row7 $w.row8  $w.which $w.row4 $w.row9 \
             -side top -e y -f both -padx 5 -pady 5
	
	label $w.row5.brokerip_label -text "Broker IP:"
	entry $w.row5.brokerip -textvariable $this-brokerip
	label $w.row6.brokerport_label -text "Broker Port:"
	entry $w.row6.brokerport -textvariable $this-brokerport
	label $w.row7.groupname_label -text "Group Name:"
	entry $w.row7.groupname -textvariable $this-groupname
	label $w.row8.listenport_label -text "Listening Port:"
	entry $w.row8.listenport -textvariable $this-listenport

	pack $w.row5.brokerip_label $w.row5.brokerip -side left
	pack $w.row6.brokerport_label $w.row6.brokerport -side left
	pack $w.row7.groupname_label $w.row7.groupname -side left	
        pack $w.row8.listenport_label $w.row8.listenport -side left

	button $w.row4.execute -text "Execute" -command "$this-c needexecute"
	pack $w.row4.execute -side top -e n -f both

    }
}


