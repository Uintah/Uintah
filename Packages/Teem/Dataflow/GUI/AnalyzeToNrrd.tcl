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

itcl_class Teem_DataIO_AnalyzeToNrrd {
    inherit Module
    constructor {config} {
        set name AnalyzeToNrrd
        set_defaults
    }

    method set_defaults {} {
	global $this-prefix
	global $this-port
	global $this-end-index
        global $this-browse
	set $this-dir [pwd]
	set $this-dir-tmp ""
	set $this-prefix ""
	set $this-start-index 0
	set $this-end-index 0
	set $this-browse 0
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }
        toplevel $w

	frame $w.row10
	frame $w.row5
	frame $w.row6
	frame $w.row7
	frame $w.row8
	frame $w.row4
	frame $w.row3 
	frame $w.row9
	frame $w.which -relief groove -borderwidth 2


        pack $w.row10 $w.row5 $w.row6 $w.row7 $w.row8 $w.which $w.row3 \
        $w.row4 $w.row9 -side top -e y -f both -padx 5 -pady 5

	button $w.row10.browse_button -text "Browse" -command \
	"set $this-dir \[ tk_chooseDirectory \
                          -parent $w \
                          -title \"Choose Directory\" \
                          -mustexist true \] "
       

	entry $w.row10.browse -textvariable $this-browse
	
	label $w.row10.dir_label -text "Directory  "
	entry $w.row10.dir -textvariable $this-dir
	label $w.row5.prefix_label -text "File Prefix (optional)  "
	entry $w.row5.prefix -textvariable $this-prefix
	label $w.row6.start-index_label -text "Start Index Number   "
	entry $w.row6.start-index -textvariable $this-start-index
	label $w.row7.end-index_label -text "End Index Number   "
	entry $w.row7.end-index -textvariable $this-end-index

	pack $w.row10.dir_label $w.row10.dir -side left
	pack $w.row10.browse_button -side right
	pack $w.row5.prefix_label $w.row5.prefix -side left
	pack $w.row6.start-index_label $w.row6.start-index -side left
	pack $w.row7.end-index_label $w.row7.end-index -side left	

        button $w.row3.add-series -text "Add Series" -command ""
	pack $w.row3.add-series -side top -e n -f both

	button $w.row4.execute -text "Execute" -command "$this-c needexecute"
	pack $w.row4.execute -side top -e n -f both

    }

}


