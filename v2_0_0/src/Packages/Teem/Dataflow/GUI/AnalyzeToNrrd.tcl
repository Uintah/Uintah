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
	global $this-file
        global $this-file-del
        global $this-messages
	set $this-file ""
        set $this-file-del ""
        set $this-messages ""
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }
        toplevel $w

	frame $w.row10
	frame $w.row8
	frame $w.row4
	frame $w.row9
	frame $w.which -relief groove -borderwidth 2
       	iwidgets::labeledframe $w.sd -labeltext "Selected Data"
	set sd [$w.sd childsite]

        pack $w.row8 $w.row10 $w.which \
        $w.sd $w.row4 $w.row9 -side top -e y -f both -padx 5 -pady 5

	button $w.row10.browse_button -text "Open Analyze File" -command \
	    "$this ChooseFile; \ 
             $this AddData"
       

	pack $w.row10.browse_button -side right -fill x -expand yes

	set selected [Scrolled_Listbox $sd.selected -width 100 -height 10 -selectmode single]

	button $sd.delete -text "Remove Data" \
	    -command "$this DeleteData"

	pack $sd.selected $sd.delete -side top -fill x -expand yes
        pack $sd.selected -side top -fill x -expand yes

	button $w.row4.execute -text "Execute" -command "$this-c needexecute"
	pack $w.row4.execute -side top -e n -f both

    }


    method ChooseFile { } {
        set w .ui[modname]

	if [ expr [winfo exists $w] ] {

            set defext ".hdr"
	
	    # File types to appers in filter box
	    set types {
	        {{Analyze Header File}        {.hdr} }
	    }

	    set $this-file [tk_getOpenFile  \
		-parent $w \
		-title "Open Analyze File" \
                -filetypes $types \
                -defaultextension $defext]
        }
    }


    method AddData { } {
    
        set w .ui[modname]
  
	if [ expr [winfo exists $w] ] {
            set sd [$w.sd childsite]
            set selected $sd.selected

            # Check to make sure there are files to add
            if { ![string equal [set $this-file] ""] } {
 
                # Check to make sure this is a unique entry

                set list_sel [$selected.list get 0 end]

                foreach cur_entry $list_sel {
                    #puts "entry = {$entry}"
                    #puts "cur_entry = {$cur_entry}"

                    if { [string equal [set $this-file] $cur_entry] } {
                        # Duplicate entry 
                        #puts "duplicate entry"
                        return                    
                    }  
                }

                # Call the c++ function that adds this data to its data 
                # structure.
                $this-c add_data
           
                # Now add entry to selected data
                $selected.list insert end [set $this-file]

            }  
        }
    } 

    method DeleteData { } {
	set w .ui[modname]

	if [ expr [winfo exists $w] ] {
	    set sd [$w.sd childsite]
	    set selected $sd.selected

            # Get the current cursor selection
            foreach i [$selected.list curselection] {
                set $this-file-del [$selected.list get $i] 
                $selected.list delete $i $i

                # Call the c++ function that deletes this data from its data 
                # structure.
                $this-c delete_data
            }

	}
    }

    # Copied from Chapter 30 of Practical Programming in Tcl and Tk
    # by Brent B. Welch.Copyright 2000 Pentice Hall. 

    method Scroll_Set {scrollbar geoCmd offset size} {
	if {$offset != 0.0 || $size != 1.0} {
	    eval $geoCmd ;# Make sure it is visible
	}
	$scrollbar set $offset $size
    }

    method Scrolled_Listbox { f args } {
	frame $f
	listbox $f.list \
		-xscrollcommand [list $this Scroll_Set $f.xscroll \
			[list grid $f.xscroll -row 1 -column 0 -sticky we]] \
		-yscrollcommand [list $this Scroll_Set $f.yscroll \
			[list grid $f.yscroll -row 0 -column 1 -sticky ns]]
	eval {$f.list configure} $args
	scrollbar $f.xscroll -orient horizontal \
		-command [list $f.list xview]
	scrollbar $f.yscroll -orient vertical \
		-command [list $f.list yview]
	grid $f.list -sticky news
	grid $f.xscroll -sticky news
	grid rowconfigure $f 0 -weight 1
	grid columnconfigure $f 0 -weight 1
	return $f.list
    }  

    method ListTransferSel { src dst } {
        foreach i [$src curselection] {
            $dst insert end [$src get $i]
        }

    }
}