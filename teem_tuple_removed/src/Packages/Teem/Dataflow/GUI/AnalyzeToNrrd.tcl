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
	global $this-num-files
	global $this-max-files
	global $this-filenames
	set $this-file ""
        set $this-file-del ""
        set $this-messages ""
	set $this-num-files 0
	set $this-max-files 0
	set $this-filenames ""
    }

    method ui {} {
	global $this-have-insight
	if {![set $this-have-insight]} {
	    tk_dialog .needinsight "Error: Need Insight" "Error: This module relies upon functionality from the Insight package to read Analyze data; however, you do not have the Insight package enabled.  You can enable the Insight package by installing ITK, re-running configure, and re-compiling.  Please see the SCIRun installation guide for more information." "" 0 "OK"
	    return
	}

        set w .ui[modname]
        if {[winfo exists $w]} {
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
        $w.sd $w.row4 $w.row9 -side top -e y -f both -padx 5 -pady 2

	button $w.row10.browse_button -text "Open Analyze File" \
	    -command "$this ChooseFile"

	pack $w.row10.browse_button -side right -fill x -expand yes

	set selected [Scrolled_Listbox $sd.selected -width 100 -height 10 -selectmode single]
	button $sd.delete -text "Remove Data" -command "$this DeleteData"

	pack $sd.selected $sd.delete -side top -fill x -expand yes -padx 4 -pady 4
        pack $sd.selected -side top -fill x -expand yes

	makeSciButtonPanel $w $w $this
	moveToCursor $w

	sync_filenames
    }


    method ChooseFile { } {
	global env
        #set w .ui[modname]
	set w [format "%s-fb" .ui[modname]]

	if { [winfo exists $w] } {
	    if { [winfo ismapped $w] == 1} {
		raise $w
	    } else {
		wm deiconify $w
	    }
	    return
	}
	
	#toplevel $w
	toplevel $w -class TkFDialog

	set defext ".hdr"

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
	
	# File types to appers in filter box
	set types {
	    {{Analyze Header File}        {.hdr} }
	}
	
# 	set $this-file [tk_getOpenFile  \
# 			    -parent $w \
# 			    -title "Open Analyze File" \
# 			    -filetypes $types \
# 			    -defaultextension $defext]
	
	makeOpenFilebox \
	    -parent $w \
	    -filevar $this-file \
	    -cancel "wm withdraw $w" \
	    -title "Open Analyze File" \
	    -filetypes $types \
	    -initialdir $initdir \
	    -defaultextension $defext \
	    -command "wm withdraw $w; $this AddData" \

	moveToCursor $w
	wm deiconify $w	
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

		# initialize a filename variable and set it to the
		# current file
		global $this-num-files
		global $this-max-files
		# Only make a new variable if the max number of files
		# that have been created is == the num-files
		if {[set $this-num-files] == [set $this-max-files]} {
		    global $this-filenames[set $this-num-files]
		    set $this-filenames[set $this-num-files] [set $this-file]

		    set $this-max-files [expr [set $this-max-files] + 1]
		}

		# increment num-files
		set $this-num-files [expr [set $this-num-files] + 1]
		

                # Call the c++ function that adds this data to its data 
                # structure.
		global $this-file
                $this-c add_data [set $this-file]
           
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

	    global $this-num-files
            # Get the current cursor selection
            foreach i [$selected.list curselection] {
                set $this-file-del [$selected.list get $i] 
                $selected.list delete $i $i

		# re-order and remove selected file
		for {set x $i} {$x < [expr [set $this-num-files]-1]} {incr x} {
		    global $this-filenames$x
		    set next [expr $x +1]
		    global $this-filenames$next
		    set $this-filenames$x [set $this-filenames$next]
		}

		# decrement num-files
		set $this-num-files [expr [set $this-num-files] - 1]

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

    method sync_filenames {} {
	set w .ui[modname]

	global $this-num-files

	if {![winfo exists $w.sd]} {
	    $this ui
	    wm withdraw $w
	}

	set sd [$w.sd childsite]
	set selected $sd.selected
	
	# make sure num-files corresponds to
	# the number of files in the selection box
	if {[set $this-num-files] != [$selected.list size]} {
	    global $this-file

	    # Make sure all filenames are in the
	    # selected listbox
	    
	    # delete all of them
	    $selected.list delete 0 end
	    $this-c clear_data
	    
	    set num [set $this-num-files]
	    set $this-num-files 0
	    
	    # add back in
	    for {set i 0} {$i < $num} {incr i} {
# 		if {[info exists $this-filenames$i]} {
# 		    # Creat them
# 		    set temp [set $this-filenames$i]
# 		    unset $this-filenames$i
# 		    set $this-file $temp
# 		    global $this-filenames$i
# 		    set $this-filenames$i [set $this-file]
# 		    $this AddData
# 		}
		global $this-filenames$i
		set $this-file [set $this-filenames$i]
		$this AddData
		
	    }
	} 
    }
}