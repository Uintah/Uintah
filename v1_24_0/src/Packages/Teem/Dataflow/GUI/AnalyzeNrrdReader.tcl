#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  License for the specific language governing rights and limitations under
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
#


itcl_class Teem_DataIO_AnalyzeNrrdReader {
    inherit Module
    constructor {config} {
        set name AnalyzeNrrdReader
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
	    set parent .
	    if { [winfo exists .standalone] } {
		set parent .standalone
	    }

	    tk_messageBox -type ok -icon info -parent $parent \
		-title "Error: Need Insight" \
		-message "Error: Need Insight" "Error: This module relies upon functionality from the Insight package to read Analyze data; however, you do not have the Insight package enabled.  You can enable the Insight package by installing ITK, re-running configure, and re-compiling.  Please see the SCIRun installation guide for more information."
	    return
	}

        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w

        frame $w.row11
	frame $w.row10
	frame $w.row8
	frame $w.row4
	frame $w.row9
	frame $w.which -relief groove -borderwidth 2
       	iwidgets::labeledframe $w.sd -labeltext "Selected Data"
	set sd [$w.sd childsite]

        pack $w.row11 $w.row8 $w.row10 $w.which \
        $w.sd $w.row4 $w.row9 -side top -e y -f both -padx 5 -pady 5

        # File selection mechanisms

        # File text box
	label $w.row11.file_label -text "File  " 
	entry $w.row11.file -textvariable $this-file -width 80

	pack $w.row11.file_label $w.row11.file -side left

        # File "Browse" button
	button $w.row11.browse_button -text " Browse " \
	    -command "$this choose_file"

	pack $w.row11.browse_button -side right

        # Add selected file
	button $w.row10.browse_button -text "Add Data" \
	    -command "$this add_data"

	pack $w.row10.browse_button -side right -fill x -expand yes

	set selected [scrolled_listbox $sd.selected -width 100 -height 10 -selectmode single]
	button $sd.delete -text "Remove Data" -command "$this delete_data"

	pack $sd.selected $sd.delete -side top -fill x -expand yes -padx 4 -pady 4
        pack $sd.selected -side top -fill x -expand yes

	makeSciButtonPanel $w $w $this
	moveToCursor $w

	sync_filenames
    }


    method choose_file { } {
        #set w .ui[modname]
	set w [format "%s-fb" .ui[modname]]

	set defext ".hdr"

	# place to put preferred data directory
	# it's used if $this-file is empty
	set initdir [netedit getenv SCIRUN_DATA]
	
	# File types to appers in filter box
	set types {
	    {{Analyze Header File}        {.hdr} }
	}
	
	if { [winfo exists $w] } {
	    if { [winfo ismapped $w] == 1} {
		raise $w
	    } else {
		wm deiconify $w
	    }

            # The open file box has already been created.  Update the 
            # configuration so that the default file will be updated to be the
            # current value of $this-file, or $initdir if $this-file is empty.
            
            biopseFDialog_Config $w open \
                [list -parent $w \
	              -filevar $this-file \
	              -cancel "wm withdraw $w" \
	              -title "Open Analyze File" \
	              -filetypes $types \
	              -initialdir $initdir \
	              -defaultextension $defext \
	              -command "wm withdraw $w"]

	    return
	}
	
	#toplevel $w
	toplevel $w -class TkFDialog

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
	    -command "wm withdraw $w" \
	    -commandname Read

	moveToCursor $w
	wm deiconify $w	

    }

    method add_data { } {
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
                    #puts "(add_data) cur_entry = {$cur_entry}"

                    if { [string equal [set $this-file] $cur_entry] } {
                        # Duplicate entry 
                        #puts "duplicate entry"
                        return                    
                    }  
                }

                # Check to make sure this file exists, has a .hdr extension,
                # and the corresponding .img file exists as well
                  
                set ext [file extension [set $this-file]]
                set img_file [file rootname [set $this-file]]
                set img_ext ".img"
                set img_file $img_file$img_ext

                if { ![string equal $ext ".hdr"] } {
                    set answer [tk_messageBox -message \
		    "'$ext' not valid Analyze header (.hdr) extension, please choose a valid header (.hdr) file." \
                    -type ok -icon info -parent $w]
                    return
                }
                if { ![file exists [set $this-file]] } {
                    set answer [tk_messageBox -message \
		    "Analyze header file [set $this-file] does not exist, please choose a valid header (.hdr) file." \
                    -type ok -icon info -parent $w]
	            return
                } elseif { ![file exists $img_file] } {
                    set answer [tk_messageBox -message \
                    "Analyze image file $img_file does not exist, please choose a different header (.hdr) file or copy the corresponding .img file to the same directory as the .hdr file" \
                    -type ok -icon info -parent $w]
                    return
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

    method delete_data { } {
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
                $this-c delete_data [set $this-file-del]

            }

	}
    }

    # Copied from Chapter 30 of Practical Programming in Tcl and Tk
    # by Brent B. Welch.Copyright 2000 Pentice Hall. 

    method scroll_set {scrollbar geoCmd offset size} {
	if {$offset != 0.0 || $size != 1.0} {
	    eval $geoCmd ;# Make sure it is visible
	}
	$scrollbar set $offset $size
    }

    method scrolled_listbox { f args } {
	frame $f
	listbox $f.list \
		-xscrollcommand [list $this scroll_set $f.xscroll \
			[list grid $f.xscroll -row 1 -column 0 -sticky we]] \
		-yscrollcommand [list $this scroll_set $f.yscroll \
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
# 		    $this add_data
# 		}
		global $this-filenames$i
		set $this-file [set $this-filenames$i]
		$this add_data
		
	    }
	} 
    }
}