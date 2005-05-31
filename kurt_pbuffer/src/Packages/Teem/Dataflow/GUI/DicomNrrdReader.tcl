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


itcl_class Teem_DataIO_DicomNrrdReader {
    inherit Module
    constructor {config} {
        set name DicomNrrdReader
        set_defaults
    }

    method set_defaults {} {
	global $this-dir
        global $this-series-uid
        global $this-series-files
        global $this-messages
        global $this-suid-sel
        global $this-series-del
        global $this-dir-tmp
	global $this-num-entries
	global $this-max-entries
	global $this-entry-dir
	global $this-entry-suid
	global $this-entry-files
        global $this-num-series
        global $this-num-files

	set $this-dir [pwd]
        set $this-series-uid ""
        set $this-series-files ""
        set $this-messages ""
        set $this-suid-sel ""
        set $this-series-del ""
        set $this-dir-tmp ""
	set $this-num-entries 0
	set $this-max-entries 0
	set $this-entry-dir ""
	set $this-entry-suid ""
	set $this-entry-files ""
        set $this-num-series 0
        set $this-num-files 0
    }

    method ui {} {
	global $this-have-insight
	if {![set $this-have-insight]} {
	    tk_dialog .needinsight "Error: Need Insight" "Error: This module relies upon functionality from the Insight package to read DICOM data; however, you do not have the Insight package enabled.  You can enable the Insight package by installing ITK, re-running configure, and re-compiling.  Please see the SCIRun installation guide for more information." "" 0 "OK"
	    return
	}

        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w  

	frame $w.row10
	frame $w.row11
	frame $w.row8
        frame $w.row4
	frame $w.row3 
	frame $w.row9
	frame $w.which -relief groove -borderwidth 2
       	iwidgets::labeledframe $w.sd -labeltext "Selected Data"
	set sd [$w.sd childsite]
        iwidgets::labeledframe $w.listing -labeltext \
        "Series ID  /  File(s) in Series"
	set listing [$w.listing childsite]

        pack $w.row10 $w.row11 $w.row8 $w.which $w.listing \
        $w.row4 $w.row3 $w.sd $w.row9 -side top -e y -f both -padx 5 -pady 5

        # Directory selection mechanisms

        # Directory "Browse" button
	button $w.row10.browse_button -text " Browse " \
	    -command "$this choose_dir; $this update_series_uids"

        # Directory text box
	label $w.row10.dir_label -text "Directory  " 
	entry $w.row10.dir -textvariable $this-dir -width 80

	pack $w.row10.dir_label $w.row10.dir -side left
	pack $w.row10.browse_button -side right

        # Directory "Load" button
	button $w.row11.load_button -text " Load " -command "$this update_series_uids"

	pack $w.row11.load_button -side right

        # Listboxes for series selection
        set seriesuid [scrolled_listbox $listing.seriesuid -height 10 -selectmode single ]

        set files [scrolled_listbox $listing.files -height 10 -selectmode extended]

        pack $listing.seriesuid $listing.files -side left -fill x -expand yes

        # Populate seriesuid listbox
	update_series_uids

        # Selecting in Series UID listbox causes text to appear in Files 
        # listbox
        bind $seriesuid <ButtonRelease-1> "$this update_series_files %W $files"

        # Text below Series UID and Files listboxes.  This text says how many
        # series' are in the Series ID listbox and how many files are in the
        # Files listbox
        label $w.row4.num_series_label -text "Number of series' in selected directory :  " 
        label $w.row4.num_series -textvariable $this-num-series

        label $w.row4.num_files_label -text "       Number of files in selected series :  " 
        label $w.row4.num_files -textvariable $this-num-files
 
        pack $w.row4.num_series_label $w.row4.num_series $w.row4.num_files_label $w.row4.num_files -side left

        # Add button
        button $w.row3.add -text "Add Data" -command "$this add_data"

        pack $w.row3.add -side top -expand true -fill both

        # Listbox for select series'

	set selected [scrolled_listbox $sd.selected -width 100 -height 10 -selectmode single]

	button $sd.delete -text "Remove Data" \
	    -command "$this delete_data"

	pack $sd.selected $sd.delete -side top -fill x -expand yes
        pack $sd.selected -side top -fill x -expand yes

	makeSciButtonPanel $w $w $this
	moveToCursor $w

	$this sync_filenames
    }


    method choose_dir { } {	
        set w .ui[modname]

	if { [ expr [winfo exists $w] ] }  {

            # Place to put preferred data directory
	    # It's used if $this-file is empty or invalid
            if { [string equal [set $this-dir] ""] || 
                 ![file exists [set $this-dir]] } {
	        set initdir [netedit getenv SCIRUN_DATA]
  	    } else {
                set initdir [set $this-dir]
            }

	    set $this-dir-tmp [ tk_chooseDirectory \
  		          -initialdir $initdir \
                          -parent $w \
                          -title "Choose Directory" \
                          -mustexist true ] 

            if { [string length [set $this-dir-tmp]] > 0 } { 
                set $this-dir [set $this-dir-tmp] 
            }
        }
        
    }

    method update_series_uids { } {
	global $this-dir
	set w .ui[modname]
	
	if [ expr [winfo exists $w] ] {
	    set listing [$w.listing childsite]
	    set seriesuid $listing.seriesuid
	    set files $listing.files
	    
	    # Delete all entries in the series uid list
	    $seriesuid.list delete 0 end
	    $files.list delete 0 end
	    
	    # Call C++ function to set my $this-series-uid variable as a list 
	    # of series uids for the series' in the current directory. If there
	    # are no series' in this dir, it sets the message variable instead
	    $this-c get_series_uid [set $this-dir]
	    
	    set len [string length $this-series-uid]
	    
	    if { [string length [set $this-series-uid]] != 0 } { 
		# Break series_uid into a list 
		set suids [set  $this-series-uid] 
		set list_suid $suids
		
		# Delete the first entry in the list -- this is always empty
		set len [llength $list_suid]
		set list_suid [lrange $list_suid 1 $len]
                set $this-num-series [llength $list_suid] 
		    
		foreach entry $list_suid {
		    $seriesuid.list insert end $entry
		}
		
		# Select first line
		$seriesuid.list selection set 0 0 
		update_series_files $seriesuid.list $files.list
		    
	    } else {
		$seriesuid.list insert end [set $this-messages]
                set $this-num-series 0 
                set $this-num-files 0 
	    }     
	}
    }

    method update_series_files { src dst } {
	global $this-dir
	global $this-suid-sel

        # Delete all entries in the files list
        $dst delete 0 end

        # Get the current cursor selection
        foreach i [$src curselection] {
            set $this-suid-sel [$src get $i] 
        }

        # Call C++ function to set my $this-files variable as a list of
        # files for the seriesuid that was selected.
        $this-c get_series_files [set $this-dir] [set $this-suid-sel]

        # Break series files into a list 
        set fls [set  $this-series-files] 
	set list_files $fls
        set $this-series-files ""
         
        # Delete the last entry in the list -- this is always empty
        set $this-num-files [llength $list_files]

        foreach entry $list_files {
            # Grab the filename of the end
            set names [split $entry "/"]
            set len [llength $names]
            set file_name [lindex $names [expr $len - 1]] 
            set $this-series-files [ concat [set $this-series-files] "\{$file_name\}" ]
            $dst insert end $file_name 
        }
    }

    method add_data { } {
	global $this-dir
	global $this-suid-sel

	# Need to pass the c++ side the current directory, the selected series 
	# uid, and the selected files in the series.  If no files are selected,
	# all files are considered to be selected.
	
	set w .ui[modname]
	
	if [ expr [winfo exists $w]] {

            set listing [$w.listing childsite]
            set files $listing.files
            set sd [$w.sd childsite]
            set selected $sd.selected

            # Check to make sure there are files to add
            if { ![string equal [set $this-series-files] ""] } {
 
                # Get selected files
                set list_files [$files.list curselection]
            
                if { [llength $list_files] > 0 } {
                    set $this-series-files ""
                    foreach i [$files.list curselection] {
                        set $this-series-files [ concat [set $this-series-files] "\{[$files.list get $i]\}" ]
                    }
                }

                # Update $this-series-files to contain only selected files 
                # If no files are selected, it contains all files
                       
                # Get start and end file
                # Break series files into a list 
                set fls [set  $this-series-files] 
                set list_files $fls
 
                set start_file [lindex $list_files 0]
                set end_file [lindex $list_files end]
                set num_files [llength $list_files]

                # Check to make sure this is a unique entry
                set entry "DIR: \"[set $this-dir]\"   SERIES UID: \"[set $this-suid-sel]\"   START FILE: \"$start_file\"   END FILE: \"$end_file\"   NUMBER OF FILES: $num_files"

                set list_sel [$selected.list get 0 end]

                foreach cur_entry $list_sel {
                    if { [string equal $entry $cur_entry] } {
                        # Duplicate entry 
                        return                    
                    }  
                }

		# initialize a filename variable and set it to the
		# current file
		global $this-num-entries
		global $this-max-entries
		global $this-series-files


		# Only make a new variable if the max number of files
		# that have been created is == the num-entries
		if {[set $this-num-entries] == [set $this-max-entries]} {

		    set i [set $this-num-entries]

		    global $this-entry-dir$i
		    global $this-entry-suid$i
		    global $this-entry-files$i

		    set $this-entry-dir$i [set $this-dir]
		    set $this-entry-suid$i [set $this-suid-sel]
		    set $this-entry-files$i ""

		    for {set j 0} {$j < [llength $list_files]} {incr j} {
			set $this-entry-files$i [ concat [set $this-entry-files$i] "\{[lindex $list_files $j]\}" ]
		    }

		    set $this-max-entries [expr [set $this-max-entries] + 1]

		}

		# increment num-entries
		set $this-num-entries [expr [set $this-num-entries] + 1]

                # Call the c++ function that adds this data to its data 
                # structure.
                $this-c add_data [set $this-dir] [set $this-suid-sel] [set $this-series-files]
           
                # Now add entry to selected data
                $selected.list insert end $entry

            }  
        }
    } 

    method add_saved_data { which } {
	global $this-entry-dir$which
	global $this-entry-suid$which
	global $this-entry-files$which

	# Need to pass the c++ side the current directory, the selected series 
	# uid, and the selected files in the series.  If no files are selected,
	# all files are considered to be selected.
	
	set w .ui[modname]
	
	if [ expr [winfo exists $w] ] {
            set sd [$w.sd childsite]
            set selected $sd.selected
	    set list_files [set $this-entry-files$which]

	    set start_file [lindex $list_files 0]
	    set end_file [lindex $list_files end]
            set num_files [llength $list_files]

	    # Check to make sure this is a unique entry
	    set entry "DIR: \"[set $this-entry-dir$which]\"   SERIES UID: \"[set $this-entry-suid$which]\"   START FILE: \"$start_file\"   END FILE: \"$end_file\"   NUMBER OF FILES: $num_files"
	    
	    

	    # increment num-entries
	    set $this-num-entries [expr [set $this-num-entries] + 1]

	    # Call the c++ function that adds this data to its data 
	    # structure.
	    set $this-entry-files$which [$this fix_entry_files [set $this-entry-files$which]]
	    
	    $this-c add_data [set $this-entry-dir$which] [set $this-entry-suid$which] [set $this-entry-files$which]
	    
	    # Now add entry to selected data
	    $selected.list insert end $entry
	    
	}  
    }
    
    method fix_entry_files {files} {
	# The old way of saving entry files was to separate files with spaces.
	# This approach didn't allow for filenames with spaces.  This
	# method inserts curly braces around the filenames so that it
	# can be parsed properly
	
	if {[string index $files 0] == "\{"} {
	    # all ready formatted correctly
	    return $files
	}

	# split by spaces
	set list_files [split $files " "]
	set files ""
        foreach f $list_files {
	    set files [ concat $files "\{$f\}" ]
	}

	return $files
    }

 
    method delete_data { } {
	global $this-num-entries
	set w .ui[modname]

	if [ expr [winfo exists $w] ] {
	    set sd [$w.sd childsite]
	    set selected $sd.selected

            # Get the current cursor selection
            foreach i [$selected.list curselection] {
                set $this-series-del [$selected.list get $i] 
                $selected.list delete $i $i

		# re-order and remove selected file
		for {set x $i} {$x < [expr [set $this-num-entries]-1]} {incr x} {
		    global $this-entry-dir$x
		    global $this-entry-suid$x
		    global $this-entry-files$x
		    set next [expr $x +1]
		    global $this-entry-dir$next
		    global $this-entry-suid$next
		    global $this-entry-files$next
		    set $this-entry-dir$x [set $this-entry-dir$next]
		    set $this-entry-suid$x [set $this-entry-suid$next]
		    set $this-entry-files$x [set $this-entry-files$next]
		}

		# decrement num-entries
		set $this-num-entries [expr [set $this-num-entries] - 1]

                # Call the c++ function that deletes this data from its data 
                # structure.
                $this-c delete_data [set $this-series-del]
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

 	global $this-num-entries

 	if {![winfo exists $w]} {
 	    $this ui
 	    wm withdraw $w
 	}


	# Select proper series
	set sd [$w.sd childsite]
	set selected $sd.selected
	
	# make sure num-entries corresponds to
	# the number of files in the selection box
 	if {[set $this-num-entries] != [$selected.list size]} {
 	    global $this-series-files

 	    # Make sure all filenames are in the
 	    # selected listbox
	    
 	    # delete all of them
	    $w.listing.childsite.seriesuid.list delete 0 end
	    
 	    set num [set $this-num-entries]
 	    set $this-num-entries 0
	    
	    global $this-dir
	    global $this-suid-sel

 	    # add back in
 	    for {set i 0} {$i < $num} {incr i} {
 		$this add_saved_data $i
 	    }

	    # update directory
	    $this update_series_uids
 	} 
    }
}

