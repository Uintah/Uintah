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

itcl_class Teem_DataIO_DicomToNrrd {
    inherit Module
    constructor {config} {
        set name DicomToNrrd
        set_defaults
    }

    method set_defaults {} {
	global $this-dir
	#global $this-start-index
	#global $this-end-index
        #global $this-browse
        global $this-series-uid
        global $this-series-files
        global $this-messages
        global $this-suid-sel
        global $this-series-del
        global $this-dir-tmp
	set $this-dir [pwd]
	#set $this-start-index 0
	#set $this-end-index 0
	#set $this-browse 0
        set $this-series-uid ""
        set $this-series-files ""
        set $this-messages ""
        set $this-suid-sel ""
        set $this-series-del ""
        set $this-dir-tmp ""
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
	frame $w.row3 
	frame $w.row9
	frame $w.which -relief groove -borderwidth 2
       	iwidgets::labeledframe $w.sd -labeltext "Selected Data"
	set sd [$w.sd childsite]
        iwidgets::labeledframe $w.listing -labeltext \
        "Series ID  /  File(s) in Series"
	set listing [$w.listing childsite]

        pack $w.row10 $w.row8 $w.which $w.listing \
        $w.row3 $w.sd $w.row4 $w.row9 -side top -e y -f both -padx 5 -pady 5

	button $w.row10.browse_button -text "Browse" -command \
	    "$this ChooseDir;\ 
             $this UpdateSeriesUIDs"
       

	#entry $w.row10.browse -textvariable $this-browse
	
	label $w.row10.dir_label -text "Directory  " 
	entry $w.row10.dir -textvariable $this-dir -width 80

	pack $w.row10.dir_label $w.row10.dir -side left
	pack $w.row10.browse_button -side right

        # Listboxes for series selection
        set seriesuid [Scrolled_Listbox $listing.seriesuid -height 10 -selectmode single ]

        set files [Scrolled_Listbox $listing.files -height 10 -selectmode extended]

        pack $listing.seriesuid $listing.files -side left -fill x -expand yes

        # Populate seriesuid listbox
        UpdateSeriesUIDs

        # Selecting in Series UID listbox causes text to appear in Files 
        # listbox
        bind $seriesuid <ButtonRelease-1> "$this UpdateSeriesFiles %W $files"

        # Add button
        button $w.row3.add -text "Add Data" -command "$this AddData"

        pack $w.row3.add -side top -expand true -fill both

        # Listbox for select series'

	set selected [Scrolled_Listbox $sd.selected -width 100 -height 10 -selectmode single]

	button $sd.delete -text "Remove Data" \
	    -command "$this DeleteData"

	pack $sd.selected $sd.delete -side top -fill x -expand yes
        pack $sd.selected -side top -fill x -expand yes

	button $w.row4.execute -text "Execute" -command "$this-c needexecute"
	pack $w.row4.execute -side top -e n -f both

    }


    method ChooseDir { } {
        set w .ui[modname]

	if [ expr [winfo exists $w] ] {
	    set $this-dir-tmp [ tk_chooseDirectory \
                          -parent $w \
                          -title "Choose Directory" \
                          -mustexist true ] 

            if { [string length [set $this-dir-tmp]] > 0 } { 
                set $this-dir [set $this-dir-tmp] 
            }
        }
    }

    method UpdateSeriesUIDs { } {
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
            $this-c get_series_uid

            set len [string length $this-series-uid]
            #puts "len = $len"            
            #puts "this-series-uid = {[set $this-series-uid]}"

            if { [string length [set $this-series-uid]] != 0 } { 
                # Break series_uid into a list 
                set suids [set  $this-series-uid] 
                #puts "suids = {$suids}"
                set list_suid [split $suids " "]
 
                # Delete the first entry in the list -- this is always empty
                set len [llength $list_suid]
                set list_suid [lrange $list_suid 1 $len]

                foreach entry $list_suid {
                  $seriesuid.list insert end $entry
                }
                
                # Select first line
                $seriesuid.list selection set 0 0 
                UpdateSeriesFiles $seriesuid.list $files.list

	    } else {
                $seriesuid.list insert end [set $this-messages]
            }     
        }
    }

    method UpdateSeriesFiles { src dst } {

        # Delete all entries in the files list
        $dst delete 0 end

        # Get the current cursor selection
        foreach i [$src curselection] {
            set $this-suid-sel [$src get $i] 
        }

        #puts "this-suid-sel = {[set $this-suid-sel]}"

        # Call C++ function to set my $this-files variable as a list of
        # files for the seriesuid that was selected.
        $this-c get_series_files 

        # Break series files into a list 
        set fls [set  $this-series-files] 
        #puts "fls = {$fls}"
        set list_files [split $fls " "]
        set $this-series-files ""
         
        # Delete the first entry in the list -- this is always empty
        set len [llength $list_files]
        set list_files [lrange $list_files 1 $len]

        foreach entry $list_files {
            # Grab the filename of the end
            set names [split $entry "/"]
            set len [llength $names]
            set file_name [lindex $names [expr $len - 1]] 
            set $this-series-files [ concat [set $this-series-files] $file_name ]
            $dst insert end $file_name 
        }
    }

    method AddData { } {
      # Need to pass the c++ side the current directory, the selected series 
      # uid, and the selected files in the series.  If no files are selected,
      # all files are considered to be selected.
    
      set w .ui[modname]

	if [ expr [winfo exists $w] ] {
            set listing [$w.listing childsite]
            #set seriesuid $listing.seriesuid
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
                        set $this-series-files [ concat [set $this-series-files] [$files.list get $i] ]
                    }

                    #puts "this-series-files = [set $this-series-files]"
                }

                # Update $this-series-files to contain only selected files 
                # If no files are selected, it contains all files
                       
                # Get start and end file
                # Break series files into a list 
                set fls [set  $this-series-files] 
                set list_files [split $fls " "]
 
                # Delete the first entry in the list -- this is always empty
                #set len [llength $list_files]
                #set list_files [lrange $list_files 1 $len]

                set start_file [lindex $list_files 0]
                set end_file [lindex $list_files end]
 
                # Check to make sure this is a unique entry
                set entry "DIR: [set $this-dir]   SERIES UID: [set $this-suid-sel]   START FILE: $start_file   END FILE: $end_file"

                set list_sel [$selected.list get 0 end]

                foreach cur_entry $list_sel {
                    #puts "entry = {$entry}"
                    #puts "cur_entry = {$cur_entry}"

                    if { [string equal $entry $cur_entry] } {
                        # Duplicate entry 
                        #puts "duplicate entry"
                        return                    
                    }  
                }

                # Call the c++ function that adds this data to its data 
                # structure.
                $this-c add_data
           
                # Now add entry to selected data
                $selected.list insert end $entry

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
                set $this-series-del [$selected.list get $i] 
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