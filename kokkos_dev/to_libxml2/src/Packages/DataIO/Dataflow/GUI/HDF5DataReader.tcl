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


# GUI for DataIO_Readers_HDF5DataReader module
# by Allen R. Sanderson
# May 2003

# This GUI interface is for selecting a file name via the makeOpenFilebox
# and other reading functions.

itcl_class DataIO_Readers_HDF5DataReader {
    inherit Module
    constructor {config} {
        set name HDF5DataReader
        set_defaults
    }

    method set_defaults {} {

	global power_app_command
	set    power_app_command ""

	global $this-animate_frame
	set $this-animate_frame ""

	global have_groups
	global have_attributes
	global have_datasets

	set have_groups     0
	set have_attributes 0
	set have_datasets   0
 
	global $this-animate_tab
	global $this-basic_tab
	global $this-extended_tab
	global $this-playmode_tab

	set $this-animate_tab ""
	set $this-basic_tab ""
	set $this-extended_tab ""
	set $this-playmode_tab ""

        global $this-selectable_min
        global $this-selectable_max
        global $this-selectable_inc
        global $this-range_min
        global $this-range_max
	global $this-playmode
	global $this-current
	global $this-execmode
	global $this-delay
	global $this-inc-amount

	global $this-update_type
	global $this-continuous

        set $this-selectable_min     0
        set $this-selectable_max     100
        set $this-selectable_inc     1
        set $this-range_min          0
        set $this-range_max          0
	set $this-playmode           once
	set $this-current            0
	set $this-execmode           "init"
	set $this-delay              0
	set $this-inc-amount         1

	set $this-update_type "On Release"
	set $this-continuous 0

	trace variable $this-current w "update idletasks;\#"

	global $this-mergeData
	global $this-assumeSVT
	global $this-animate
	global $this-animate-style
	global $this-animate-frame
	global $this-animate-frame2
	global $this-animate-nframes

	set $this-mergeData 1
	set $this-assumeSVT 1
	set $this-animate   0
	set $this-animate-style 0
	set $this-animate-frame 0
	set $this-animate-frame2 "0"
	set $this-animate-nframes 1

	global max_dims
	set max_dims 6

	global $this-filename
	global $this-datasets
	global $this-dumpname
	global $this-selectionString
	global $this-regexp

	set $this-filename ""
	set $this-datasets ""
	set $this-dumpname ""
	set $this-selectionString ""
	set $this-regexp 0

	global $this-ports
	global $this-ndims
	set $this-ports ""
	set $this-ndims 0

	for {set i 0} {$i < $max_dims} {incr i 1} {
	    global $this-$i-dim
	    global $this-$i-start
	    global $this-$i-start2
	    global $this-$i-count
	    global $this-$i-count2
	    global $this-$i-stride
	    global $this-$i-stride2

	    set $this-$i-dim     2
	    set $this-$i-start   0
	    set $this-$i-start2 "0"
	    set $this-$i-count    1
	    set $this-$i-count2  "1"
	    set $this-$i-stride    1
	    set $this-$i-stride2  "1"
	}

	global allow_selection
	set allow_selection true

	global read_error
	set read_error 0

	trace variable $this-update_type    w "$this update_type_callback"
	trace variable $this-selectable_max w "$this update_range_callback"
    }

    method set_power_app_cmd { cmd } {
	global power_app_command
	set power_app_command $cmd
    }

    method make_file_open_box {} {
	global $this-filename

	set w [format "%s-filebox" .ui[modname]]

	if {[winfo exists $w]} {
	    moveToCursor $w
	    wm deiconify $w
	    return;
	}

	toplevel $w -class TkFDialog

	global current_cursor
	$w config -cursor $current_cursor

	# place to put preferred data directory
	# it's used if $this-filename is empty
	set initdir [netedit getenv SCIRUN_DATA]
	
	#######################################################
	# to be modified for particular reader

	# extansion to append if no extension supplied by user
	set defext ".h5"
	set title "Open HDF5 file"
	
	# file types to appers in filter box
	set types {
	    {{HDF5 File} {.h5}}
	    {{All Files} {.* }}
	}
	
	######################################################
	
	global current_cursor	
	$w config -cursor $current_cursor

	makeOpenFilebox \
	    -parent $w \
	    -filevar $this-filename \
	    -commandname "Open" \
	    -command "$w config -cursor watch; \
                      $this set_watch_cursor; \
                      $this-c update_file 0;
                      $w config -cursor $current_cursor;\
                      wm withdraw $w" \
	    -cancel "wm withdraw $w" \
	    -title $title \
	    -filetypes $types \
	    -initialdir $initdir \
	    -defaultextension $defext


	$w config -cursor $current_cursor

	moveToCursor $w
	wm deiconify $w
    }

    method ui {} {
  	global $this-mergeData
  	global $this-assumeSVT
  	global $this-animate
  	global $this-animate-style
  	global $this-animate-frame
  	global $this-animate-frame2
  	global $this-animate-nframes

  	global $this-filename
  	global $this-datasets
  	global $this-dumpname
  	global $this-selectionString
  	global $this-regexp

  	global $this-ports
  	global $this-ndims
  	global max_dims

        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }

	# Before building the tree save the current selections since
	# they are erased when the tree is built.
	set datasets [set $this-datasets]

	toplevel $w

	global current_cursor
	set current_cursor [$w cget -cursor]

	# read an HDF5 file
	iwidgets::labeledframe $w.browser -labeltext "HDF5 File Browser"
	set f [$w.browser childsite]

	frame $f.fname
	label $f.fname.l -text "File:"
	entry $f.fname.e -width 64 -textvariable $this-filename

	pack $f.fname.l -side left -anchor nw -padx 3

	pack $f.fname.e \
	    -side left -anchor nw -padx 3 -fill both -expand 1

	bind $f.fname.e <Return> \
	    "$w config -cursor watch; \
             $this set_watch_cursor; \
             $this-c update_file 0;
             $w config -cursor $current_cursor;"

	frame $f.buttons

	button $f.buttons.browse -text "Browse" \
	    -command "$this make_file_open_box"

	button $f.buttons.clear -text "Clear" \
	    -command "$this clear"

	pack $f.buttons.browse $f.buttons.clear -side left -fill x -expand yes

	pack $f.fname $f.buttons -side top -fill x -expand yes

  	pack $w.browser -side top -pady 10 -fill x -expand yes


	option add *TreeView.font { Courier 12 }
#	option add *TreeView.Button.background grey95
#	option add *TreeView.Button.activeBackground grey90
#	option add *TreeView.Column.background grey90
	option add *TreeView.Column.titleShadow { grey70 1 }
#	option add *TreeView.Column.titleFont { Helvetica 12 bold }
	option add *TreeView.Column.font { Courier 12 }

	iwidgets::labeledframe $w.treeview -labeltext "File Treeview"

	set treeframe [$w.treeview childsite]

	global tree
	set tree [blt::tree create]    

	set treeview [Scrolled_Treeview $treeframe.tree \
			  -width 600 -height 225 \
			  -selectmode multiple \
			  -selectcommand [list $this SelectNotify] \
			  -tree $tree]

  	pack $treeframe.tree -side top -pady 10 -fill x -expand yes

	$treeview column configure treeView -text Node
	$treeview column insert end "Node-Type" "Data-Type" "Value"
	$treeview column configure "Node-Type" "Data-Type" "Value" \
	    -justify left -edit no
	$treeview column configure treeView -hide no -edit no
#	$treeview text configure -selectborderwidth 0

	focus $treeview

  	pack $w.treeview -fill both -expand yes -side top

	$treeview column bind all <ButtonRelease-3> {
	    %W configure -flat no
	}

	$treeview column bind all <ButtonRelease-3> {
	    %W configure -flat no
	}

	foreach column [$treeview column names] {
	    $treeview column configure $column \
		-command [list $this SortColumn $column]
	}



	iwidgets::labeledframe $w.search -labeltext "Search Selection"
	set search [$w.search childsite]


	iwidgets::entryfield $search.name -textvariable $this-selectionString

	frame $search.options

	checkbutton $search.options.regexp -variable $this-regexp
	label $search.options.label -text "Reg-Exp" -width 7 \
	    -anchor w -just left
	button $search.options.path -text "Full Path" \
	    -command "$this AddSelection 0"
	button $search.options.node -text "Terminal"  \
	    -command "$this AddSelection 1"

	pack $search.options.regexp -side left -padx 5
	pack $search.options.label -side left
	pack $search.options.path $search.options.node -padx 5 -side left
	pack $search.options -side right

	pack $search.name -side left -fill x -expand yes -pady 10
	pack $w.search -fill x -expand yes -side top -pady 10


#       Selected Data
	iwidgets::labeledframe $w.sd -labeltext "Selected Data"
	set sd [$w.sd childsite]

	set listbox [Scrolled_Listbox $sd.listbox \
			 -width 100 -height 8 \
			 -selectmode extended]

	if { [string first "\{" $datasets] == 0 } {
	    set tmp $datasets
	} else {
	    set tmp "\{"
	    append tmp $datasets
	    append tmp "\}"
	}

	foreach dataset $tmp {
	    $listbox insert end $dataset
	}

	frame $sd.buttons

	button $sd.buttons.delete -text "Delete Selection" \
	    -command "$this DeleteSelection"

	button $sd.buttons.deleteall -text "Delete All" \
	    -command "$this DeleteAll"

	pack $sd.buttons.delete $sd.buttons.deleteall -side left -fill x -expand yes

	pack $sd.listbox $sd.buttons -side top -fill x -expand yes
	pack $w.sd -fill x -expand yes -side top


	iwidgets::labeledframe $w.dm -labeltext "Data Management"
	set dm [$w.dm childsite]

	frame $dm.merge

	frame $dm.merge.none
	radiobutton $dm.merge.none.button -variable $this-mergeData -value 0
	label $dm.merge.none.label -text "No Merging" -width 20 \
	    -anchor w -just left
	
	pack $dm.merge.none.button $dm.merge.none.label -side left


	frame $dm.merge.like
	radiobutton $dm.merge.like.button -variable $this-mergeData -value 1
	label $dm.merge.like.label -text "Merge like data" -width 20 \
	    -anchor w -just left
	
	pack $dm.merge.like.button $dm.merge.like.label -side left



	frame $dm.merge.time
	radiobutton $dm.merge.time.button -variable $this-mergeData -value 2
	label $dm.merge.time.label -text "Merge time data" -width 20 \
	    -anchor w -just left
	
	pack $dm.merge.time.button $dm.merge.time.label -side left

	pack $dm.merge.none $dm.merge.like $dm.merge.time -side top


	frame $dm.svt

	checkbutton $dm.svt.button -variable $this-assumeSVT
	label $dm.svt.label -text "Assume Vector-Tensor data" \
	    -width 30 -anchor w -just left
	
	pack $dm.svt.button $dm.svt.label  -side left


	frame $dm.animate

	checkbutton $dm.animate.button -variable $this-animate \
	    -command "$this animate"
	label $dm.animate.label -text "Animate selected data" \
	    -width 22 -anchor w -just left
	
	pack $dm.animate.button $dm.animate.label -side left


	pack $dm.merge $dm.svt $dm.animate -side left


	pack $w.dm -fill x -expand yes -side top -pady 10


	iwidgets::labeledframe $w.sample -labeltext "Data Sampling"
	set sample [$w.sample childsite]

	frame $sample.l
	label $sample.l.direction -text "Index"  -width 5 -anchor w -just left
	label $sample.l.start     -text "Start"  -width 5 -anchor w -just left
	label $sample.l.count     -text "Count"  -width 6 -anchor w -just left
	label $sample.l.stride    -text "Stride" -width 6 -anchor w -just left

	pack $sample.l.direction -side left
	pack $sample.l.start     -side left -padx 100
	pack $sample.l.count     -side left -padx 110
	pack $sample.l.stride    -side left -padx  50

	frame $sample.msg
	label $sample.msg.text -text "Data selections do not have the same dimensions" -width 50 -anchor w -just center

	pack $sample.msg.text -side top


	for {set i 0} {$i < $max_dims} {incr i 1} {
	    if       { $i == 0 } { set index i
	    } elseif { $i == 1 } { set index j
	    } elseif { $i == 2 } { set index k
	    } elseif { $i == 3 } { set index l
	    } elseif { $i == 4 } { set index m
	    } elseif { $i == 5 } { set index n
	    } else               { set index ".." }

	    global $this-$i-dim
	    global $this-$i-start
	    global $this-$i-start2
	    global $this-$i-count
	    global $this-$i-count2
	    global $this-$i-stride
	    global $this-$i-stride2

	    set $this-$i-start2  [set $this-$i-start]
	    set $this-$i-count2  [set $this-$i-count]
	    set $this-$i-stride2 [set $this-$i-stride]

	    # Update the sliders to have the new end values.

	    set start_val 1
	    set count_val  [expr [set $this-$i-dim] ]
	    set count_val1 [expr [set $this-$i-dim] - 1 ]

	    frame $sample.$i

	    label $sample.$i.l -text " $index :" -width 3 -anchor w -just left

	    pack $sample.$i.l -side left


	    scaleEntry2 $sample.$i.start \
		0 $count_val1 200 \
		$this-$i-start $this-$i-start2

	    scaleEntry2 $sample.$i.count \
		1 $count_val  200 \
		$this-$i-count $this-$i-count2

	    scaleEntry2 $sample.$i.stride \
		1 [expr [set $this-$i-dim] - 1] 100 \
		$this-$i-stride $this-$i-stride2

	    pack $sample.$i.l $sample.$i.start $sample.$i.count \
		    $sample.$i.stride -side left
#	    grid $sample.$i.l $sample.$i.start $sample.$i.count 
#		    $sample.$i.stride
	}

	pack $sample.l -side top -padx 10 -pady 5

	if { [set $this-ndims] == 0 } {
	    pack $sample.msg -side top -pady 5
	} else {
	    for {set i 0} {$i < [set $this-ndims]} {incr i 1} {
		pack $sample.$i
	    }
	}

	pack $w.sample -fill x -expand yes -side top

	# When building the UI prevent the selection from taking place
	# since it is not valid.
	global allow_selection
	set allow_selection false

	if { [string length [set $this-dumpname]] > 0 } {
	    set_watch_cursor
                      
	    # Make sure the dump file is up to date.
	    $this-c check_dumpfile 0
	    # Rebuild the tree.
	    build_tree [set $this-dumpname]

	    if { [string length $datasets] > 0 } {

		if { [string first "\{" $datasets] == 0 } {
		    set tmp $datasets
		} else {
		    set tmp "\{"
		    append tmp $datasets
		    append tmp "\}"
		}

		# Reselect the datasets
		foreach dataset $tmp {
		    set ids [eval $treeview find -exact -full "{$dataset}"]
		    
		    foreach id $ids {
 			if {"$id" != ""} {
			    if { [eval $treeview entry isopen $id] == 0 } {
				$treeview open $id
			    }
			    
			    $treeview selection set $id
			} else {
			    set message "Could not find dataset: "
			    append message $dataset
			    $this-c error $message
			    return
			}
		    }
		}
	    }

	    reset_cursor
	}

	# Make sure the datasets are saved once everything is built.
	set $this-datasets $datasets
 	set allow_selection true

	global $this-ports
	updateSelection [set $this-ports]

	animate

	global power_app_command

	if { [in_power_app] } {
	    makeSciButtonPanel $w $w $this -no_execute -no_close -no_find \
		"\"Close\" \"wm withdraw $w; $power_app_command\" \"Hides this GUI\""
	} else {
	    makeSciButtonPanel $w $w $this
	}
	 
	moveToCursor $w
    }

    method scaleEntry2 { win start count length var1 var2 } {
	frame $win 
	pack $win -side top -padx 5

	scale $win.s -from $start -to $count -length $length \
	    -variable $var1 -orient horizontal -showvalue false \
	    -command "$this updateSliderEntry $var1 $var2"

	entry $win.e -width 4 -text $var2

	bind $win.e <KeyRelease> \
	    "$this manualSliderEntry $start $count $var1 $var2"

	pack $win.s -side left
	pack $win.e -side bottom -padx 5
    }

    method updateSliderEntry {var1 var2 someUknownVar} {
	set $var2 [set $var1]
    }

    method manualSliderEntry { start count var1 var2 } {
	if { ![string is integer [set $var2]] } {
	    set $var2 [set $var1] }

	if { [set $var2] < $start } {
	    set $var2 $start }
	
	if { [set $var2] > $count } {
	    set $var2 $count }
	
	set $var1 [set $var2]
    }

    method clear {} {
	set w .ui[modname]

	if [ expr [winfo exists $w] ] {
	    set sd [$w.sd childsite]
	    set listbox $sd.listbox
	    $listbox.list delete 0 end
	
	    global tree
	    $tree delete root
	}

	global $this-filename
	global $this-datasets
	global $this-dumpname
	global $this-animate

	set $this-filename ""
	set $this-datasets ""
	set $this-dumpname ""

	set $this-animate 0
    }

    method build_tree { filename } {

	global $this-dumpname
	set $this-dumpname $filename

	global $this-datasets
	set $this-datasets ""

	set w .ui[modname]

	if [ expr [winfo exists $w] ] {
	    set_watch_cursor

	    set sd [$w.sd childsite]
	    set listbox $sd.listbox
	    $listbox.list delete 0 end

	    set treeframe [$w.treeview childsite]
	    set treeview $treeframe.tree.tree

	    global tree
	    $tree delete root

	    if {[catch {open $filename r} fileId]} {
		global read_error

		# the file may have been removed from the /tmp dir so
		# try to recreate it.
		if { $read_error == 0 } {
		    set message "Can not find "
		    append message $filename
		    $this-c error $message
		    return
		} else {
		    set message "Can not open "
		    append message $filename
		    $this-c error $message
		    return
		}
	    } elseif {[gets $fileId line] >= 0 &&
		      [string first HDF5* $line] != 1 } {
		process_file $tree root $fileId $line
	    } else {
		$this-c error "Not an HDF5 file."
		return
	    }

	    close $fileId

	    $treeview entry configure "attribute" -foreground green4
	    $treeview entry configure "dataset"   -foreground cyan4
	    
	    set ids [$treeview entry children root]
	    foreach id $ids {
		$treeview open $id
	    }

	    reset_cursor
	}
    }


    method process_file { tree parent fileId input } {

	global have_groups
	global have_attributes
	global have_datasets

	set have_groups     0
	set have_attributes 0
	set have_datasets   0
 
	while {[gets $fileId line] >= 0 && [string first "\}" $line] == -1} {

	    if { [string first "GROUP" $line] != -1 } {
		process_group $tree $parent $fileId $line
	    } else {
		$this-c error "File hierarchy mal formed."
		return
	    }
	}
    }

    method process_group { tree parent fileId input } {

	set gname [process_name $input]
	set info(type) @blt::tv::normalOpenFolder
	set info(Node-Type) ""
	set info(Data-Type) ""
	set info(Value) ""
	set node [$tree insert $parent -tag "group" -label $gname \
		      -data [array get info]]

	global have_groups
	set have_groups 1

	while {[gets $fileId line] >= 0 && [string first "\}" $line] == -1} {

	    if { [string first "GROUP" $line] != -1 } {
		process_group $tree $node $fileId $line
	    } elseif { [string first "ATTRIBUTE" $line] != -1 } {
		process_attribute $tree $node $fileId $line
	    } elseif { [string first "DATASET" $line] != -1 } {
		process_dataset $tree $node $fileId $line
	    } else {
		set message "Unknown token: "
		append message $line
		$this-c error $message
		return
	    }
	}
    }

    method process_attribute { tree parent fileId input } {

	if {[gets $fileId line] >= 0 && \
	    [string first "DATATYPE" $line] == -1} {
	    set message "Bad datatype formation: "
	    append message $line
	    $this-c error $message
	    return
	}

	set start [string first "\"" $line]
	set end   [string  last "\"" $line]
	set type  [string range $line [expr $start+1] [expr $end-1]]

	if {[gets $fileId line] >= 0 && \
	    [string first "DATA" $line] == -1} {
	    set message "Bad attribute data formation: "
	    append message $line
	    $this-c error $message
	    return
	}

	set attr ""

	while { [gets $fileId line] >= 0 && [string first "\}" $line] == -1 } {

	    if { [string length $attr] < 32 } { 
		if { [string length $attr] > 0 } { 
		    append attr ", "
		}
		append attr [string trim $line]
	    } elseif { [string first "..." $attr] == -1 } { 
		append attr " ..."
	    }
	}

	if {[gets $fileId line] >= 0 && [string first "\}" $line] == -1} {
	    set message "Bad attribute formation: "
	    append message $line
	    $this-c error $message
	    return
	} else {
	    set aname [process_name $input]
	    set info(type) ""
	    set info(Node-Type) "Attribute"
	    set info(Data-Type) $type
	    set info(Value) $attr
	    $tree insert $parent -tag "attribute" -label $aname \
		-data [array get info]

	    global have_attributes
	    set have_attributes 1
	}
    }

    method process_dataset { tree parent fileId input } {

	if {[gets $fileId line] >= 0 && \
	    [string first "DATATYPE" $line] == -1} {
	    set message "Bad dataset formation: "
	    append message $line
	    $this-c error $message
	    return
	}

	set start [string first "\"" $line]
	set end   [string  last "\"" $line]
	set type  [string range $line [expr $start+1] [expr $end-1]]

	if {[gets $fileId line] >= 0 && \
	    [string first "DATASPACE" $line] == -1} {
	    set message "Bad dataset formation: "
	    append message $line
	    $this-c error $message
	    return
	}

	set start [string first "(" $line]
	set end   [string  last ")" $line]
	set dims  [string range $line $start $end]

	set dsname [process_name $input]
	set info(type) ""
	set info(Node-Type) "DataSet"
	set info(Data-Type) $type
	set info(Value) $dims
	set node [$tree insert $parent -tag "dataset" -label $dsname \
		      -data [array get info]]
  
	global have_datasets
	set have_datasets 1

	while {[gets $fileId line] >= 0 && [string first "\}" $line] == -1} {

	    if { [string first "ATTRIBUTE" $line] != -1 } {
		process_attribute $tree $node $fileId $line
	    } else {
		set message "Unknown token: "
		append message $line
		$this-c error $message
		return
	    }
	}
    }

    method process_name { line } {
	set start [string first "\"" $line]
	set end   [string  last "\"" $line]

	return [string range $line [expr $start+1] [expr $end-1]]
    }

    method set_size {ndims dims} {
	global $this-ndims
	set $this-ndims $ndims

	set i 0

	foreach dim $dims {
	    global $this-$i-dim
	    set $this-$i-dim $dim

	    incr i 1
	}

	set w .ui[modname]

	# Update the count values to be at the initials values.

	if [ expr [winfo exists $w] ] {

	    set sample [$w.sample childsite]

	    global max_dims

	    pack forget $sample.msg

	    for {set i 0} {$i < $max_dims} {incr i 1} {
		pack forget $sample.$i
	    }

	    if { [set $this-ndims] == 0 } {
		pack $sample.msg -side top -pady 5
	    } else {
		for {set i 0} {$i < [set $this-ndims]} {incr i 1} {
		    pack $sample.$i -side top -padx 10 -pady 5
		}
	    }
	    
	    for {set i 0} {$i < [set $this-ndims]} {incr i 1} {
		global $this-$i-start
		global $this-$i-start2
		global $this-$i-count
		global $this-$i-count2
		global $this-$i-stride
		global $this-$i-stride2

		set count_val  [expr [set $this-$i-dim] ]
		set count_val1 [expr [set $this-$i-dim] - 1]

		if [ expr [winfo exists $w] ] {

		    # Update the sliders to the new bounds.
		    $sample.$i.start.s  configure -from 0 -to $count_val1
		    $sample.$i.count.s  configure -from 1 -to $count_val
		    $sample.$i.stride.s configure -from 1 -to $count_val

		    bind $sample.$i.start.e <KeyRelease> \
			"$this manualSliderEntry 0 $count_val1 $this-$i-start  $this-$i-start2"
		    bind $sample.$i.count.e  <KeyRelease> \
			"$this manualSliderEntry 1 $count_val  $this-$i-count  $this-$i-count2"
		    bind $sample.$i.stride.e <KeyRelease> \
			"$this manualSliderEntry 1 $count_val  $this-$i-stride $this-$i-stride2"
		}

		# Update the count values to be at the initials values.
		set $this-$i-start 0	    
		set $this-$i-count $count_val
		set $this-$i-stride  1

		# Update the text values.
		set $this-$i-start2 [set $this-$i-start]
		set $this-$i-count2  [set $this-$i-count]
		set $this-$i-stride2  [set $this-$i-stride]
	    }
	}
    }


    method SortColumn { column } {
	set w .ui[modname]

	if [ expr [winfo exists $w] ] {

	    set treeframe [$w.treeview childsite]
	    set treeview $treeframe.tree.tree

	    set old [$treeview sort cget -column] 
	    set decreasing 0
	    if { "$old" == "$column" } {
		set decreasing [$treeview sort cget -decreasing]
		set decreasing [expr !$decreasing]
	    }
	    $treeview sort configure -decreasing $decreasing -column $column -mode integer
	    $treeview configure -flat yes
	    $treeview sort auto yes

	    blt::busy hold $treeview
	    update
	    blt::busy release $treeview
	}
    }

    method SelectNotify { } {

	global allow_selection

	if { $allow_selection == "true" } {

	    set allow_selection false

	    set w .ui[modname]

	    if [ expr [winfo exists $w] ] {

		set treeframe [$w.treeview childsite]
		set treeview $treeframe.tree.tree

		focus $treeview

		set ids [$treeview curselection]

		if { $ids != "" } {

		    global have_groups
		    global have_attributes

		    if { $have_groups == 1 } {
			set groups [$treeview tag nodes "group"]
		    } else {
			set groups ""
		    }

		    if { $have_attributes == 1 } {
			set attributes [$treeview tag nodes "attribute"]
		    } else {
			set attributes ""
		    }

		    foreach id $ids {

			# Check to see if the selection is an attribute
			foreach attribute $attributes {
			    if { $attribute == $id } { 
				$treeview selection clear $id
				break
			    }
			}

			# Check to see if the selection is a group
			foreach group $groups {
			    if { $group == $id } { 
				$treeview selection clear $id
				SelectChildrenDataSet $id
				break
			    }
			}
		    }		    
		}

		global $this-ports
		updateSelection [set $this-ports]
	    }

	    set allow_selection true
	}
    }

    method SelectChildrenDataSet { parent } {
	set w .ui[modname]

	if [ expr [winfo exists $w] ] {

	    set treeframe [$w.treeview childsite]
	    set treeview $treeframe.tree.tree

	    global have_groups
	    global have_datasets

	    if { $have_groups == 1 } {
		set groups [$treeview tag nodes "group"]
	    } else {
		set groups ""
	    }

	    if { $have_datasets == 1 } {
		set datasets [$treeview tag nodes "dataset"]
	    } else {
		set datasets ""
	    }

	    set children [eval $treeview entry children $parent]
	    
	    foreach child $children {

		foreach dataset $datasets {
		    if { $dataset == $child } {
# Open the node so it can be selected properly
			if { [eval $treeview entry isopen $child] == 1 } {
			    $treeview selection set $child
			} else {
			    $treeview open $child
			    $treeview selection set $child
			    $treeview close $child
			}
			break
		    }
		}

		foreach group $groups {
		    if { $group == $child } { 
			SelectChildrenDataSet $child
			break
		    }
		}
	    }
	}
    }


    method updateSelection { ports } {

	set $this-ports $ports

	set w .ui[modname]

	if [ expr [winfo exists $w] ] {
	    set sd [$w.sd childsite]
	    set listbox $sd.listbox
	    $listbox.list delete 0 end

	    set treeframe [$w.treeview childsite]
	    set treeview $treeframe.tree.tree

	    set ids [$treeview curselection]

	    if { $ids != "" } {
		set names [eval $treeview get -full $ids]

		global $this-datasets

		if { {[set $this-datasets]} != {$names} } {
		    global $this-ports
		    set $ports ""
		    set $this-ports ""
		}
		
		set $this-datasets $names

		if { [string first "\{" $names] == 0 } {
		    set tmp $names
		} else {
		    set tmp "\{"
		    append tmp $names
		    append tmp "\}"
		}

		set cc 0

		foreach dataset $tmp {

		    set dsname $dataset

		    if { [string length $ports] > 0 } {
			set port [string range $ports [expr $cc*4] [expr $cc*4+3]]

			if { [string length $port] > 0 } {
			    append dsname "   Port "
			    append dsname $port
			}
		    }

		    $listbox.list insert end $dsname

		    incr cc
		}

		$this-c update_selection;
	    } else {
		global $this-datasets
		global $this-ports
		set $ports ""
		set $this-ports ""
		set $this-datasets ""
	    }
	}
    }


    method AddSelection { node } {
	set w .ui[modname]

	if [ expr [winfo exists $w] ] {

	    global allow_selection
	    set allow_selection false

	    global $this-selectionString
	    global $this-regexp

	    set treeframe [$w.treeview childsite]
	    set treeview $treeframe.tree.tree

	    set path -full
	    if { $node == 1 } {
		set path -name
	    }
	    
	    set match -exact
	    if {[set $this-regexp] == 1 } {
		set match -glob
	    }

	    set ids [eval $treeview find \
			 $match $path "{[set $this-selectionString]}" \
			    ]

	    foreach id $ids {
		if { [eval $treeview entry isopen $id] == 1 } {
		    $treeview selection set $id
		} else {
		    $treeview open $id
		    $treeview selection set $id
		    $treeview close $id
		}
	    }
	    
	    set allow_selection true

	    SelectNotify
	}
    }

    method DeleteAll { } {
	set w .ui[modname]

	if [ expr [winfo exists $w] ] {
	    set sd [$w.sd childsite]
	    set listbox $sd.listbox
	    $listbox.list delete 0 end

	    set treeframe [$w.treeview childsite]
	    set treeview $treeframe.tree.tree

	    set ids [$treeview curselection]

	    if { $ids != "" } {
		foreach id $ids {
		    $treeview selection clear $id
		    $treeview open $id
		}
	    }
	}
    }
	    
    method DeleteSelection { } {
	set w .ui[modname]

	if [ expr [winfo exists $w] ] {
	    set sd [$w.sd childsite]
	    set listbox $sd.listbox

	    set treeframe [$w.treeview childsite]
	    set treeview $treeframe.tree.tree

	    set indices [$listbox.list curselection]

	    if { [string first "\{" [set $this-datasets]] == 0 } {
		set tmp [set $this-datasets]
	    } else {
		set tmp "\{"
		append tmp [set $this-datasets]
		append tmp "\}"
	    }
	    
	    set index 0
	    # Reselect the datasets
	    foreach dataset $tmp {

		foreach idx $indices {
		    if { $index == $idx } { 
			set ids [eval $treeview find -exact -full "{$dataset}"]
			
			foreach id $ids {
			    if {"$id" != ""} {
				$treeview selection clear $id
				$treeview open $id
			    } else {			    
				set message "Could not find dataset: "
				append message $dataset
				$this-c error $message
				return
			    }
			}
		    }
		}

		incr index
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

    method Scrolled_Text { f args } {
	frame $f
	eval {text $f.text -wrap none \
		  -xscrollcommand [list $f.xscroll set] \
		  -yscrollcommand [list $f.yscroll set]} $args
	scrollbar $f.xscroll -orient horizontal \
	    -command [list $f.text xview]
	scrollbar $f.yscroll -orient vertical \
	    -command [list $f.text yview]
	grid $f.text $f.yscroll -sticky news
	grid $f.xscroll -sticky news
	grid rowconfigure $f 0 -weight 1
	grid columnconfigure $f 0 -weight 1
	return $f.text
    }

    method Scrolled_Treeview { f args } {
	frame $f

	if { [string match "IRIX*" $::tcl_platform(os)] } {
	    eval {blt::treeview $f.tree \
		      -xscrollcommand [list $f.xscroll set] \
		      -yscrollcommand [list $f.yscroll set]} $args
	} else {
	    eval {blt::treeview $f.tree \
		      -xscrollcommand [list $f.xscroll set] \
		      -yscrollcommand [list $f.yscroll set]}
	    eval { $f.tree configure } $args
	}

	scrollbar $f.xscroll -orient horizontal \
	    -command [list $f.tree xview]
	scrollbar $f.yscroll -orient vertical \
	    -command [list $f.tree yview]
	grid $f.tree $f.yscroll -sticky news
	grid $f.xscroll -sticky news
	grid rowconfigure $f 0 -weight 1
	grid columnconfigure $f 0 -weight 1
	return $f.tree
    }

    method animate {} {
	$this-c update_selection;

	global power_app_command

	if { ![in_power_app] } {
	    set w .ui[modname]
	    
	    if [ expr [winfo exists $w] ] {
		if { [set $this-animate] } {

		    set a [format "%s-animate" .ui[modname]]
	    
		    if {[winfo exists $a]} {
			set child [lindex [winfo children $a] 0]
		
			# $w withdrawn by $child's procedures
			raise $child
			return
		    }
		    
		    toplevel $a	
		    build_animate_ui $a

		} else {
		    set a [format "%s-animate" .ui[modname]]

		    if {[winfo exists $a]} {
			destroy $a
		    }
		}
	    }
	}
    }

    method maybeRestart { args } {
	upvar \#0 $this-execmode execmode
	if ![string equal $execmode play] return

	$this-c needexecute
    }

    method change_tab { which } {
	global $this-animate_tab
	set initialized 1

	# change tab for attached/detached

	if {$initialized != 0} {
	    if {$which == 0} {
		[set $this-animate_tab] view "Basic"
		
	    } elseif {$which == 1} {
		[set $this-animate_tab] view "Extended"
		
	    } elseif {$which == 2} {
		[set $this-animate_tab] view "Playmode"
	    }
	}
    }

    method build_animate_ui { w } {

	global $this-animate_frame
	set $this-animate_frame $w

	### Tabs
	iwidgets::tabnotebook $w.tnb -width 250 \
	    -height 250 -tabpos n
	pack $w.tnb -padx 0 -pady 0 -anchor n -fill both -expand 1

	global $this-animate_tab
	set animate_tab $w.tnb
	set $this-animate_tab $animate_tab

	global $this-basic_tab
	set basic_tab [$w.tnb add -label "Basic" -command "$this change_tab 0"]
	set $this-basic_tab $basic_tab

	frame $basic_tab.vcr -relief groove -borderwidth 2
	set vcr $basic_tab.vcr

	# load the VCR button bitmaps
	set image_dir [netedit getenv SCIRUN_SRCDIR]/pixmaps
	set rewind    [image create photo -file ${image_dir}/rewind-icon.ppm]
	set stepb     [image create photo -file ${image_dir}/step-back-icon.ppm]
	set pause     [image create photo -file ${image_dir}/pause-icon.ppm]
	set play      [image create photo -file ${image_dir}/play-icon.ppm]
	set stepf     [image create photo -file ${image_dir}/step-forward-icon.ppm]
	set fforward  [image create photo -file ${image_dir}/fast-forward-icon.ppm]

	# Create and pack the VCR buttons frame
	button $vcr.rewind -image $rewind \
	    -command "set $this-execmode rewind;   $this-c needexecute"
	button $vcr.stepb -image $stepb \
	    -command "set $this-execmode stepb;    $this-c needexecute"
	button $vcr.pause -image $pause \
	    -command "set $this-execmode stop;     $this-c needexecute"
	button $vcr.play  -image $play  \
	    -command "set $this-execmode play;     $this-c needexecute"
	button $vcr.stepf -image $stepf \
	    -command "set $this-execmode step;     $this-c needexecute"
	button $vcr.fforward -image $fforward \
	    -command "set $this-execmode fforward; $this-c needexecute"

	pack $vcr.rewind $vcr.stepb $vcr.pause \
	    $vcr.play $vcr.stepf $vcr.fforward -side left -fill both -expand 1
	global ToolTipText
	Tooltip $vcr.rewind   $ToolTipText(VCRrewind)
	Tooltip $vcr.stepb    $ToolTipText(VCRstepback)
	Tooltip $vcr.pause    $ToolTipText(VCRpause)
	Tooltip $vcr.play     $ToolTipText(VCRplay)
	Tooltip $vcr.stepf    $ToolTipText(VCRstepforward)
	Tooltip $vcr.fforward $ToolTipText(VCRfastforward)


	upvar \#0 $this-selectable_min min
	upvar \#0 $this-selectable_max max


	iwidgets::labeledframe $basic_tab.cur -labelpos nw -labeltext "Current"
	set tmp [$basic_tab.cur childsite]
	scale $tmp.cur -variable $this-current \
	    -from $min -to $max \
	    -showvalue true -orient horizontal -length 200 \
	    -command "$this updateCurrentEntry"
	pack $tmp.cur -side left -fill both -expand 1

	bind $tmp.cur <ButtonRelease> "$this updateCurrentEntryOnRelease"


	iwidgets::labeledframe $basic_tab.opt -labelpos nw -labeltext "Options"
	set opt [$basic_tab.opt childsite]
	
	iwidgets::optionmenu $opt.update -labeltext "Update:" \
		-labelpos w -command "$this set_update_type $opt.update"
	$opt.update insert end Auto Manual "On Release"
	$opt.update select [set $this-update_type]

	pack $opt.update -side left -fill both -expand 1


	pack $basic_tab.vcr -padx 5 -pady 10 -fill x -expand 0
	pack $basic_tab.cur -padx 5 -pady  5 -fill x -expand 0
	pack $basic_tab.opt -padx 5 -pady  5 -fill x -expand 0


	global $this-extended_tab
	set extended_tab [$w.tnb add -label "Extended" -command "$this change_tab 1"]
	set $this-extended_tab $extended_tab
	
	# Save range, creating the scale resets it to defaults.
	set rmin [set $this-range_min]
	set rmax [set $this-range_max]

	# Create the various range sliders
	iwidgets::labeledframe $extended_tab.min -labelpos nw -labeltext "Start"
	set tmp [$extended_tab.min childsite]
        scale $tmp.min -variable $this-range_min \
	    -from $min -to $max \
	    -showvalue true -orient horizontal -length 200 \
	    -command "$this maybeRestart"
	pack $tmp.min  -side left -fill both -expand 1

	iwidgets::labeledframe $extended_tab.max -labelpos nw -labeltext "End"
	set tmp [$extended_tab.max childsite]
	scale $tmp.max -variable $this-range_max \
	    -from $min -to $max \
	    -showvalue true -orient horizontal -length 200 \
	    -command "$this maybeRestart"
	pack $tmp.max  -side left -fill both -expand 1

	iwidgets::labeledframe $extended_tab.inc -labelpos nw -labeltext "Increment"
	set tmp [$extended_tab.inc childsite]
	scale $tmp.inc -variable $this-inc-amount \
	    -from 1 -to [expr $max-$min] \
	    -showvalue true -orient horizontal -length 200 \
	    -command "$this maybeRestart"
	pack $tmp.inc  -side left -fill both -expand 1

	pack $extended_tab.min $extended_tab.max $extended_tab.inc \
	    -padx 5 -fill x -expand 0

	# Restore range to pre-loaded value
	set $this-range_min $rmin
	set $this-range_max $rmax

	

	global $this-playmode_tab
	set playmode_tab [$w.tnb add -label "Playmode" -command "$this change_tab 2"]
	set $this-playmode_tab $playmode_tab
	set playmode $playmode_tab

	radiobutton $playmode.once -text "Once" \
		-variable $this-playmode -value once
	radiobutton $playmode.loop -text "Loop" \
		-variable $this-playmode -value loop
	radiobutton $playmode.bounce1 -text "Bounce" \
		-variable $this-playmode -value bounce1
	radiobutton $playmode.bounce2 -text "Bounce with repeating endpoints" \
		-variable $this-playmode -value bounce2

	radiobutton $playmode.inc_w_exec -text "Increment with Execute" \
	    -variable $this-playmode -value inc_w_exec

	# Save the delay since the iwidget resets it
	global $this-delay
	set delay [set $this-delay]
	iwidgets::spinint $playmode.delay -labeltext {Step Delay (ms)} \
	    -range {0 86400000} -justify right -width 5 -step 10 \
	    -textvariable $this-delay -repeatdelay 300 -repeatinterval 10
	$playmode.delay delete 0 end
	$playmode.delay insert 0 $delay

	trace variable $this-delay w "$this maybeRestart;\#"

	pack $playmode.once $playmode.loop \
	    $playmode.bounce1 $playmode.bounce2 $playmode.inc_w_exec\
	    $playmode.delay -side top -anchor w


	# Create the sci button panel

	global power_app_command

	if { ![in_power_app] } {
	    makeSciButtonPanel $w $w $this "-no_execute"
	}

	update
	change_tab 0
    }

    method update_type_callback { name1 name2 op } {
	global $this-animate_frame
	set w [set $this-animate_frame]

        if {[winfo exists $w]} {

	    upvar \#0 $this-basic_tab basic_tab

	    set opt [$basic_tab.opt childsite]

	    $opt.update select [set $this-update_type]
	}
    }

    method set_update_type { w } {
	global $w
	global $this-continuous
	global $this-update_type

	set $this-update_type [$w get]

	if { [set $this-update_type] == "Auto" } {
	    set $this-continuous 1
	} else {
	    set $this-continuous 0
	}
    }

    method update_range_callback {name element op} {
	global $this-animate_frame
	set w [set $this-animate_frame]

        if {[winfo exists $w]} {

	    upvar \#0 $this-selectable_min min
	    upvar \#0 $this-selectable_max max

	    upvar \#0 $this-basic_tab basic_tab
	    upvar \#0 $this-extended_tab extended_tab

	    set tmp [$basic_tab.cur childsite]
            $tmp.cur configure -from $min -to $max

	    set tmp [$extended_tab.min childsite]
            $tmp.min configure -from $min -to $max
	    set tmp [$extended_tab.max childsite]
	    $tmp.max configure -from $min -to $max
	    set tmp [$extended_tab.inc childsite]
	    $tmp.inc configure -from 1 -to [expr $max-$min]

	    set $this-range_min $min
	    set $this-range_max $max
	}
    }

    method updateCurrentEntry { op } {

	global $this-execmode
	set $this-execmode init

	global $this-continuous

	if { [set $this-continuous] == 1.0 } {
	    eval "$this-c needexecute"
	} elseif { [set $this-update_type] == "Auto" } {
	    set $this-continuous 1
	} 
    }

    method updateCurrentEntryOnRelease { } {

	global $this-execmode
	set $this-execmode init

	global $this-animate_frame
	set w [set $this-animate_frame]

        if {[winfo exists $w]} {

	    upvar \#0 $this-basic_tab basic_tab

	    set opt [$basic_tab.opt childsite]
	    
	    $opt.update select [set $this-update_type]

	    if { [$opt.update get] == "On Release" } {
		eval "$this-c needexecute"
	    }
	}
    }

    method set_watch_cursor {} {
	set w .ui[modname]

	if [ expr [winfo exists $w] ] {
	    $w config -cursor watch
	    update idletasks
	}
    }

    method reset_cursor {} {
	set w .ui[modname]

	if [ expr [winfo exists $w] ] {
	    global current_cursor
	    $w config -cursor $current_cursor
	    update idletasks
	}
    }
}
