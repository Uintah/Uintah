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

# GUI for HDF5DataReader module
# by Allen R. Sanderson
# May 2003

# This GUI interface is for selecting a file name via the makeOpenFilebox
# and other reading functions.

itcl_class Teem_DataIO_HDF5DataReader {
    inherit Module
    constructor {config} {
        set name HDF5DataReader
        set_defaults
    }

    method set_defaults {} {

	global $this-mergeData
	global $this-assumeSVT

	set $this-mergeData 1
	set $this-assumeSVT 1

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

	global $this-ndims
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
    }

    method make_file_open_box {} {
	global env
	global $this-filename

	set w [format "%s-fb" .ui[modname]]

	if {[winfo exists $w]} {
	    set child [lindex [winfo children $w] 0]

	    # $w withdrawn by $child's procedures
	    raise $child
	    return;
	}

	toplevel $w
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
	
	#######################################################
	# to be modified for particular reader

	# extansion to append if no extension supplied by user
	set defext ".fld"
	set title "Open HDF5 file"
	
	# file types to appers in filter box
	set types {
	    {{HDF5 File} {.h5}}
	    {{All Files} {.* }}
	}
	
	######################################################
	
	makeOpenFilebox \
	    -parent $w \
	    -filevar $this-filename \
	    -command "$this-c update_file; destroy $w" \
	    -cancel "destroy $w" \
	    -title $title \
	    -filetypes $types \
	    -initialdir $initdir \
	    -defaultextension $defext
    }

    method ui {} {
	global env
	set w .ui[modname]

        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }

#	if {[winfo exists $w]} {
#	    set child [lindex [winfo children $w] 0]
#
#	    # $w withdrawn by $child's procedures
#	    raise $child
#	    return;
#	}

	# Before building the tree save the current selections since
	# they erased when the tree is built.
	set datasets [set $this-datasets]


	toplevel $w

	# read an HDF5 file
	iwidgets::labeledframe $w.f -labeltext "HDF5 File Browser"
	set f [$w.f childsite]
	
	iwidgets::entryfield $f.fname -labeltext "File:" \
	    -textvariable $this-filename

	button $f.sel -text "Browse" \
	    -command "$this make_file_open_box"

	pack $f.fname $f.sel -side top -fill x -expand yes
	pack $w.f -fill x -expand yes -side top


	option add *TreeView.font { Courier 12 }
	#option add *TreeView.Button.background grey95
	#option add *TreeView.Button.activeBackground grey90
	#option add *TreeView.Column.background grey90
	option add *TreeView.Column.titleShadow { grey70 1 }
	#option add *TreeView.Column.titleFont { Helvetica 12 bold }
	option add *TreeView.Column.font { Courier 12 }

#	iwidgets::scrolledframe $w.treeview
	iwidgets::labeledframe $w.treeview -labeltext "File Treeview"

	set treeframe [$w.treeview childsite]

	global tree
	set tree [blt::tree create]    

	set treeview [Scrolled_Treeview $treeframe.tree \
			  -width 0 \
			  -selectmode multiple \
			  -selectcommand [list $this SelectNotify] \
			  -tree $tree]

  	pack $treeframe.tree -fill x -expand yes

	$treeview column configure treeView -text Node
	$treeview column insert end "Node-Type" "Data-Type" "Value"
	$treeview column configure "Node-Type" "Data-Type" "Value" \
	    -justify left -edit no
	$treeview column configure treeView -hide no -edit no
	$treeview text configure -selectborderwidth 0

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

	pack $treeview -fill x -expand yes -side top


	iwidgets::labeledframe $w.sel -labeltext "Search Selection"
	set sel [$w.sel childsite]

	global $this-selectionString
	global $this-regexp

	iwidgets::entryfield $sel.name -textvariable $this-selectionString

	label $sel.label -text "Reg-Exp" -width 7 -anchor w -just left
	checkbutton $sel.regexp -variable $this-regexp
	button $sel.path -text "Full Path" -command "$this AddSelection 0"
	button $sel.node -text "Terminal"  -command "$this AddSelection 1"

	pack $sel.node $sel.path $sel.regexp $sel.label -side right -padx 3
	pack $sel.name -side left -fill x -expand yes
	pack $w.sel -fill x -expand yes -side top



	iwidgets::labeledframe $w.sd -labeltext "Selected Data"
	set sd [$w.sd childsite]

	set listbox [Scrolled_Listbox $sd.listbox -width 100 -height 10 -selectmode extended]

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

	button $sd.delete -text "Delete Selection" \
	    -command "$this DeleteSelection"

	pack $sd.listbox $sd.delete -side top -fill x -expand yes
	pack $w.sd -fill x -expand yes -side top


	iwidgets::labeledframe $w.dm -labeltext "Data Management"
	set dm [$w.dm childsite]

	label $dm.mlikelabel -text "Merge like data" -width 15 -anchor w -just left
	radiobutton $dm.mergelike -variable $this-mergeData -value 1
	
	label $dm.mtimelabel -text "Merge time data" -width 15 -anchor w -just left
	radiobutton $dm.mergetime -variable $this-mergeData -value 2
	
	label $dm.asvt -text "Assume Vector-Tensor data" \
	    -width 33 -anchor w -just left
	checkbutton $dm.svt -variable $this-assumeSVT
	
	pack $dm.mlikelabel -side left
	pack $dm.mergelike  -side left
	pack $dm.mtimelabel -side left -padx  20
	pack $dm.mergetime  -side left
	pack $dm.asvt   -side left -padx  20
	pack $dm.svt    -side left

	pack $w.dm -fill x -expand yes -side top



	frame $w.l
	label $w.l.direction -text "Index"  -width 5 -anchor w -just left
	label $w.l.start     -text "Start"  -width 5 -anchor w -just left
	label $w.l.count     -text "Count"  -width 6 -anchor w -just left
	label $w.l.stride    -text "Stride" -width 6 -anchor w -just left

	pack $w.l.direction -side left
	pack $w.l.start     -side left -padx 100
	pack $w.l.count     -side left -padx 110
	pack $w.l.stride    -side left -padx  50

	frame $w.msg
	label $w.msg.text -text "Data selections do not have the same dimensions" -width 50 -anchor w -just center

	pack $w.msg.text -side top


	global $this-ndims
	global max_dims

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

	    frame $w.$i

	    label $w.$i.l -text " $index :" -width 3 -anchor w -just left

	    pack $w.$i.l -side left

	    scaleEntry2 $w.$i.start \
		0 $count_val1 200 \
		$this-$i-start $this-$i-start2

	    scaleEntry2 $w.$i.count \
		1 $count_val  200 \
		$this-$i-count $this-$i-count2

	    scaleEntry2 $w.$i.stride \
		1 [expr [set $this-$i-dim] - 1] 100 $this-$i-stride $this-$i-stride2

	    pack $w.$i.l $w.$i.start $w.$i.count \
		    $w.$i.stride -side left
#	    grid $w.$i.l $w.$i.start $w.$i.count 
#		    $w.$i.stride
	}

	frame $w.misc
	button $w.misc.execute -text "Execute" -command "$this-c needexecute"
	button $w.misc.close -text Close -command "destroy $w"
	pack $w.misc.execute $w.misc.close -side left -padx 25

	pack $w.l -side top -padx 10 -pady 5

	if { [set $this-ndims] == 0 } {
	    pack $w.msg -side top -pady 5
	} else {
	    for {set i 0} {$i < [set $this-ndims]} {incr i 1} {
		pack $w.$i
	    }
	}

	pack $w.misc -side top -padx 10 -pady 5	    


	# When building the UI prevent the selection from taking place
	# since it is not valid.
	global allow_selection
	set allow_selection false

	if { [string length [set $this-dumpname]] > 0 } {
	    # Rebuild the tree
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
		    set id [eval $treeview find -exact -full "{$dataset}"]

		    if {"$id" != ""} {
			if { [eval $treeview entry isopen $id] == 0 } {
			    $treeview open $id
			}

			$treeview selection set $id
		    } else {
			set message "Could not find dataset: "
			append message $dataset
			$this-c error $message
		    }
		}
	    }
	}

	# Makesure the datasets are saved once everything is built.
	set $this-datasets $datasets

 	set allow_selection true
    }

    method scaleEntry2 { win start count length var1 var2 } {
	frame $win 
	pack $win -side top -padx 5

	scale $win.s -from $start -to $count -length $length \
	    -variable $var1 -orient horizontal -showvalue false \
	    -command "$this updateSliderEntry $var1 $var2"

	entry $win.e -width 4 -text $var2

	bind $win.e <Return> "$this manualSliderEntry $start $count $var1 $var2"

	pack $win.s -side left
	pack $win.e -side bottom -padx 5
    }

    method updateSliderEntry {var1 var2 someUknownVar} {
	set $var2 [set $var1]
    }

    method manualSliderEntry { start count var1 var2 } {
	if { [set $var2] < $start } {
	    set $var2 $start }
	
	if { [set $var2] > $count } {
	    set $var2 $count }
	
	set $var1 [set $var2]
    }

    method build_tree { filename } {

	global $this-dumpname
	set $this-dumpname $filename

	global $this-datasets
	set $this-datasets ""

	set w .ui[modname]

	if [ expr [winfo exists $w] ] {
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
		    set read_error 1
		    $this-c update_file
		    return
		} else {
		    set read_error 0
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
	    }

	    close $fileId

	    $treeview entry configure "attribute" -foreground green4
	    $treeview entry configure "dataset"   -foreground cyan4
	    
	    set ids [$treeview entry children root]
	    foreach id $ids {
		$treeview open $id
	    }
	}
    }


    method process_file { tree parent fileId input } {

	while {[gets $fileId line] >= 0 && [string first "\}" $line] == -1} {

	    if { [string first "GROUP" $line] != -1 } {
		process_group $tree $parent $fileId $line
	    } else {
		$this-c error "File hierarchy mal formed."
	    }
	}
    }

    method process_group { tree parent fileId input } {

	set name [process_name $input]
	set info(type) @blt::tv::normalOpenFolder
	set info(Node-Type) ""
	set info(Data-Type) ""
	set info(Value) ""
	set node [$tree insert $parent -tag "group" -label $name \
		      -data [array get info]]

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
		$this-c error message
	    }
	}
    }

    method process_attribute { tree parent fileId input } {

	if {[gets $fileId line] >= 0 && \
	    [string first "DATA" $line] == -1} {
	    set message "Bad attribute data formation: "
	    append message $line
	    $this-c error message
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
	    $this-c error message
	} else {
	    set name [process_name $input]
	    set info(type) ""
	    set info(Node-Type) "Attribute"
	    set info(Data-Type) ""
	    set info(Value) $attr
	    $tree insert $parent -tag "attribute" -label $name \
		-data [array get info]
	}
    }

    method process_dataset { tree parent fileId input } {

	if {[gets $fileId line] >= 0 && \
	    [string first "DATATYPE" $line] == -1} {
	    set message "Bad dataset formation: "
	    append message $line
	    $this-c error message
	    return
	}

	set start [string first "\"" $line]
	set end   [string  last "\"" $line]
	set type  [string range $line [expr $start+1] [expr $end-1]]

	if {[gets $fileId line] >= 0 && \
	    [string first "DATASPACE" $line] == -1} {
	    set message "Bad dataset formation: "
	    append message $line
	    $this-c error message
	    return
	}

	set start [string first "(" $line]
	set end   [string  last ")" $line]
	set dims  [string range $line $start $end]

	if {[gets $fileId line] >= 0 && [string first "\}" $line] == -1} {
	    set message "Bad dataset formation: "
	    append message $line
	    $this-c error message
	} else {
	    set name [process_name $input]
	    set info(type) ""
	    set info(Node-Type) "DataSet"
	    set info(Data-Type) $type
	    set info(Value) $dims
	    $tree insert $parent -tag "dataset" -label $name \
		-data [array get info]
	}
    }

    method process_name { line } {
	set start [string first "\"" $line]
	set end   [string  last "\"" $line]

	set name [string range $line [expr $start+1] [expr $end-1]]

	return $name
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

	    global max_dims

	    pack forget $w.msg

	    for {set i 0} {$i < $max_dims} {incr i 1} {
		pack forget $w.$i
	    }

	    pack forget $w.misc

	    if { [set $this-ndims] == 0 } {
		pack $w.msg -side top -pady 5
	    } else {
		for {set i 0} {$i < [set $this-ndims]} {incr i 1} {
		    pack $w.$i -side top -padx 10 -pady 5
		}
	    }
	    
	    pack $w.misc -side top -padx 10 -pady 5	    

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
		    $w.$i.start.s  configure -from 0 -to $count_val1
		    $w.$i.count.s  configure -from 1 -to $count_val
		    $w.$i.stride.s configure -from 1 -to $count_val

		    bind $w.$i.start.e <Return> \
			"$this manualSliderEntry 0 $count_val1 $this-$i-start $this-$i-start2 $i"
		    bind $w.$i.count.e  <Return> \
			"$this manualSliderEntry 1 $count_val $this-$i-count $this-$i-count2"
		    bind $w.$i.stride.e  <Return> \
			"$this manualSliderEntry 1 $count_val $this-$i-stride $this-$i-stride2"
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

		set groups     [$treeview tag nodes "group"]
		set attributes [$treeview tag nodes "attribute"]

		focus $treeview

		set ids [$treeview curselection]

		if { $ids != "" } {
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

		updateSelection
	    }

	    set allow_selection true
	}
    }

    method SelectChildrenDataSet { parent } {
	set w .ui[modname]

	if [ expr [winfo exists $w] ] {

	    set treeframe [$w.treeview childsite]
	    set treeview $treeframe.tree.tree

	    set datasets [$treeview tag nodes "dataset"]
	    set groups   [$treeview tag nodes "group"]

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


    method updateSelection { } {

	set w .ui[modname]

	if [ expr [winfo exists $w] ] {
	    set treeframe [$w.treeview childsite]
	    set treeview $treeframe.tree.tree

	    set ids [$treeview curselection]
	    set names [eval $treeview get -full $ids]

	    global $this-datasets
	    set $this-datasets $names

	    set sd [$w.sd childsite]
	    set listbox $sd.listbox
	    $listbox.list delete 0 end

	    if { [string first "\{" $names] == 0 } {
		set tmp $names
	    } else {
		set tmp "\{"
		append tmp $names
		append tmp "\}"
	    }

	    foreach dataset $tmp {
		$listbox.list insert end $dataset
	    }

	    $this-c update_selection;
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
			set id [eval $treeview find -exact -full "{$dataset}"]
			
			if {"$id" != ""} {
			    $treeview selection clear $id
			    $treeview open $id
			} else {			    
			    set message "Could not find dataset: "
			    append message $dataset
			    $this-c error $message
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
	eval {blt::treeview $f.tree \
		  -xscrollcommand [list $f.xscroll set] \
		  -yscrollcommand [list $f.yscroll set]}
	eval {$f.tree configure} $args

	scrollbar $f.xscroll -orient horizontal \
	    -command [list $f.tree xview]
	scrollbar $f.yscroll -orient vertical \
	    -command [list $f.tree yview]
#	grid $f.tree $f.yscroll -sticky news
#	grid $f.xscroll -sticky news
#	grid rowconfigure $f 0 -weight 1
#	grid columnconfigure $f 0 -weight 1
	return $f.tree
    }

 }


