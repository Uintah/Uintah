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

	set $this-mergeData 0
	set $this-assumeSVT 0

	global max_dims
	set max_dims 6

	global $this-filename
	global $this-datasets
	global $this-dumpname
	set $this-filename ""
	set $this-datasets ""
	set $this-dumpname ""

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
	set title "Open nrrd file"
	
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

	# When building the UI prevent the selection from taking place
	# since it is not valid.
	global allow_selection
	set allow_selection false


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


	# Before building the tree save the current selections since
	# they erased when the tree is built.
	set datasets [set $this-datasets]



	option add *TreeView.font { Courier 12 }
	#option add *TreeView.Button.background grey95
	#option add *TreeView.Button.activeBackground grey90
	#option add *TreeView.Column.background grey90
	option add *TreeView.Column.titleShadow { grey70 1 }
	#option add *TreeView.Column.titleFont { Helvetica 12 bold }
	option add *TreeView.Column.font { Courier 12 }

	iwidgets::scrolledframe $w.treeview

	set treeframe [$w.treeview childsite]

	global tree
	set tree [blt::tree create]    

	blt::treeview $treeframe.tree \
	    -width 0 \
	    -selectmode multiple \
	    -selectcommand [list $this SelectDataSet] \
	    -tree $tree

  	pack $treeframe.tree

	$treeframe.tree column configure treeView -text Group
	$treeframe.tree column insert end Type Value

	$treeframe.tree column configure Type Value -justify left -edit no
	$treeframe.tree column configure treeView -hide no -edit no
	$treeframe.tree text configure -selectborderwidth 0

	focus $treeframe.tree

  	pack $w.treeview -fill both -expand yes -side top

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
		    set id [eval $treeframe.tree find -exact -full "{$dataset}"]

		    if {"$id" != ""} {
			$treeframe.tree selection set $id
		    } else {

			set message "Could not find dataset: "
			append message $dataset
			$this-c error $message
		    }
		}
	    }
	}

	$treeframe.tree column bind all <ButtonRelease-3> {
	    %W configure -flat no
	}

	foreach column [$treeframe.tree column names] {
	    $treeframe.tree column configure $column -command [list $this SortColumn $column]
	}

	pack $treeframe.tree -fill x -expand yes -side top



	iwidgets::labeledframe $w.dm -labeltext "Data Management"
	set dm [$w.dm childsite]

	label $dm.mlabel -text "Merge like data" -width 15 -anchor w -just left
	checkbutton $dm.merge -variable $this-mergeData
	
	label $dm.asvt -text "Assume Scalar-Vector-Tensor data" -width 33 -anchor w -just left
	checkbutton $dm.svt -variable $this-assumeSVT
	
	pack $dm.mlabel -side left
	pack $dm.merge  -side left
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

	global $this-ndims

	global max_dims

	for {set i 0} {$i < $max_dims} {incr i 1} {
	    if     { $i == 0 } { set index i
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

	    # Update the sliders to have the new end values.

	    set start_val 1
	    set count_val  [expr [set $this-$i-dim] ]
	    set count_val1 [expr [set $this-$i-dim] - 1 ]

	    frame $w.$i

	    label $w.$i.l -text " $index :" -width 3 -anchor w -just left

	    pack $w.$i.l -side left

	    scaleEntry4 $w.$i.start \
		0 $count_val1 200 \
		$this-$i-start $this-$i-start2 $i

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

	for {set i 0} {$i < [set $this-ndims]} {incr i 1} {
	    pack $w.$i
	}

	pack $w.misc -side top -padx 10 -pady 5	    

	# Makesure the datasets are saved once everything is built.
	set $this-datasets $datasets
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


    method scaleEntry4 { win start count length var1 var2 i } {
	frame $win 
	pack $win -side top -padx 5

	scale $win.s -from $start -to $count -length $length \
	    -variable $var1 -orient horizontal -showvalue false \
	    -command "$this updateSliderEntry4 $i"

	entry $win.e -width 4 -text $var2

	bind $win.e <Return> \
	    "$this manualSliderEntry4 $start $count $var1 $var2 $i"

	pack $win.s -side left
	pack $win.e -side bottom -padx 5
    }

    method updateSliderEntry4 { i someUknownVar } {

	global $this-$i-start
	global $this-$i-start2
	global $this-$i-count
	global $this-$i-count2

	set $this-$i-start2 [set $this-$i-start]
	set $this-$i-count2  [set $this-$i-count]
    }

    method manualSliderEntry4 { start count var1 var2 i } {

	if { [set $var2] < $start } {
	    set $var2 $start }
	
	if { [set $var2] > $count } {
	    set $var2 $count }
	
	set $var1 [set $var2]

	updateSliderEntry4 $i 0
    }

    method build_tree { filename } {

	global $this-dumpname
	set $this-dumpname $filename

	global $this-datasets
	set $this-datasets ""

	set w .ui[modname]

	if [ expr [winfo exists $w] ] {
	    set treeframe [$w.treeview childsite]

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

	    $treeframe.tree entry configure "attribute" -foreground green4
	    $treeframe.tree entry configure "dataset"   -foreground cyan4
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
	set info(Type) ""
	set info(Value) ""
	set node [$tree insert $parent -label $name -data [array get info]]

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
	    set info(Type) "Attribute"
	    set info(Value) $attr
	    $tree insert $parent  -tag "attribute" -label $name \
		-data [array get info]
	}
    }

    method process_dataset { tree parent fileId input } {

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
	    set info(Type) "DataSet"
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

	    for {set i 0} {$i < $max_dims} {incr i 1} {
		pack forget $w.$i
	    }

	    pack forget $w.misc

	    for {set i 0} {$i < [set $this-ndims]} {incr i 1} {
		pack $w.$i -side top -padx 10 -pady 5
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
			"$this manualSliderEntry4 0 $count_val1 $this-$i-start $this-$i-start2 $i"
		    bind $w.$i.count.e  <Return> \
			"$this manualSliderEntry  1 $count_val $this-$i-count $this-$i-count2"
		    bind $w.$i.stride.e  <Return> \
			"$this manualSliderEntry  1 $count_val $this-$i-stride $this-$i-stride2"
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

	    set old [$treeframe.tree sort cget -column] 
	    set decreasing 0
	    if { "$old" == "$column" } {
		set decreasing [$treeframe.tree sort cget -decreasing]
		set decreasing [expr !$decreasing]
	    }
	    $treeframe.tree sort configure -decreasing $decreasing -column $column -mode integer
	    $treeframe.tree configure -flat yes
	    $treeframe.tree sort auto yes

	    blt::busy hold $treeframe.tree
	    update
	    blt::busy release $treeframe.tree
	}
    }

    method SelectDataSet { } {

	global allow_selection

	if { "$allow_selection" == "true" } {

	    set w .ui[modname]

	    global tree

	    if [ expr [winfo exists $w] ] {

		set treeframe [$w.treeview childsite]

		set datasets [$treeframe.tree tag nodes "dataset"]

		set ids [$treeframe.tree curselection]

		foreach id $ids {

		    set deactivate true

		    foreach dataset $datasets {

			if { $dataset == $id } { 
			    set deactivate "false"
			    break
			}
		    }

		    if { $deactivate == "true" } {
			$treeframe.tree selection clear $id
		    }
		}

		set ids [$treeframe.tree curselection]
		set names [eval $treeframe.tree get -full $ids]

		global $this-datasets
		set $this-datasets $names

		$this-c update_selection;
	    }
	} else {
	    set allow_selection true
	}
    }
}
