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

# GUI for FusionFieldReader module
# by Allen R. Sanderson
# May 2003

# This GUI interface is for selecting a file name via the makeOpenFilebox
# and other reading functions.

catch {rename Fusion_DataIO_FusionFieldReader ""}

itcl_class Fusion_DataIO_HDF5ProtoType {
    inherit Module
    constructor {config} {
        set name HDF5ProtoType
        set_defaults
    }

    method set_defaults {} {
	global $this-filename
	global $this-dumpname
	set $this-filename ""
	set $this-dumpname ""

	global $this-readall
	global $this-dataset
	global $this-dataset2
	global $this-ndatasets

	set $this-readall 0
	set $this-dataset 0
	set $this-dataset2 "0"
	set $this-ndatasets 1

	global $this-ndims

	set $this-ndims 3

	for {set i 0} {$i < 3} {incr i 1} {
	    if { $i == 0 } {
		set index i
	    } elseif { $i == 1 } {
		set index j
	    } elseif { $i == 2 } {
		set index k
	    }

	    global $this-$index-dim
	    global $this-$index-start
	    global $this-$index-start2
	    global $this-$index-count
	    global $this-$index-count2
	    global $this-$index-stride
	    global $this-$index-stride2
	    global $this-$index-wrap

	    set $this-$index-dim     2
	    set $this-$index-start   0
	    set $this-$index-start2 "0"
	    set $this-$index-count    1
	    set $this-$index-count2  "1"
	    set $this-$index-stride    1
	    set $this-$index-stride2  "1"
	    set $this-$index-wrap    0
	}
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
	    build_tree [set $this-dumpname]
	}

	$treeframe.tree column bind all <ButtonRelease-3> {
	    %W configure -flat no
	}

	foreach column [$treeframe.tree column names] {
	    $treeframe.tree column configure $column -command [list $this SortColumn $column]
	}

	pack $treeframe.tree -fill x -expand yes -side top







	iwidgets::labeledframe $w.d -labeltext "File Data Set"
	set d [$w.d childsite]

#	label $d.title1 -text "Read all"  -width 8 -anchor w -just left
#	pack $d.title1  -side left
#	checkbutton $d.readall -variable $this-read-all 
#	pack $d.readall -side left -padx 5
#	label $d.title2 -text "Single"    -width  6 -anchor w -just left
#	pack $d.title2  -side left -padx 1
	scaleEntry2 $d.dataset 0 [expr [set $this-ndatasets] - 1] 200 \
	    $this-dataset $this-dataset2
	pack $d.dataset -side left

	pack $w.d -fill x -expand yes -side top


	frame $w.l
	label $w.l.direction -text "Index"  -width 5 -anchor w -just left
	label $w.l.start     -text "Start"  -width 5 -anchor w -just left
	label $w.l.count     -text "Count"  -width 6 -anchor w -just left
	label $w.l.stride    -text "Stride" -width 6 -anchor w -just left
	label $w.l.wrap      -text "Wrap"   -width 4 -anchor w -just left

	pack $w.l.direction -side left
	pack $w.l.start     -side left -padx  90
	pack $w.l.count     -side left -padx 110
	pack $w.l.stride    -side left -padx  50
	pack $w.l.wrap      -side left

#	grid $w.l.direction $w.l.start $w.l.count $w.l.stride $w.l.wrap

	global $this-ndims

	for {set i 0} {$i < 3} {incr i 1} {
	    if { $i == 0 } {
		set index i
	    } elseif { $i == 1 } {
		set index j
	    } elseif { $i == 2 } {
		set index k
	    }

	    global $this-$index-dim
	    global $this-$index-start
	    global $this-$index-start2
	    global $this-$index-count
	    global $this-$index-count2
	    global $this-$index-stride
	    global $this-$index-stride2
	    global $this-$index-wrap

	    # Update the sliders to have the new end values.

	    set start_val 1
	    set count_val  [expr [set $this-$index-dim] ]
	    set count_val1 [expr [set $this-$index-dim] - 1 ]

	    frame $w.$index

	    label $w.$index.l -text " $index :" -width 3 -anchor w -just left

	    pack $w.$index.l -side left

	    scaleEntry4 $w.$index.start \
		0 $count_val1 200 \
		$this-$index-start $this-$index-start2 $index

	    scaleEntry2 $w.$index.count \
		1 $count_val  200 \
		$this-$index-count $this-$index-count2

	    scaleEntry2 $w.$index.stride \
		1 [expr [set $this-$index-dim] - 1] 100 $this-$index-stride $this-$index-stride2

	    checkbutton $w.$index.wrap -variable $this-$index-wrap 
#		    -state $wrap -disabledforeground "" \
#		    -command "$this wrap $index"

	    pack $w.$index.l $w.$index.start $w.$index.count \
		    $w.$index.stride $w.$index.wrap -side left
#	    grid $w.$index.l $w.$index.start $w.$index.count 
#		    $w.$index.stride $w.$index.wrap
	}

	frame $w.misc
	button $w.misc.execute -text "Execute" -command "$this-c needexecute"
	button $w.misc.close -text Close -command "destroy $w"
	pack $w.misc.execute $w.misc.close -side left -padx 25

	if { [set $this-ndims] == 3 } {
	    pack $w.l $w.i $w.j $w.k $w.misc -side top -padx 10 -pady 5
	} elseif { [set $this-ndims] == 2 } {
	    pack $w.l $w.i $w.j $w.misc -side top -padx 10 -pady 5	    
	} elseif { [set $this-ndims] == 1 } {
	    pack $w.l $w.i $w.misc -side top -padx 10 -pady 5	    
	}
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


    method scaleEntry4 { win start count length var1 var2 index } {
	frame $win 
	pack $win -side top -padx 5

	scale $win.s -from $start -to $count -length $length \
	    -variable $var1 -orient horizontal -showvalue false \
	    -command "$this updateSliderEntry4 $index"

	entry $win.e -width 4 -text $var2

	bind $win.e <Return> \
	    "$this manualSliderEntry4 $start $count $var1 $var2 $index"

	pack $win.s -side left
	pack $win.e -side bottom -padx 5
    }

    method updateSliderEntry4 { index someUknownVar } {

	global $this-$index-start
	global $this-$index-start2
	global $this-$index-count
	global $this-$index-count2

	set $this-$index-start2 [set $this-$index-start]
	set $this-$index-count2  [set $this-$index-count]
    }

    method manualSliderEntry4 { start count var1 var2 index } {

	if { [set $var2] < $start } {
	    set $var2 $start }
	
	if { [set $var2] > $count } {
	    set $var2 $count }
	
	set $var1 [set $var2]

	updateSliderEntry4 $index 0
    }

    method build_tree { filename } {

	global $this-dumpname
	set $this-dumpname $filename

	set w .ui[modname]

	if [ expr [winfo exists $w] ] {
	    set treeframe [$w.treeview childsite]

	    global tree
	    $tree delete root

	    if {[catch {open $filename r} fileId]} {
		
		set message "Can not open "
		append message $filename
		$this-c error message
		return

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
	    set node [$tree insert $parent  -tag "attribute" -label $name -data [array get info]]
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
	    set node [$tree insert $parent -tag "dataset" -label $name -data [array get info]]
	}
    }

    method process_name { line } {
	set start [string first "\"" $line]
	set end   [string  last "\"" $line]

	set name [string range $line [expr $start+1] [expr $end-1]]

	return $name
    }

    method set_size {ndatasets ndims idim jdim kdim} {

	global $this-readall
	global $this-ndatasets
	global $this-dataset
	global $this-dataset2

	global $this-ndims
	global $this-i-dim
	global $this-j-dim
	global $this-k-dim

	set $this-ndatasets $ndatasets
	set $this-ndims $ndims
	set $this-i-dim $idim
	set $this-j-dim $jdim
	set $this-k-dim $kdim

	set w .ui[modname]

	set d [$w.d childsite]
	$d.dataset.s configure -from 0 -to [expr [set $this-ndatasets] - 1]

	# Update the count values to be at the initials values.
	set $this-readall 0
	set $this-dataset 0	    
	set $this-dataset2 [set $this-dataset]

	if [ expr [winfo exists $w] ] {
	    pack forget $w.i
	    pack forget $w.k
	    pack forget $w.j
	    pack forget $w.misc
	    
	    if { [set $this-ndims] == 3 } {
		pack $w.l $w.i $w.j $w.k $w.misc -side top -padx 10 -pady 5
	    } elseif { [set $this-ndims] == 2 } {
		pack $w.l $w.i $w.j $w.misc -side top -padx 10 -pady 5	    
	    } elseif { [set $this-ndims] == 1 } {
		pack $w.l $w.i $w.misc -side top -padx 10 -pady 5	    
	    }
	}

	for {set i 0} {$i < 3} {incr i 1} {
	    if { $i == 0 } {
		set index i
	    } elseif { $i == 1 } {
		set index j
	    } elseif { $i == 2 } {
		set index k
	    }

	    global $this-$index-start
	    global $this-$index-start2
	    global $this-$index-count
	    global $this-$index-count2
	    global $this-$index-stride
	    global $this-$index-stride2

	    set count_val  [expr [set $this-$index-dim] ]
	    set count_val1 [expr [set $this-$index-dim] - 1]

	    if [ expr [winfo exists $w] ] {

		# Update the sliders to the new bounds.
		$w.$index.start.s  configure -from 0 -to $count_val1
		$w.$index.count.s  configure -from 1 -to $count_val
		$w.$index.stride.s configure -from 1 -to $count_val

		bind $w.$index.start.e <Return> \
		    "$this manualSliderEntry4 0 $count_val1 $this-$index-start $this-$index-start2 $index"
		bind $w.$index.count.e  <Return> \
		    "$this manualSliderEntry  1 $count_val $this-$index-count $this-$index-count2"
		bind $w.$index.stride.e  <Return> \
		    "$this manualSliderEntry  1 $count_val $this-$index-stride $this-$index-stride2"
	    }

	    # Update the count values to be at the initials values.
	    set $this-$index-start 0	    
	    set $this-$index-count $count_val
	    set $this-$index-stride  1

	    # Update the text values.
	    set $this-$index-start2 [set $this-$index-start]
	    set $this-$index-count2  [set $this-$index-count]
	    set $this-$index-stride2  [set $this-$index-stride]
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
	    puts stderr $names
	}
    }
}
