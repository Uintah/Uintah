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


# GUI for DataIO_Readers_MDSPlusDataReader module
# by Allen R. Sanderson
# March 2002



itcl_class DataIO_Readers_MDSPlusDataReader {
    inherit Module
    constructor {config} {
        set name MDSPlusDataReader
        set_defaults
    }

    method set_defaults {} {
	global $this-power_app
	global power_app_command

	set    $this-power_app 0
	set    power_app_command ""

	global $this-num-entries
	set $this-num-entries 0
  
	global $this-check
	global $this-server
	global $this-tree
	global $this-shot
	global $this-signal
	global $this-status
	global $this-port

	set $this-check 0
	set $this-server "atlas.gat.com"
	set $this-tree "NIMROD"
	set $this-shot "10089"
	set $this-signal ""
	set $this-status "Unkonwn"
	set $this-port "na"

	global $this-search-path
	set $this-search-path 1

	global $this-mergeData
	global $this-assumeSVT

	set $this-mergeData 1
	set $this-assumeSVT 1

	global allow_selection
	set allow_selection true
    }
    
    method set_power_app { cmd } {
	global $this-power_app
	global power_app_command

	set $this-power_app 1
	set power_app_command $cmd
    }

    method ui {} {
	global $this-load-server
	global $this-load-tree
	global $this-load-shot
	global $this-load-signal

	global $this-search-server
	global $this-search-tree
	global $this-search-shot
	global $this-search-signal
	global $this-search-path

	global $this-mergeData
	global $this-assumeSVT

        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }

        toplevel $w

	global current_cursor
	set current_cursor [$w cget -cursor]


	iwidgets::labeledframe $w.loader -labeltext "MDS Tree Loader"

	set loadframe [$w.loader childsite]

	frame $loadframe.box

	frame $loadframe.box.title
	label $loadframe.box.title.server -text "Server" -width 16 -relief groove
	label $loadframe.box.title.tree   -text "Tree"   -width 12 -relief groove
	label $loadframe.box.title.shot   -text "Shot"   -width  8 -relief groove
	label $loadframe.box.title.signal -text "Signal" -width 48 -relief groove

	pack $loadframe.box.title.server $loadframe.box.title.tree \
	    $loadframe.box.title.shot $loadframe.box.title.signal \
	    -side left

	pack $loadframe.box.title


	frame $loadframe.box.entry
	entry $loadframe.box.entry.server -textvariable $this-load-server \
	    -width 16
	entry $loadframe.box.entry.tree   -textvariable $this-load-tree   \
	    -width 12
	entry $loadframe.box.entry.shot   -textvariable $this-load-shot   \
	    -width  8
	entry $loadframe.box.entry.signal -textvariable $this-load-signal \
	    -width 48

	pack $loadframe.box.entry.server $loadframe.box.entry.tree \
	    $loadframe.box.entry.shot $loadframe.box.entry.signal \
	    -side left 

	pack $loadframe.box.title $loadframe.box.entry -side top \
	     -fill both


	button $loadframe.load -text "  Load  " \
	    -command "$this AddRootSignals"

	pack $loadframe.box  -padx 15 -side left
	pack $loadframe.load -padx 20 -side left


  	pack $w.loader -side top -pady 10 -fill x -expand yes



	iwidgets::labeledframe $w.treeview -labeltext "MDS Treeview"

	set treeframe [$w.treeview childsite]

	option add *TreeView.font { Courier 12 }
#	option add *TreeView.Button.background grey95
#	option add *TreeView.Button.activeBackground grey90
#	option add *TreeView.Column.background grey90
	option add *TreeView.Column.titleShadow { grey70 1 }
#	option add *TreeView.Column.titleFont { Helvetica 12 bold }
	option add *TreeView.Column.font { Courier 12 }

	global tree
	set tree [blt::tree create]    

	set treeview [Scrolled_Treeview $treeframe.tree \
			  -width 600 -height 225 \
			  -selectcommand [list $this SelectNotify] \
			  -tree $tree]
	
	#-selectmode multiple \

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

	frame $search.box

	frame $search.box.title
	label $search.box.title.server -text "Server" -width 16 -relief groove
	label $search.box.title.tree   -text "Tree"   -width 12 -relief groove
	label $search.box.title.shot   -text "Shot"   -width  8 -relief groove
	label $search.box.title.signal -text "Signal" -width 40 -relief groove

	pack $search.box.title.server $search.box.title.tree \
	    $search.box.title.shot $search.box.title.signal \
	    -side left

	pack $search.box.title

	frame $search.box.entry
	entry $search.box.entry.server -textvariable $this-search-server \
	    -width 16
	entry $search.box.entry.tree   -textvariable $this-search-tree   \
	    -width 12
	entry $search.box.entry.shot   -textvariable $this-search-shot   \
	    -width  8
	entry $search.box.entry.signal -textvariable $this-search-signal \
	    -width 40

	pack $search.box.entry.server $search.box.entry.tree \
	    $search.box.entry.shot $search.box.entry.signal \
	    -side left 

	pack $search.box.title $search.box.entry -side top \
	     -fill both


	frame $search.options

	frame $search.options.path

	frame $search.options.path.absolute

	radiobutton $search.options.path.absolute.button \
	    -variable $this-search-path -value 1
	label $search.options.path.absolute.label -text "Absolute" -width 9 \
	    -anchor w -just left

	pack $search.options.path.absolute.button \
	    $search.options.path.absolute.label -side left


	frame $search.options.path.relative

	radiobutton $search.options.path.relative.button \
	    -variable $this-search-path -value 0
	label $search.options.path.relative.label -text "Relative" -width 9 \
	    -anchor w -just left

	pack $search.options.path.relative.button \
	    $search.options.path.relative.label -side left


	pack $search.options.path.absolute $search.options.path.relative -side top

	button $search.options.search -text " Search " \
	    -command "$this-c search"


	pack $search.options.path -padx 5 -side left
	pack $search.options.search -side left
	pack $search.box $search.options -side left

	pack $w.search -side top -fill both -expand yes -pady 10


#       Selected Data
	iwidgets::labeledframe $w.sd -labeltext "Selected Data"

	set sd [$w.sd childsite]

	frame $sd.title
	label $sd.title.check  -text ""       -width  3 -relief groove
	label $sd.title.server -text "Server" -width 16 -relief groove
	label $sd.title.tree   -text "Tree"   -width 12 -relief groove
	label $sd.title.shot   -text "Shot"   -width  8 -relief groove
	label $sd.title.signal -text "Signal" -width 48 -relief groove
	label $sd.title.status -text "Status" -width  8 -relief groove
	label $sd.title.port   -text "Port"   -width  4 -relief groove
	label $sd.title.empty  -text ""        -width 3 -relief groove

	pack $sd.title.check \
	    $sd.title.server $sd.title.tree $sd.title.shot $sd.title.signal \
	    $sd.title.status $sd.title.port $sd.title.empty \
	    -side left 

	pack $sd.title -fill x

	iwidgets::scrolledframe $sd.entries -hscrollmode none
	pack $sd.entries -side top -fill both -expand yes

	create_entries


	frame $sd.controls
	button $sd.controls.add -text "Add New Entry" \
	    -command "$this addEntry"
	button $sd.controls.delete -text "Delete Checked Entry" \
	    -command "$this deleteEntry 0"
	button $sd.controls.deleteall -text "Delete All Entries" \
	    -command "$this deleteEntry 1"
	pack $sd.controls.add $sd.controls.delete $sd.controls.deleteall \
	    -side left -fill x -expand y

	pack $sd.controls -side top -fill both -expand yes

	pack $w.sd -side top -fill both -expand yes -pady 10

#       Data Management
	iwidgets::labeledframe $w.dm -labeltext "Data Management"
	set dm [$w.dm childsite]

	frame $dm.merge

	frame $dm.merge.none

	radiobutton $dm.merge.none.button -variable $this-mergeData -value 0
	label $dm.merge.none.label -text "No Merging" -width 12 \
	    -anchor w -just left
	
	pack $dm.merge.none.button $dm.merge.none.label -side left


	frame $dm.merge.like

	radiobutton $dm.merge.like.button -variable $this-mergeData -value 1
	label $dm.merge.like.label -text "Merge like data" -width 16 \
	    -anchor w -just left
	
	pack $dm.merge.like.button $dm.merge.like.label -side left


	frame $dm.merge.time

	radiobutton $dm.merge.time.button -variable $this-mergeData -value 2
	label $dm.merge.time.label -text "Merge time data" -width 16 \
	    -anchor w -just left
	
	pack $dm.merge.time.button $dm.merge.time.label -side left


	pack $dm.merge.none $dm.merge.like $dm.merge.time -side left

	frame $dm.svt

	checkbutton $dm.svt.button -variable $this-assumeSVT
	label $dm.svt.label -text "Assume Vector-Tensor data" \
	    -width 30 -anchor w -just left
	
	pack $dm.svt.button $dm.svt.label -side left
	pack $dm.merge $dm.svt -side left
	pack $w.dm -fill x -expand yes -side top


	global $this-power_app
	global power_app_command

	if { [set $this-power_app] } {
	    makeSciButtonPanel $w $w $this -no_execute -no_close -no_find \
		"\"Close\" \"wm withdraw $w; $power_app_command\" \"Hides this GUI\""
	} else {
	    makeSciButtonPanel $w $w $this
	}
	 
	moveToCursor $w
    }

    method create_entries {} {
	set w .ui[modname]
	if {[winfo exists $w]} {

	    set sd [$w.sd childsite]
	    set entries [$sd.entries childsite]

	    # Create the new variables and entries if needed.
	    for {set i 0} {$i < [set $this-num-entries]} {incr i} {
		
		if { [catch { set t [set $this-check-$i] } ] } {
		    set $this-check-$i [set $this-check]
		}
		if { [catch { set t [set $this-server-$i] } ] } {
		    set $this-server-$i [set $this-server]
		}
		if { [catch { set t [set $this-tree-$i]}] } {
		    set $this-tree-$i [set $this-tree]
		}
		if { [catch { set t [set $this-shot-$i]}] } {
		    set $this-shot-$i [set $this-shot]
		}
		if { [catch { set t [set $this-signal-$i]}] } {
		    set $this-signal-$i [set $this-signal]
		}
		if { [catch { set t [set $this-status-$i]}] } {
		    set $this-status-$i [set $this-status]
		}
		if { [catch { set t [set $this-port-$i]}] } {
		    set $this-port-$i [set $this-port]
		}

		if {![winfo exists $entries.e-$i]} {
		    frame $entries.e-$i
		    checkbutton $entries.e-$i.check -variable $this-check-$i
		    entry $entries.e-$i.server \
			-textvariable $this-server-$i -width 16
		    entry $entries.e-$i.tree \
			-textvariable $this-tree-$i   -width 12
		    entry $entries.e-$i.shot \
			-textvariable $this-shot-$i   -width  8
		    entry $entries.e-$i.signal \
			-textvariable $this-signal-$i -width 48
		    entry $entries.e-$i.status -state disabled \
			-textvariable $this-status-$i -width  8
		    entry $entries.e-$i.port -state disabled \
			-textvariable $this-port-$i   -width  4

		    pack $entries.e-$i.check \
			$entries.e-$i.server \
			$entries.e-$i.tree \
			$entries.e-$i.shot \
			$entries.e-$i.signal \
			$entries.e-$i.status \
			$entries.e-$i.port \
			-side left
		    pack $entries.e-$i 
		}
	    }

	    # Destroy all the left over entries from prior runs.
	    while {[winfo exists $entries.e-$i]} {
		destroy $entries.e-$i
		incr i
	    }
	}
    }

    method AddRootSignals { } {
	set w .ui[modname]
	if [ expr [winfo exists $w] ] {

	    set_watch_cursor
	    $this-c update_tree root root
	}
    }

    method addEntry {} {
	global $this-num-entries

# Save the defaults for the next new entry.
	if { [set $this-num-entries] > 0 } {
	    set i [expr [set $this-num-entries] - 1]

	    set $this-server [set $this-server-$i]
	    set $this-tree   [set $this-tree-$i]
	    set $this-shot   [set $this-shot-$i]
	}

	set i [set $this-num-entries]

# Add in the new entry using the defaults.

	set $this-check-$i 0
	set $this-server-$i [set $this-server]
	set $this-tree-$i   [set $this-tree]
	set $this-shot-$i   [set $this-shot]
	set $this-signal-$i [set $this-signal]
	set $this-status-$i [set $this-status]
	set $this-port-$i   [set $this-port]

	incr $this-num-entries

	create_entries
    }

    method deleteEntry { all } {
	set w .ui[modname]
	if {[winfo exists $w]} {

	    set sd [$w.sd childsite]
	    set entries [$sd.entries childsite]

	    global $this-num-entries

	    set j 0

	    if { $all == 0 } {
		for {set i 0} {$i < [set $this-num-entries]} {incr i} {

		    # Shift the enties in the list. 
		    if { [set $this-check-$i] == 0 } {
			set $this-check-$j 0
			set $this-server-$j [set $this-server-$i]
			set $this-tree-$j   [set $this-tree-$i]
			set $this-shot-$j   [set $this-shot-$i]
			set $this-signal-$j [set $this-signal-$i]
			set $this-status-$j [set $this-status-$i]
			set $this-port-$j   [set $this-port-$i]
			incr j
		    }
		}
	    }

	    set $this-num-entries $j

	    create_entries
	    
	    if { [set $this-num-entries] == -1 } {
		global $this-signal
		global $this-status
		global $this-port

		set $this-signal ""
		set $this-status "Unknown"
		set $this-port "na"

		addEntry
	    }
	}
    }

    method setEntry { signal } {
	set i [set $this-num-entries]

	set $this-check-$i 0
	set $this-server-$i [set $this-search-server]
	set $this-tree-$i   [set $this-search-tree]
	set $this-shot-$i   [set $this-search-shot]
	set $this-signal-$i $signal
	set $this-status-$i [set $this-status]
	set $this-port-$i   [set $this-port]
	
	incr $this-num-entries
	
	create_entries
    }


    method build_tree { filename parent } {

	global $this-dumpname
	set $this-dumpname $filename

	global $this-signals
	set $this-signals ""

	set w .ui[modname]

	if [ expr [winfo exists $w] ] {
	    set treeframe [$w.treeview childsite]
	    set treeview $treeframe.tree.tree

	    global tree

	    if { $parent == "root" } {
		$tree delete $parent
	    }

	    if {[catch {open $filename r} fileId]} {
		global read_error

		# the file may have been removed from the /tmp dir so
		# try to recreate it.
		if { $read_error == 0 } {
		    set message "Can not find "
		    append message $filename
		    $this-c error $message
		    reset_cursor
		    return
		} else {
		    set message "Can not open "
		    append message $filename
		    $this-c error $message
		    reset_cursor
		    return
		}
	    } elseif {[gets $fileId line] >= 0 &&
		      [string first MDSPlus* $line] != 1 } {
		    process_file $tree $parent $fileId $line

	    } else {
		$this-c error "Not an MDSPlus file."
		reset_cursor
		return
	    }

	    close $fileId

	    $treeview entry configure "attribute" -foreground green4
	    $treeview entry configure "signal"   -foreground cyan4
	    
	    $treeview open $parent
	    set ids [$treeview entry children $parent]
	    foreach id $ids {
		$treeview open $id
	    }

	    reset_cursor
	}
    }


    method process_file { tree parent fileId input } {

	global have_nodes
	global have_attributes
	global have_signals

	set have_nodes      0
	set have_attributes 0
	set have_signals    0
 
	while {[gets $fileId line] >= 0 && [string first "\}" $line] == -1} {

	    if { [string first "NODE" $line] != -1 } {
		if { $parent == "root" } {
		    process_node $tree $parent $fileId $line
		} else {
		    process_node $tree $parent $fileId ""
		} 
	    } else {
		$this-c error "File hierarchy mal formed."
		return
	    }
	}
    }


    method process_node { tree parent fileId input } {
	
	if { $input != "" } {
	    set name [process_name $input]
	    set info(type) @blt::tv::normalOpenFolder
	    set info(Node-Type) ""
	    set info(Data-Type) ""
	    set info(Value) ""
	    set node [$tree insert $parent -tag "node" -label $name \
			  -data [array get info]]
	} else {
	    set node $parent
	}


	global have_nodes
	set have_nodes 1

	while {[gets $fileId line] >= 0 && [string first "\}" $line] == -1} {

	    if { [string first "NODE" $line] != -1 } {
		process_node $tree $node $fileId $line
	    } elseif { [string first "ATTRIBUTE" $line] != -1 } {
		process_attribute $tree $node $fileId $line
	    } elseif { [string first "SIGNAL" $line] != -1 } {
		process_signal $tree $node $fileId $line
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

    method process_signal { tree parent fileId input } {

	if {[gets $fileId line] >= 0 && \
	    [string first "DATATYPE" $line] == -1} {
	    set message "Bad signal formation: "
	    append message $line
	    $this-c error $message
	    return
	}

	set start [string first "\"" $line]
	set end   [string  last "\"" $line]
	set type  [string range $line [expr $start+1] [expr $end-1]]

	if {[gets $fileId line] >= 0 && \
	    [string first "DATASPACE" $line] == -1} {
	    set message "Bad signal formation: "
	    append message $line
	    $this-c error $message
	    return
	}

	set start [string first "(" $line]
	set end   [string  last ")" $line]
	set dims  [string range $line $start $end]

	set dsname [process_name $input]
	set info(type) ""
	set info(Node-Type) "Signal"
	set info(Data-Type) $type
	set info(Value) $dims
	set node [$tree insert $parent -tag "signal" -label $dsname \
		      -data [array get info]]
  
	global have_signals
	set have_signals 1

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

		set id [$treeview curselection]

		if { $id != "" } {

		    global have_nodes
		    global have_attributes

		    if { $have_nodes == 1 } {
			set nodes [$treeview tag nodes "node"]
		    } else {
			set nodes ""
		    }

		    if { $have_attributes == 1 } {
			set attributes [$treeview tag nodes "attribute"]
		    } else {
			set attributes ""
		    }

		    # Check to see if the selection is an attribute
		    foreach attribute $attributes {
			if { $attribute == $id } { 
			    $treeview selection clear $id
			    break
			}
		    }
		    
		    # Check to see if the selection is a node
		    foreach node $nodes {
			if { $node == $id } { 
			    $treeview selection clear $id
			    AddChildrenSignals $id
			    break
			}
		    }		    
		}
		
		updateSelection
	    }

	    set allow_selection true
	}
    }

    method AddChildrenSignals { parent } {
	set w .ui[modname]

	if [ expr [winfo exists $w] ] {

	    set treeframe [$w.treeview childsite]
	    set treeview $treeframe.tree.tree
	    
	    set children [eval $treeview entry children $parent]

	    if { $children == "" } {

		set name [eval $treeview get -full $parent]

		# Remove any braces {}
		set pos 0
		while { $pos != -1 } {
		    set pos [string last "\{" $name]
		    set name [string replace $name $pos $pos ""]
		}
		set pos 0
		while { $pos != -1 } {
		    set pos [string last "\}" $name]
		    set name [string replace $name $pos $pos ""]
		}
		# Replace all of ' ' in the base name with '.'.
		set pos 0
		while { $pos != -1 } {
		    set pos [string last " " $name]
		    set name [string replace $name $pos $pos "."]
		}

		set_watch_cursor
		$this-c update_tree $name $parent 
	    }
	}
    }


    method updateSelection {} {

	set w .ui[modname]

	if [ expr [winfo exists $w] ] {
	    set treeframe [$w.treeview childsite]
	    set treeview $treeframe.tree.tree

	    set id [$treeview curselection]

	    if { $id != "" } {
		set signal [eval $treeview get -full $id]

		# Remove any braces {}
		set pos 0
		while { $pos != -1 } {
		    set pos [string last "\{" $signal]
		    set signal [string replace $signal $pos $pos ""]
		}
		set pos 0
		while { $pos != -1 } {
		    set pos [string last "\}" $signal]
		    set signal [string replace $signal $pos $pos ""]
		}
		# Replace all of ' ' in the base name with '.'.
		set pos 0
		while { $pos != -1 } {
		    set pos [string last " " $signal]
		    set signal [string replace $signal $pos $pos "."]
		}

		global $this-server
		global $this-tree
		global $this-shot
		global $this-signal
		global $this-status
		global $this-port

		global $this-load-server
		global $this-load-tree
		global $this-load-shot


		set $this-server [set $this-load-server]
		set $this-tree   [set $this-load-tree]
		set $this-shot   [set $this-load-shot]
		set $this-signal $signal
		set $this-status "Unkonwn"
		set $this-port "na"

		global $this-num-entries

		set add "true"

		for {set i 0} {$i < [set $this-num-entries]} {incr i} {
		    
		    # See if the signal already exists
		    if { [set $this-server-$i] == [set $this-load-server] &&
			 [set $this-tree-$i]   == [set $this-tree] &&
			 [set $this-shot-$i]   == [set $this-shot] &&
			 [set $this-signal-$i] == [set $this-signal] } {
			set add "false"
			break
		    }
		}

		if { $add == "true" } {
		    addEntry
		}
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
			 $match $path "{[set $this-selectionString]}" ]

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
#	    set sd [$w.sd childsite]
#	    set listbox $sd.listbox

	    set treeframe [$w.treeview childsite]
	    set treeview $treeframe.tree.tree

#	    set indices [$listbox.list curselection]
#	    set indices ""

	    if { [string first "\{" [set $this-signals]] == 0 } {
		set tmp [set $this-signals]
	    } else {
		set tmp "\{"
		append tmp [set $this-signals]
		append tmp "\}"
	    }
	    
	    set index 0
	    # Reselect the signals
	    foreach signal $tmp {

		foreach idx $indices {
		    if { $index == $idx } { 
			set ids [eval $treeview find -exact -full "{$signal}"]
			
			foreach id $ids {
			    if {"$id" != ""} {
				$treeview selection clear $id
				$treeview open $id
			    } else {			    
				set message "Could not find signal: "
				append message $signal
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

    method set_watch_cursor {} {
	set w .ui[modname]

	if [ expr [winfo exists $w] ] {
	    global current_cursor
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
