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
	global $this-num-entries
	set $this-num-entries 1
  
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

	global $this-mergeData
	global $this-assumeSVT

	set $this-mergeData 1
	set $this-assumeSVT 1
    }

    method ui {} {
	global $this-server
	global $this-tree
	global $this-shot
	global $this-signal

	global $this-mergeData
	global $this-assumeSVT

        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }

        toplevel $w


	iwidgets::scrolledframe $w.entries -hscrollmode none

	frame $w.title
	label $w.title.check  -text ""       -width  3 -relief groove
	label $w.title.server -text "Server" -width 24 -relief groove
	label $w.title.tree   -text "Tree"   -width 12 -relief groove
	label $w.title.shot   -text "Shot"   -width  8 -relief groove
	label $w.title.signal -text "Signal" -width 32 -relief groove
	label $w.title.status -text "Status" -width  8 -relief groove
	label $w.title.port   -text "Port"   -width  8 -relief groove
	label $w.title.empty  -text ""        -width 3 -relief groove

	pack $w.title.check \
	    $w.title.server $w.title.tree $w.title.shot $w.title.signal \
	    $w.title.status $w.title.port $w.title.empty \
	    -side left 

	pack $w.title  -fill x
	pack $w.entries -side top -fill both -expand yes

	create_entries


	frame $w.controls
	button $w.controls.add -text "Add Entry" \
	    -command "$this addEntry"
	button $w.controls.delete -text "Delete Entry" \
	    -command "$this deleteEntry"
	pack $w.controls.add $w.controls.delete \
	    -side left -fill x -expand y

	pack $w.controls -side top -fill both -expand yes -pady 10



	iwidgets::labeledframe $w.search -labeltext "Search Selection"
	set search [$w.search childsite]

	global $this-search-server
	global $this-search-tree
	global $this-search-shot
	global $this-search-signal
	global $this-search-regexp


	frame $search.box

	frame $search.box.title
	label $search.box.title.server -text "Server" -width 24 -relief groove
	label $search.box.title.tree   -text "Tree"   -width 12 -relief groove
	label $search.box.title.shot   -text "Shot"   -width  8 -relief groove
	label $search.box.title.signal -text "Signal" -width 32 -relief groove
	label $search.box.title.blank  -text ""       -width 22

	pack $search.box.title.server $search.box.title.tree \
	    $search.box.title.shot $search.box.title.signal \
	    -side left

	pack $search.box.title

	frame $search.box.entry
	entry $search.box.entry.server -textvariable $this-search-server \
	    -width 24
	entry $search.box.entry.tree   -textvariable $this-search-tree   \
	    -width 12
	entry $search.box.entry.shot   -textvariable $this-search-shot   \
	    -width  8
	entry $search.box.entry.signal -textvariable $this-search-signal \
	    -width 32

	pack $search.box.entry.server $search.box.entry.tree \
	    $search.box.entry.shot $search.box.entry.signal \
	    $search.box.title.blank \
	    -side left 

	pack $search.box.title $search.box.entry -side top \
	     -fill both -expand yes


	frame $search.options

	frame $search.options.regexp

	checkbutton $search.options.regexp.button -variable $this-search-regexp
	label $search.options.regexp.label -text "Reg-Exp" -width 10 \
	    -anchor w -just left
	pack $search.options.regexp.button $search.options.regexp.label \
	    -side left

	button $search.options.search -text " Search " \
	    -command "$this-c search"

	pack $search.options.regexp $search.options.search -side top

	pack  $search.box $search.options -side left



	pack $w.search -fill x -expand yes -side top


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

	frame $w.misc
	button $w.misc.dismiss -text Dismiss -command "destroy $w"
	button $w.misc.execute -text "Download" -command "$this-c needexecute"

	pack $w.misc.execute $w.misc.dismiss -side left -padx 10
	pack $w.misc -pady 10
    }

    method create_entries {} {
	set w .ui[modname]
	if {[winfo exists $w]} {

	    set entries [$w.entries childsite]

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
			-textvariable $this-server-$i -width 24
		    entry $entries.e-$i.tree \
			-textvariable $this-tree-$i   -width 12
		    entry $entries.e-$i.shot \
			-textvariable $this-shot-$i   -width  8
		    entry $entries.e-$i.signal \
			-textvariable $this-signal-$i -width 32
		    entry $entries.e-$i.status -state disabled \
			-textvariable $this-status-$i -width  8
		    entry $entries.e-$i.port -state disabled \
			-textvariable $this-port-$i   -width  8

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

    method deleteEntry {} {
	set w .ui[modname]
	if {[winfo exists $w]} {

	    set entries [$w.entries childsite]

	    global $this-num-entries

	    set j 0

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

	    set $this-num-entries $j

	    create_entries
	    
	    if { [set $this-num-entries] == 0 } {
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
}
