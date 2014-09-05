###############################################################################
# File: SCIRun/src/Packages/VS/Dataflow/GUI/HotBox.tcl
#
# Description: TCL UI specification for the HotBox in SCIRun.
#
# Author: Stewart Dickson
###############################################################################


itcl_class VS_DataFlow_HotBox {
  inherit Module
  
  constructor { config } {
    set name HotBox

    set_defaults
  }

  method set_defaults {} {
    global $this-gui_name
    global $this-gui_label(1)
    global $this-gui_label(2)
    global $this-gui_label(3)
    global $this-gui_label(4)
    global $this-gui_label(5)
    global $this-gui_label(6)
    global $this-gui_label(7)
    global $this-gui_label(8)
    global $this-gui_label(9)
    global $this-gui_is_injured(1)
    global $this-gui_is_injured(2)
    global $this-gui_is_injured(3)
    global $this-gui_is_injured(4)
    global $this-gui_is_injured(5)
    global $this-gui_is_injured(6)
    global $this-gui_is_injured(7)
    global $this-gui_is_injured(8)
    global $this-gui_is_injured(9)
    global $this-gui_parent(0)
    global $this-gui_parent(1)
    global $this-gui_parent(2)
    global $this-gui_parent(3)
    global $this-gui_parent(4)
    global $this-gui_parent(5)
    global $this-gui_parent(6)
    global $this-gui_parent(7)
    global $this-gui_parent_list
    global $this-gui_parlist_name
    global $this-gui_sibling(0)
    global $this-gui_sibling(1)
    global $this-gui_sibling(2)
    global $this-gui_sibling(3)
    global $this-gui_sibling_list
    global $this-gui_siblist_name
    global $this-gui_child(0)
    global $this-gui_child(1)
    global $this-gui_child(2)
    global $this-gui_child(3)
    global $this-gui_child(4)
    global $this-gui_child(5)
    global $this-gui_child(6)
    global $this-gui_child(7)
    global $this-gui_child(8)
    global $this-gui_child(9)
    global $this-gui_child(10)
    global $this-gui_child(11)
    global $this-gui_child(12)
    global $this-gui_child(13)
    global $this-gui_child(14)
    global $this-gui_child(15)
    global $this-gui_child_list
    global $this-gui_childlist_name
    global $this-FME_on
    global $this-Files_on
    global $this-Ontology_on
    global $this-enableDraw
    global $this-currentselection
    global $this-currentTime
    # values: "fromHotBoxUI", "UIsetProbeLoc" or "fromProbe"
    global $this-selectionsource
    global $this-datafile
    # In: HotBox.cc: #define VS_DATASOURCE_OQAFMA 1
    #                #define VS_DATASOURCE_FILES 2
    global $this-datasource
    global $this-anatomydatasource
    global $this-adjacencydatasource
    global $this-boundingboxdatasource
    global $this-injurylistdatasource
    global $this-oqafmadatasource
    global $this-geometrypath
    global $this-hipvarpath
    global $this-querytype
    # The Probe Widget UI
    global $this-gui_probeLocx
    global $this-gui_probeLocy
    global $this-gui_probeLocz
    global $this-gui_probe_scale
    global $this-selnameloc

    set $this-gui_name $this
    set $this-gui_label(1) "------"
    set $this-gui_label(2) "------"
    set $this-gui_label(3) "------"
    set $this-gui_label(4) "------"
    set $this-gui_label(5) "------"
    set $this-gui_label(6) "------"
    set $this-gui_label(7) "------"
    set $this-gui_label(8) "------"
    set $this-gui_label(9) "------"
    set $this-gui_is_injured(1) "0"
    set $this-gui_is_injured(2) "0"
    set $this-gui_is_injured(3) "0"
    set $this-gui_is_injured(4) "0"
    set $this-gui_is_injured(5) "0"
    set $this-gui_is_injured(6) "0"
    set $this-gui_is_injured(7) "0"
    set $this-gui_is_injured(8) "0"
    set $this-gui_is_injured(9) "0"
    set $this-gui_parent(0) ""
    set $this-gui_parent(1) ""
    set $this-gui_parent(2) ""
    set $this-gui_parent(3) ""
    set $this-gui_parent(4) ""
    set $this-gui_parent(5) ""
    set $this-gui_parent(6) ""
    set $this-gui_parent(7) ""
    set $this-gui_parent_list [list [set $this-gui_parent(0)] \
                                    [set $this-gui_parent(1)] \
                                    [set $this-gui_parent(2)] \
                                    [set $this-gui_parent(3)] \
                                    [set $this-gui_parent(4)] \
                                    [set $this-gui_parent(5)] \
                                    [set $this-gui_parent(6)] \
                                    [set $this-gui_parent(7)]]
    # this holds the name of this instance of the list
    set $this-gui_parlist_name $this-gui_parent_list
    set $this-gui_sibling(0) ""
    set $this-gui_sibling(1) ""
    set $this-gui_sibling(2) ""
    set $this-gui_sibling(3) ""
    set $this-gui_sibling_list [list [set $this-gui_sibling(0)] \
                                     [set $this-gui_sibling(1)] \
                                     [set $this-gui_sibling(2)] \
                                     [set $this-gui_sibling(3)]]
    # this holds the name of this instance of the list
    set $this-gui_siblist_name $this-gui_sibling_list
    set $this-gui_child(0) ""
    set $this-gui_child(1) ""
    set $this-gui_child(2) ""
    set $this-gui_child(3) ""
    set $this-gui_child(4) ""
    set $this-gui_child(5) ""
    set $this-gui_child(6) ""
    set $this-gui_child(7) ""
    set $this-gui_child(8) ""
    set $this-gui_child(9) ""
    set $this-gui_child(10) ""
    set $this-gui_child(11) ""
    set $this-gui_child(12) ""
    set $this-gui_child(13) ""
    set $this-gui_child(14) ""
    set $this-gui_child(15) ""
    set $this-gui_child_list [list [set $this-gui_child(0)] \
                                   [set $this-gui_child(1)] \
                                   [set $this-gui_child(2)] \
                                   [set $this-gui_child(3)] \
                                   [set $this-gui_child(4)] \
                                   [set $this-gui_child(5)] \
                                   [set $this-gui_child(6)] \
                                   [set $this-gui_child(7)] \
                                   [set $this-gui_child(8)] \
                                   [set $this-gui_child(9)] \
                                   [set $this-gui_child(10)] \
                                   [set $this-gui_child(11)] \
                                   [set $this-gui_child(12)] \
                                   [set $this-gui_child(13)] \
                                   [set $this-gui_child(14)] \
                                   [set $this-gui_child(15)]]
    # this holds the name of this instance of the list
    set $this-gui_childlist_name $this-gui_child_list
    set $this-FME_on "no"
    set $this-Files_on "no"
    set $this-Ontology_on "no"
    set $this-enableDraw "no"
    set $this-currentselection ""
    set $this-currentTime "1"
    set $this-selectionsource "UIsetProbeLoc"
    set $this-datafile ""
    set $this-datasource "2"
    set $this-anatomydatasource ""
    set $this-adjacencydatasource ""
    set $this-boundingboxdatasource ""
    set $this-injurylistdatasource ""
    set $this-oqafmadatasource ""
    set $this-geometrypath ""
    set $this-querytype "2"
    set $this-gui_probeLocx "0."
    set $this-gui_probeLocy "0."
    set $this-gui_probeLocz "0."
    set $this-gui_probe_scale "1.0"
    set $this-selnameloc "unknown, 0., 0., 0." 

  }
  # end method set_defaults

  method launch_filebrowser { whichdatasource } {
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

    # extansion to append if no extension supplied by user
    set defext ".csv"

    # file types to appers in filter box
    set types {
        {{Field File}     {.csv}      }
        {{Field File}     {.xml}      }
        {{All Files} {.*}   }
    }
                                                                                
    ######################################################
    # Create a new top-level window for the file browser
                                                                                
    if {$whichdatasource == "anatomy"} {
    set title "Open MasterAntomy file"
    set $this-anatomydatasource [ tk_getOpenFile \
        -title $title \
        -filetypes $types \
        -initialdir $initdir \
        -defaultextension $defext ]
    if { [set $this-anatomydatasource] != "" } {
         $this-c needexecute
       }
    } elseif {$whichdatasource == "adjacency"} {
    set title "Open Adjacency Map file"
    set $this-adjacencydatasource [ tk_getOpenFile \
        -title $title \
        -filetypes $types \
        -initialdir $initdir \
        -defaultextension $defext ]
    if { [set $this-adjacencydatasource ] != "" } {
         $this-c needexecute
       }
    } elseif {$whichdatasource == "boundingbox"} {
    set title "Open Bounding Box file"
    set $this-boundingboxdatasource [ tk_getOpenFile \
        -title $title \
        -filetypes $types \
        -initialdir $initdir \
        -defaultextension $defext ]
    if { [set $this-boundingboxdatasource ] != "" } {
         $this-c needexecute
       }
    } elseif {$whichdatasource == "injurylist"} {
    set title "Open Injury List file"
    set defext ".xml"
    set $this-injurylistdatasource [ tk_getOpenFile \
        -title $title \
        -filetypes $types \
        -initialdir $initdir \
        -defaultextension $defext ]
    if { [set  $this-injurylistdatasource] != "" } {
         $this-c needexecute
       }
    } elseif {$whichdatasource == "geometry"} {
    set title "Set Geometry path"
    set $this-geometrypath [ tk_chooseDirectory \
        -title $title \
        -initialdir $initdir ]
    if { [set  $this-geometrypath] != "" } {
         $this-c needexecute
       }
    } elseif {$whichdatasource == "HIPvars"} {
    set title "Set HIP var file path"
    set $this-hipvarpath [ tk_chooseDirectory \
        -title $title \
        -initialdir $initdir ]
    if { [set  $this-hipvarpath] != "" } {
         $this-c needexecute
       }
    }
  }
  # end method launch_filebrowser

  method toggle_FME_on {} {
    if {[set $this-FME_on] == "yes"} {
      # toggle FME control off
      set $this-FME_on "no"
    } else {set $this-FME_on "yes"}
  }
  # end method toggle_FME_on

  method toggle_Files_on {} {
    if {[set $this-Files_on] == "yes"} {
      # toggle Files control off
      set $this-Files_on "no"
    } else {set $this-Files_on "yes"}
    set w .ui[modname]
    destroy $w
    ui
  }
  # end method toggle_Files_on

  method toggle_Ontology_on {} {     
    if {[set $this-Ontology_on] == "yes"} { 
      # toggle Ontology control off
      set $this-Ontology_on "no"
    } else {set $this-Ontology_on "yes"}
    set w .ui[modname]
    destroy $w
    ui
  }
  # end method toggle_Ontology_on


  method toggle_enableDraw {} {
    if {[set $this-enableDraw] == "yes"} {
      # toggle HotBox output Geometry off
      set $this-enableDraw "no"
    } else {set $this-enableDraw "yes"}
    # re-execute the module to draw/erase graphics
    $this-c needexecute
  }
  # end method toggle_enableDraw

  method set_selection { selection_id } {
    puts "VS_DataFlow_HotBox::set_selection{$selection_id}"
    set selection "unknown"
    if { [info exists $this-gui_label($selection_id)] } {
        set selection [set $this-gui_label($selection_id)]
    } else {
        puts "VS_DataFlow_HotBox::set_selection: not found"
    }

    if {[set $this-FME_on] == "yes"} {
      # focus on the selection in the FME
      exec /usr/local/SCIRun/src/Packages/VS/Standalone/scripts/VSgetFME.p $selection
    }
    set $this-currentselection $selection
    # tell the HotBox module that the
    # current selection was changed
    # from the HotBox UI -- not the Probe
    set $this-selectionsource "fromHotBoxUI"
    # trigger the HotBox execution to reflect selection change
    $this-c needexecute
  }
  # end method set_selection

  method setProbeLoc {} {
    set $this-selectionsource "UIsetProbeLoc"
    # trigger the HotBox execution to move the probe
    $this-c needexecute
  }
  # end method setProbeLoc

  method set_probeSelection { name1 name2 op } {
    # $this-selnameloc is of the form "<selection>,<locX>,<locY>,<locZ>"
    # separate tokens at commas
    set fields [split [set $this-selnameloc] ","]
    # set current selection
    set $this-currentselection [lindex $fields 0]
    # tell the HotBox module that the
    # current selection was changed
    # from the HotBox UI -- not the Probe
    set $this-selectionsource "fromHotBoxUI"
    # set probe location
    set $this-gui_probeLocx [lindex $fields 1]
    set $this-gui_probeLocy [lindex $fields 2]
    set $this-gui_probeLocz [lindex $fields 3]
    $this setProbeLoc
  }
  # end method set_probeSelection

  method set_hier_selection { selection_id } {
    set w .ui[modname]
    if { [winfo exists $w.hier.multi.l$selection_id] } {
        set selindex [$w.hier.multi.l$selection_id curselection]
    } else { puts "VS_DataFlow_HotBox::set_hier_selection: not found" }
    if { $selection_id == 1 } {
        if { [info exists $this-gui_parent($selindex)] } {
            set $this-currentselection [set $this-gui_parent($selindex)]
        } else { puts "VS_DataFlow_HotBox::set_hier_selection: not found" }
    } elseif { $selection_id == 2 } {
        if { [info exists $this-gui_sibling($selindex)] } {
            set $this-currentselection [set $this-gui_sibling($selindex)]
        } else { puts "VS_DataFlow_HotBox::set_hier_selection: not found" }
    } elseif { $selection_id == 3 } {
        if { [info exists $this-gui_child($selindex)] } {
            set $this-currentselection [set $this-gui_child($selindex)]
        } else { puts "VS_DataFlow_HotBox::set_hier_selection: not found" }
    }
    # tell the HotBox module that the
    # current selection was changed
    # from the HotBox UI -- not the Probe
    set $this-selectionsource "fromHotBoxUI"
    # trigger the HotBox execution to reflect selection change
    $this-c needexecute
  }
  # end method set_hier_selection

  method set_data_source { datasource } {
    set $this-datasource $datasource
  }
  # end method set_data_source

  method set_querytype { querytype } {
    set $this-querytype $querytype 
  }
  # end method set_querytype

  # called by the Hierarchy Browser listboxes
  method yscroll {w args} \
  {
    set w .ui[modname]
    if {![winfo exists $w.hier.vs]} { return }
    eval [linsert $args 0 $w.hier.vs set]
    yview $w moveto [lindex [$w.hier.vs get] 0]
  }

  # called by the Hierarchy Browser scroll bar
  method yview {w args} \
  {
    set w .ui[modname]
    variable {}
    if {$($w:yview)} { return }
    set ($w:yview) 1
    foreach i {1 2 3} { eval $w.hier.multi.l$i yview $args }
    set ($w:yview) 0
  }
  # end method yview

  #############################################################################
  # method ui
  #
  # Build the HotBox UI
  #############################################################################
  method ui {} {
    set w .ui[modname]
    if { [winfo exists $w] } {
      raise $w
      return
    }

    ################################
    # show/hide the data source URIs
    ################################
    toplevel $w
    checkbutton $w.togFilesUI -text "Data Sources" -command "$this toggle_Files_on"
    ################################
    # if selected, show data sources
    ################################
    if { [set $this-Files_on] == "yes" } {
    frame $w.files
    frame $w.files.row1
    label $w.files.row1.anatomylabel -text "Anatomy Data Source: "
    entry $w.files.row1.filenamentry -textvar $this-anatomydatasource -width 50
    button  $w.files.row1.browsebutton -text "Browse..." -command "$this launch_filebrowser anatomy"

    frame $w.files.row2
    label $w.files.row2.adjacencylabel -text "Adjacency Data Source: "
    entry $w.files.row2.filenamentry -textvar $this-adjacencydatasource -width 50
    button  $w.files.row2.browsebutton -text "Browse..." -command "$this launch_filebrowser adjacency"

    frame $w.files.row3
    label $w.files.row3.boundingboxlabel -text "Bounding Box Data Source: "
    entry $w.files.row3.filenamentry -textvar $this-boundingboxdatasource -width 50
    button  $w.files.row3.browsebutton -text "Browse..." -command "$this launch_filebrowser boundingbox"

    frame $w.files.row4
    label $w.files.row4.injurylistlabel -text "Injury List Data Source: "
    entry $w.files.row4.filenamentry -textvar $this-injurylistdatasource -width 50
    button  $w.files.row4.browsebutton -text "Browse..." -command "$this launch_filebrowser injurylist"

    frame $w.files.row5
    label $w.files.row5.geompathlabel -text "Geometry Directory: "
    entry $w.files.row5.dirnamentry -textvar $this-geometrypath -width 50
    button  $w.files.row5.browsebutton -text "Browse..." -command "$this launch_filebrowser geometry"

    frame $w.files.row6
    label $w.files.row6.hippathlabel -text "HIP data Directory: "
    entry $w.files.row6.dirnamentry -textvar $this-hipvarpath -width 50
    button  $w.files.row6.browsebutton -text "Browse..." -command "$this launch_filebrowser HIPvars"

    frame $w.files.row7
    label $w.files.row7.oqafmaURIlabel -text "OQAFMA URL: "
    entry $w.files.row7.oqafmaURIentry -textvar $this-oqafmadatasource -width 50

    pack $w.files.row1 $w.files.row2 $w.files.row3 $w.files.row4 $w.files.row5 $w.files.row6 $w.files.row7 -side top -anchor w
    pack $w.files.row1.anatomylabel $w.files.row1.filenamentry\
	$w.files.row1.browsebutton\
        -side left -anchor n -expand yes -fill x
    pack $w.files.row2.adjacencylabel $w.files.row2.filenamentry\
	$w.files.row2.browsebutton\
        -side left -anchor n -expand yes -fill x
    pack $w.files.row3.boundingboxlabel $w.files.row3.filenamentry\
	$w.files.row3.browsebutton\
        -side left -anchor n -expand yes -fill x
    pack $w.files.row4.injurylistlabel $w.files.row4.filenamentry\
	$w.files.row4.browsebutton\
        -side left -anchor n -expand yes -fill x
    pack $w.files.row5.geompathlabel $w.files.row5.dirnamentry\
        $w.files.row5.browsebutton\
        -side left -anchor n -expand yes -fill x
    pack $w.files.row6.hippathlabel $w.files.row6.dirnamentry\
        $w.files.row6.browsebutton\
        -side left -anchor n -expand yes -fill x
    pack $w.files.row7.oqafmaURIlabel $w.files.row7.oqafmaURIentry\
	-side left -anchor n -expand yes -fill x
    }
    # end if { [set $this-Files_on] == "yes" }

    frame $w.f
    #############################################################
    # the UI buttons for selecting anatomical names (adjacencies)
    #############################################################
    frame $w.f.row1

#    if { [set $this-gui_is_injured(1)] == "1" } {
#    button $w.f.row1.nw -background red -textvariable $this-gui_label(1) -command "$this set_selection 1"
#    } else {
#    button $w.f.row1.nw -background gray -textvariable $this-gui_label(1) -command "$this set_selection 1"
#    }
#    if { [set $this-gui_is_injured(2)] == "1" } {
#    button $w.f.row1.n -background red -textvariable $this-gui_label(2) -command "$this set_selection 2"
#    } else {
#    button $w.f.row1.n -background gray -textvariable $this-gui_label(2) -command "$this set_selection 2"
#    }
#    if { [set $this-gui_is_injured(3)] == "1" } {
#    button $w.f.row1.ne -background red -textvariable $this-gui_label(3) -command "$this set_selection 3"
#    } else {
#    button $w.f.row1.ne -background gray -textvariable $this-gui_label(3) -command "$this set_selection 3"
#    }
#    frame $w.f.row2
#    if { [set $this-gui_is_injured(4)] == "1" } {
#    button $w.f.row2.west -background red -textvariable $this-gui_label(4) -command "$this set_selection 4"
#    } else {
#    button $w.f.row2.west -background gray -textvariable $this-gui_label(4) -command "$this set_selection 4"
#    }
#    button $w.f.row2.c  -background yellow  -textvariable $this-gui_label(5) -command "$this set_selection 5"
#    if { [set $this-gui_is_injured(6)] == "1" } {
#    button $w.f.row2.e  -background red -textvariable $this-gui_label(6) -command "$this set_selection 6"
#    } else {
#    button $w.f.row2.e -background gray -textvariable $this-gui_label(6) -command "$this set_selection 6"
#    }
#    frame $w.f.row3
#    if { [set $this-gui_is_injured(7)] == "1" } {
#    button $w.f.row3.sw -background red -textvariable $this-gui_label(7) -command "$this set_selection 7"
#    } else {
#    button $w.f.row3.sw -background gray -textvariable $this-gui_label(7) -command "$this set_selection 7"
#    }
#    if { [set $this-gui_is_injured(8)] == "1" } {
#    button $w.f.row3.s  -background red -textvariable $this-gui_label(8) -command "$this set_selection 8"
#    } else {
#    button $w.f.row3.s -background gray  -textvariable $this-gui_label(8) -command "$this set_selection 8"
#    }
#    if { [set $this-gui_is_injured(9)] == "1" } {
#    button $w.f.row3.se -background red -textvariable $this-gui_label(9) -command "$this set_selection 9"
#    } else {
#    button $w.f.row3.se -background gray -textvariable $this-gui_label(9) -command "$this set_selection 9"
#    }

    button $w.f.row1.nw   -background gray -textvariable $this-gui_label(1) -command "$this set_selection 1"
    button $w.f.row1.n    -background gray -textvariable $this-gui_label(2) -command "$this set_selection 2"
    button $w.f.row1.ne   -background gray -textvariable $this-gui_label(3) -command "$this set_selection 3"
frame $w.f.row2

    button $w.f.row2.west -background gray -textvariable $this-gui_label(4) -command "$this set_selection 4"
    button $w.f.row2.c    -background yellow  -textvariable $this-gui_label(5) -command "$this set_selection 5"
    button $w.f.row2.e    -background gray -textvariable $this-gui_label(6) -command "$this set_selection 6"
frame $w.f.row3

    button $w.f.row3.sw   -background gray -textvariable $this-gui_label(7) -command "$this set_selection 7"
    button $w.f.row3.s    -background gray -textvariable $this-gui_label(8) -command "$this set_selection 8"
    button $w.f.row3.se   -background gray -textvariable $this-gui_label(9) -command "$this set_selection 9"

    pack $w.f.row1 $w.f.row2 $w.f.row3 -side top -anchor w
    pack $w.f.row1.nw $w.f.row1.n $w.f.row1.ne\
	-side left -anchor n -expand yes -fill x
    pack $w.f.row2.west $w.f.row2.c $w.f.row2.e\
	-side left -anchor n -expand yes -fill x
    pack $w.f.row3.sw $w.f.row3.s $w.f.row3.se\
        -side left -anchor n -expand yes -fill x

    ######################################
    # Probe UI
    ######################################
    frame $w.probeUI
    frame $w.probeUI.loc
#    tk_optionMenu $w.probeUI.loc.hotlist $this-selnameloc "Pericardium,94.5,60.1,52.3" "Myocardial zone 7,96.9,60.0,71.4" "Myocardial zone 12,113.9,69.5,83.7" "Myocardial zone 12,80.3,70.3,105.0" "Upper lobe of left lung,87.8,68.3,106.6"
tk_optionMenu $w.probeUI.loc.hotlist $this-selnameloc "Human" "Pericardium,448.98, 231.04, 1447.96"  "Myocardial zone 1,412.79, 222.0, 1426.14"  "Myocardial zone 6,422.93, 212.41, 1434.0" "Myocardial zone 7,427.36, 242.22, 1434.51" "Myocardial zone 12,434.3, 229.4, 1444.6" "Porcine" "Left ventricle,68.64, 35.58, 35.47"  "Right ventricle, 52.68, 32.84, 38.40" 
#tk_optionMenu $w.probeUI.loc.hotlist $this-selnameloc "Left ventricle,68.64, 35.58, 35.47"  "Right ventricle, 52.68, 32.84, 38.40"  
    trace var $this-selnameloc w "$this set_probeSelection"
    label $w.probeUI.loc.locLabel -text "Cursor Location" -just left
    entry $w.probeUI.loc.locx -width 10 -textvariable $this-gui_probeLocx
    entry $w.probeUI.loc.locy -width 10 -textvariable $this-gui_probeLocy
    entry $w.probeUI.loc.locz -width 10 -textvariable $this-gui_probeLocz
    bind $w.probeUI.loc.locx <KeyPress-Return> "$this setProbeLoc"
    bind $w.probeUI.loc.locy <KeyPress-Return> "$this setProbeLoc"
    bind $w.probeUI.loc.locz <KeyPress-Return> "$this setProbeLoc"
#    bind $w.probeUI.loc.locz <KeyPress-Return> "$this setProbeLoc"
    pack $w.probeUI.loc.locLabel $w.probeUI.loc.locx $w.probeUI.loc.locy $w.probeUI.loc.locz $w.probeUI.loc.hotlist\
                -side left -anchor n -expand yes -fill x

    frame $w.probeUI.slideTime
    scale $w.probeUI.slideTime.slide -orient horizontal -label "Cursor Size" \
             -from 0 -to 40 -showvalue true \
             -variable $this-gui_probe_scale -resolution 0.25 -tickinterval 10
    set $w.probeUI.slideTime.slide $this-gui_probe_scale
    bind $w.probeUI.slideTime.slide <ButtonRelease> "$this-c needexecute"
    bind $w.probeUI.slideTime.slide <B1-Motion> "$this-c needexecute"

    ######################################
    # time
    ######################################
    label $w.probeUI.slideTime.timeLabel -text "Time"
    entry $w.probeUI.slideTime.timeVal -width 5 -textvariable $this-currentTime
    bind $w.probeUI.slideTime.timeVal <KeyPress-Return> "$this-c needexecute"
    pack $w.probeUI.slideTime.slide $w.probeUI.slideTime.timeLabel $w.probeUI.slideTime.timeVal -side left -expand yes -fill x
    pack $w.probeUI.slideTime $w.probeUI.loc -side bottom -expand yes -fill x

    checkbutton $w.togOntologyUI -text "Ontology" -command "$this toggle_Ontology_on"

    ######################################
    # FMA Hierarchy UI
    ######################################
    variable {}
    set ($w:yview) 0
    frame $w.hier
    frame $w.hier.titles
    frame $w.hier.multi
    canvas $w.hier.titles.l1 -width 150 -height 20
    $w.hier.titles.l1 create text 20 10 -text "Parent"
    listbox $w.hier.multi.l1 -listvariable $this-gui_parent_list \
            -yscrollc [list $this yscroll $w]
    bind $w.hier.multi.l1 <<ListboxSelect>> "$this set_hier_selection 1"
    canvas $w.hier.titles.l2 -width 150 -height 20
    $w.hier.titles.l2 create text 25 10 -text "Selection"
    listbox $w.hier.multi.l2 -listvariable $this-gui_sibling_list \
            -yscrollc [list $this yscroll $w]
    bind $w.hier.multi.l2 <<ListboxSelect>> "$this set_hier_selection 2"
    canvas $w.hier.titles.l3 -width 100 -height 20
    $w.hier.titles.l3 create text 20 10 -text "Child"
    listbox $w.hier.multi.l3 -listvariable $this-gui_child_list \
            -yscrollc [list $this yscroll $w]
    bind $w.hier.multi.l3 <<ListboxSelect>> "$this set_hier_selection 3"
    scrollbar $w.hier.vs -command [list $this yview $w]
    grid $w.hier.titles -column 0 -row 0 -sticky nsew
    grid $w.hier.multi -column 0 -row 1 -sticky nsew
    grid $w.hier.vs -column 1 -row 0 -rowspan 2 -sticky ns
    grid $w.hier.titles.l1 -column 0 -row 0 -padx 1
    grid $w.hier.titles.l2 -column 1 -row 0 -padx 1
    grid $w.hier.titles.l3 -column 2 -row 0 -padx 1 -sticky ew
    grid $w.hier.multi.l1 -column 0 -row 1 -sticky ns
    grid $w.hier.multi.l2 -column 1 -row 1 -sticky ns
    grid $w.hier.multi.l3 -column 2 -row 1 -sticky ns -sticky ewns
    grid rowconfigure    $w.hier 1 -weight 1
    grid columnconfigure $w.hier 0 -weight 1
    grid rowconfigure    $w.hier.titles 1 -weight 1
    grid columnconfigure $w.hier.titles 2 -weight 1
    grid rowconfigure    $w.hier.multi 1 -weight 1
    grid columnconfigure $w.hier.multi 2 -weight 1
    # pack $w.hier -fill both -expand 1

# There was an error here in the code I downloaded on March 3 so I
    # restored the original code form previous version of HotBox.tcl - RCW
    ### *** magic occurs here *** ###
    ### C++ HotBox Module takes control asynchronously
    ### and performs the following on selection change
    set $this-gui_parent_list [list [set $this-gui_parent(0)] \
                                    [set $this-gui_parent(1)] \
                                    [set $this-gui_parent(2)] \
                                    [set $this-gui_parent(3)] \
                                    [set $this-gui_parent(4)] \
                                    [set $this-gui_parent(5)] \
                                    [set $this-gui_parent(6)] \
                                    [set $this-gui_parent(7)]]
    set $this-gui_sibling_list [list [set $this-gui_sibling(0)] \
                                     [set $this-gui_sibling(1)] \
                                     [set $this-gui_sibling(2)] \
                                     [set $this-gui_sibling(3)]]
    set $this-gui_child_list [list [set $this-gui_child(0)] \
                                   [set $this-gui_child(1)] \
                                   [set $this-gui_child(2)] \
                                   [set $this-gui_child(3)] \
                                   [set $this-gui_child(4)] \
                                   [set $this-gui_child(5)] \
                                   [set $this-gui_child(6)] \
                                   [set $this-gui_child(7)] \
                                   [set $this-gui_child(8)] \
                                   [set $this-gui_child(9)] \
                                   [set $this-gui_child(10)] \
                                   [set $this-gui_child(11)] \
                                   [set $this-gui_child(12)] \
                                   [set $this-gui_child(13)] \
                                   [set $this-gui_child(14)] \
                                   [set $this-gui_child(15)]]
    ######################################
    # Query UI
    ######################################
    frame $w.controls
    radiobutton $w.controls.adjOQAFMA -value 1 -text "OQAFMA" -variable $this-datasource -command "$this set_data_source 1"
    radiobutton $w.controls.adjFILES -value 2 -text "Files" -variable $this-datasource -command "$this set_data_source 2"

    checkbutton $w.controls.togFME -text "Connect to FME" -command "$this toggle_FME_on"

    checkbutton $w.controls.enableDraw -text "Enable Viewer HotBox" -command "$this toggle_enableDraw"

    ######################################
    # Close the HotBox UI Window
    ######################################
#    button $w.controls.close -text "Close" -command "destroy $w" 

    button $w.close -text "Close" -command "destroy $w"
#    pack $w.controls.adjOQAFMA $w.controls.adjFILES $w.controls.togFME $w.controls.enableDraw $w.controls.close -side left -expand yes -fill x
    pack $w.controls.adjOQAFMA $w.controls.adjFILES $w.controls.togFME $w.controls.enableDraw -side left -expand yes -fill x

    frame $w.controls2
    label $w.controls2.querytypelabel -text "Query Type: "
    radiobutton $w.controls2.containsbutton -value 2 -text "Contains" -variable $this-querytype -command "$this set_querytype 2"
    radiobutton $w.controls2.partsbutton -value 3 -text "Parts" -variable $this-querytype -command "$this set_querytype 3"
    radiobutton $w.controls2.partcontainsbutton -value 4 -text "Part Contains" -variable $this-querytype -command "$this set_querytype 4"
    pack $w.controls2.querytypelabel $w.controls2.containsbutton $w.controls2.partsbutton $w.controls2.partcontainsbutton -side left -expand yes -fill x

    if { [set $this-Ontology_on] == "yes" } {
      if { [set $this-Files_on] == "yes" } {
        pack $w.togFilesUI $w.files $w.f $w.probeUI $w.togOntologyUI $w.hier $w.controls2 $w.controls $w.close -side top -expand yes -fill both -padx 5 -pady 5
      } else {
    pack $w.togFilesUI $w.f $w.probeUI $w.togOntologyUI $w.hier $w.controls2 $w.controls $w.close -side top -expand yes -fill both -padx 5 -pady 5
      }
    } else {
      if { [set $this-Files_on] == "yes" } {
        pack $w.togFilesUI $w.files $w.f $w.probeUI $w.togOntologyUI $w.close -side top -expand yes -fill both -padx 5 -pady 5
    } else {
        pack $w.togFilesUI $w.f $w.probeUI $w.togOntologyUI $w.close -side top -expand yes -fill both -padx 5 -pady 5
      } 
    }

# Replaced code - RCW April 11, 2005
#     if { [set $this-Files_on] == "yes" } {   
#       pack $w.togFilesUI $w.files $w.f $w.probeUI $w.togOntologyUI $w.hier $w.controls2 $w.controls -side top -expand yes -fill both -padx 5 -pady 5
#    } else {
#       pack $w.togFilesUI $w.f $w.probeUI $w.controls2 $w.controls -side top -expand yes -fill both -padx 5 -pady 5
#    }


# pack $w.title -side top
  }
# end method ui
}
# end itcl_class VS_DataFlow_HotBox
