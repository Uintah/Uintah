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
    global $this-gui_label1
    global $this-gui_label2
    global $this-gui_label3
    global $this-gui_label4
    global $this-gui_label5
    global $this-gui_label6
    global $this-gui_label7
    global $this-gui_label8
    global $this-gui_label9
    global $this-FME_on
    global $this-currentselection
    global $this-datafile
    global $this-datasource
    global $this-anatomydatasource
    global $this-adjacencydatasource

    set $this-gui_label1 "label1"
    set $this-gui_label2 "label2"
    set $this-gui_label3 "label3"
    set $this-gui_label4 "label4"
    set $this-gui_label5 "label5"
    set $this-gui_label6 "label6"
    set $this-gui_label7 "label7"
    set $this-gui_label8 "label8"
    set $this-gui_label9 "label9"
    set $this-FME_on "yes"
    set $this-currentselection ""
    set $this-datafile ""
    set $this-datasource ""
    set $this-anatomydatasource ""
    set $this-adjacencydatasource ""
  }
  # end method set_defaults

  method launch_filebrowser {} {
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
    # set defext ".csv"
    # set title "Open LabelMap file"

    # file types to appers in filter box
    set types {
        {{Field File}     {.csv}      }
        {{All Files} {.*}   }
    }
                                                                                
    ######################################################
                                                                                
    makeOpenFilebox \
        -parent $w \
        -filevar $this-datafile \
        -command "$this-c needexecute; wm withdraw $w" \
        -cancel "wm withdraw $w" \
        -title $title \
        -filetypes $types \
        -initialdir $initdir \
        -defaultextension $defext
  }
  # end method launch_filebrowser

  method toggle_FME_on {} {
    if {[set $this-FME_on] == "yes"} {
      # toggle FME control off
      set $this-FME_on "no"
    } else {set $this-FME_on "yes"}

  }
  # end method toggle_FME_on

  method set_selection { selection_id } {
    puts "VS_DataFlow_HotBox::set_selection{$selection_id}"
    switch $selection_id {
	1 {
            set selection [set $this-gui_label1]
	}
	2 {
            set selection [set $this-gui_label2]
	}
	3 {
            set selection [set $this-gui_label3]
	}
	4 {
            set selection [set $this-gui_label4]
	}
	5 {
            set selection [set $this-gui_label5]
	}
	6 {
            set selection [set $this-gui_label6]
	}
	7 {
            set selection [set $this-gui_label7]
	}
	8 {
            set selection [set $this-gui_label8]
	}
	9 {
            set selection [set $this-gui_label9]
	}
        default {
          puts "VS_DataFlow_HotBox::set_selection: not found"
          break
	}
    }
    # end switch $selection_id
    if {[set $this-FME_on] == "yes"} {
      # focus on the selection in the FME
      exec VSgetFME.p $selection
    }
    set $this-currentselection $selection
  }
  # end method set_selection

  method set_data_source { datasource } {
    set $this-datasource $datasource
  }
  # end method set_data_source

  method ui {} {
    set w .ui[modname]
    if { [winfo exists $w] } {
      raise $w
      return
    }

    toplevel $w

    frame $w.f
    # the UI buttons for selecting anatomical names (adjacencies)
    frame $w.f.row1
    button $w.f.row1.nw  -textvariable $this-gui_label1 -command "$this set_selection 1"
    button $w.f.row1.n   -textvariable $this-gui_label2 -command "$this set_selection 2"
    button $w.f.row1.ne  -textvariable $this-gui_label3 -command "$this set_selection 3"
    frame $w.f.row2
    button $w.f.row2.west -textvariable $this-gui_label4 -command "$this set_selection 4"
    button $w.f.row2.c    -textvariable $this-gui_label5 -command "$this set_selection 5"
    button $w.f.row2.e   -textvariable $this-gui_label6 -command "$this set_selection 6"
    frame $w.f.row3
    button $w.f.row3.sw  -textvariable $this-gui_label7 -command "$this set_selection 7"
    button $w.f.row3.s   -textvariable $this-gui_label8 -command "$this set_selection 8"
    button $w.f.row3.se  -textvariable $this-gui_label9 -command "$this set_selection 9"

    pack $w.f.row1 $w.f.row2 $w.f.row3 -side top -anchor w
    pack $w.f.row1.nw $w.f.row1.n $w.f.row1.ne\
	-side left -anchor n -expand yes -fill x
    pack $w.f.row2.west $w.f.row2.c $w.f.row2.e\
	-side left -anchor n -expand yes -fill x
    pack $w.f.row3.sw $w.f.row3.s $w.f.row3.se\
        -side left -anchor n -expand yes -fill x

    frame $w.controls
    set ::datasource "2"
    radiobutton $w.controls.adjOQAFMA -value 1 -text "OQAFMA" -variable datasource -command "$this set_data_source $::datasource"
    radiobutton $w.controls.adjFILES -value 2 -text "Files" -variable datasource -command "$this set_data_source $::datasource"

    checkbutton $w.controls.togFME -text "Connect to FME" -command "$this toggle_FME_on"
    $w.controls.togFME select

    button $w.controls.close -text "Close" -command "destroy $w"
    pack $w.controls.adjOQAFMA $w.controls.adjFILES $w.controls.togFME $w.controls.close -side left -expand yes -fill x

    pack $w.f $w.controls -side top -expand yes -fill both -padx 5 -pady 5
# pack $w.title -side top
  }
# end method ui
}
# end itcl_class VS_DataFlow_HotBox
