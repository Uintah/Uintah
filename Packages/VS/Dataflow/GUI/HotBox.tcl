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
    set $this-anatomydatasource ""
    set $this-adjacencydatasource ""
  }

  method ui {} {
    set w .ui[modname]
    if { [winfo exists $w] } {
      raise $w
      return
    }

    toplevel $w

    # set initdir ""
                                                                                
    # place to put preferred data directory
    # it's used if $this-filename is empty
                                                                                
    # if {[info exists env(SCIRUN_DATA)]} {
    #   set initdir $env(SCIRUN_DATA)
    # } elseif {[info exists env(SCI_DATA)]} {
    #   set initdir $env(SCI_DATA)
    # } elseif {[info exists env(PSE_DATA)]} {
    #   set initdir $env(PSE_DATA)
    # }

    # extansion to append if no extension supplied by user
    # set defext ".csv"
    # set title "Open LabelMap file"

    # file types to appers in filter box
    # set types {
    #     {{Field File}     {.csv}      }
    #     {{All Files} {.*}   }
    # }
                                                                                
    ######################################################
                                                                                
    # makeOpenFilebox \
    #     -parent $w \
    #     -filevar $this-datasource \
    #     -command "$this-c needexecute; wm withdraw $w" \
    #     -cancel "wm withdraw $w" \
    #     -title $title \
    #     -filetypes $types \
    #     -initialdir $initdir \
    #     -defaultextension $defext

    frame $w.f
    frame $w.f.row1
    entry $w.f.row1.nw  -width 10 -textvariable $this-gui_label1
    entry $w.f.row1.n   -width 10 -textvariable $this-gui_label2
    entry $w.f.row1.ne  -width 10 -textvariable $this-gui_label3
    frame $w.f.row2
    entry $w.f.row2.west -width 10 -textvariable $this-gui_label4
    entry $w.f.row2.c   -width 10 -textvariable $this-gui_label5
    entry $w.f.row2.e   -width 10 -textvariable $this-gui_label6
    frame $w.f.row3
    entry $w.f.row3.sw  -width 10 -textvariable $this-gui_label7
    entry $w.f.row3.s   -width 10 -textvariable $this-gui_label8
    entry $w.f.row3.se  -width 10 -textvariable $this-gui_label9

    pack $w.f.row1 $w.f.row2 $w.f.row3 -side top -anchor w
    pack $w.f.row1.nw $w.f.row1.n $w.f.row1.ne\
	-side left -anchor n -expand yes -fill x
    pack $w.f.row2.west $w.f.row2.c $w.f.row2.e\
	-side left -anchor n -expand yes -fill x
    pack $w.f.row3.sw $w.f.row3.s $w.f.row3.se\
        -side left -anchor n -expand yes -fill x

    frame $w.controls
    button $w.controls.reset -text "Reset" -command ""

    button $w.controls.close -text "Close" -command "destroy $w"
    pack $w.controls.reset $w.controls.close -side left -expand yes -fill x

    pack $w.f $w.controls -side top -expand yes -fill both -padx 5 -pady 5
# pack $w.title -side top
  }
}
