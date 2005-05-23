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


itcl_class CardioWave_DataIO_TimeDataReader {
    inherit Module 

    constructor {config} {
        set name TimeDataReader
        set_defaults
    }

    method set_defaults {} {   
	
		# Variables for file selector
		setGlobal $this-filename	""
		 
		# Variables for wrapper creation
		
		# Variables for nd matrix information
		
		# Variables for play buttons
		
    setGlobal $this-dimension         2
    setGlobal $this-row_or_col        column
    setGlobal $this-selectable_min    0
		setGlobal $this-selectable_max    100
		setGlobal $this-selectable_inc    1
		setGlobal $this-selectable_units  ""
		setGlobal $this-range_min         0
		setGlobal $this-range_max         100
		setGlobal $this-playmode          once
    setGlobal $this-current           0
		setGlobal $this-execmode          init
    setGlobal $this-delay             0
		setGlobal $this-inc-amount        1
		setGlobal $this-send-amount       1
		trace variable $this-current w "update idletasks;\#"
		setGlobal $this-scrollbar         ""
    setGlobal $this-cur               ""
		setGlobal $this-filename          ""
		setGlobal $this-filename-set      ""
		setGlobal $this-filename-entry    ""		
      
    }
	
	method maybeRestart { args } {
		upvar \#0 $this-execmode execmode
		if ![string equal $execmode play] return
		$this-c needexecute
    }
	
    method Restart { args } {
          $this-c needexecute
    }
	
    method update_range { args } {
  
    global $this-scrollbar
    global $this-cur
    global $this-dimension
		
    set w .ui[modname]
    if {[winfo exists $w]} {
		
      set wroc $w.loopframe.f1.roc
      destroy $wroc.r
        
      # Create and pack the Row of Column frame
      frame $wroc.r 
      pack $wroc.r -fill both -expand yes
      
      if [expr {[set $this-dimension] == 2}] then {
        radiobutton $wroc.r.row -text "Row" -variable $this-row_or_col -value row -command "set $this-execmode update; $this-c needexecute"
        radiobutton $wroc.r.col -text "Column" -variable $this-row_or_col -value column -command "set $this-execmode update; $this-c needexecute"
        pack $wroc.r.row $wroc.r.col -side left -expand yes -fill both
      }

      if [expr {[set $this-dimension] == 3}] then {
        radiobutton $wroc.r.x -text "X" -variable $this-row_or_col -value x -command "set $this-execmode update; $this-c needexecute"
        radiobutton $wroc.r.y -text "Y" -variable $this-row_or_col -value y -command "set $this-execmode update; $this-c needexecute"
        radiobutton $wroc.r.z -text "Z" -variable $this-row_or_col -value z -command "set $this-execmode update; $this-c needexecute"
        pack $wroc.r.x $wroc.r.y $wroc.r.y -side left -expand yes -fill both
      }

      if [expr {[set $this-dimension] == 4}] then {
        radiobutton $wroc.r.x -text "X" -variable $this-row_or_col -value x -command "set $this-execmode update; $this-c needexecute"
        radiobutton $wroc.r.y -text "Y" -variable $this-row_or_col -value y -command "set $this-execmode update; $this-c needexecute"
        radiobutton $wroc.r.z -text "Z" -variable $this-row_or_col -value z -command "set $this-execmode update; $this-c needexecute"
        radiobutton $wroc.r.t -text "T" -variable $this-row_or_col -value t -command "set $this-execmode update; $this-c needexecute"
        pack $wroc.r.x $wroc.r.y $wroc.r.y $wroc.r.t -side left -expand yes -fill both
      }
      
      set scrollbar [set $this-scrollbar]
      upvar \#0 $this-selectable_min min $this-selectable_max max 
      upvar \#0 $this-selectable_units units $this-row_or_col roc
      set pre $roc		
      
      $scrollbar.min configure -label "Start $pre:" -from $min -to $max
      $scrollbar.cur config -label "Current $pre:" -from $min -to $max
      $scrollbar.max config -label "End $pre:" -from $min -to $max
      $scrollbar.inc config -label "Increment current $pre by:" -from 1 -to [expr $max-$min]

    }
		
    set w [format "%s-control" .ui[modname]]
    if {[winfo exists $w]} {
    
      set scrollbar [set $this-cur]
      upvar \#0 $this-selectable_min min $this-selectable_max max 
      upvar \#0 $this-selectable_units units $this-row_or_col roc
      set pre $roc		
			
      $scrollbar.cur config -label "Current $pre:" -from $min -to $max	
        }
    }
	
	
    method ui {} {
    
    global $this-dimension
    global $this-scrollbar
    global $this-filename
    global $this-filename-entry
    global $this-execmode
    global $this-current
    global $this-range_max
    
    set w .ui[modname]
		
		# test whether gui is already out there
		# raise will move the window to the front
		# so the user can modify the settings

    if {[winfo exists $w]} {
			raise $w
      return
    }
    toplevel $w

    frame $w.fileframe  -relief groove -borderwidth 2
    set fileframe $w.fileframe 
    pack $w.fileframe -fill x

    frame $fileframe.f1
    frame $fileframe.f2
		
    pack $fileframe.f1 $fileframe.f2 -side top -fill x -expand yes

    label $fileframe.f1.label -text "Time series file"
    entry $fileframe.f1.file -textvariable $this-filename
    button $fileframe.f1.browse -text "Browse" -command "$this ChooseTimeDataFile"
    set $this-filename-entry $fileframe.f1.file  
    pack $fileframe.f1.label -side left
    pack $fileframe.f1.file  -side left -fill x -expand yes
    pack $fileframe.f1.browse -side left
    
    frame $w.infoframe 
    set infoframe $w.infoframe 
    pack $w.infoframe -fill x

    frame $w.loopframe -relief groove -borderwidth 2 
    set loopframe $w.loopframe
    pack $w.loopframe -fill x

    frame $loopframe.f1
    frame $loopframe.f2 
    pack $loopframe.f1 -side left -anchor n
    pack $loopframe.f2 -side left -anchor n -fill x  -expand yes 
    frame $loopframe.f1.playmode -relief groove -borderwidth 2
    frame $loopframe.f1.vcr -relief groove -borderwidth 2
    frame $loopframe.f1.roc -relief groove -borderwidth 2
    frame $loopframe.f1.detach -relief groove -borderwidth 2
    frame $loopframe.f2.scrollbar -relief groove -borderwidth 2
    pack $loopframe.f1.vcr $loopframe.f1.roc $loopframe.f1.playmode $loopframe.f1.detach -side top -anchor w -fill x 
    pack $loopframe.f2.scrollbar -side top -anchor w -fill x -expand yes 
                    
    set playmode $loopframe.f1.playmode
    set vcr $loopframe.f1.vcr
    set roc $loopframe.f1.roc
    set scrollbar $loopframe.f2.scrollbar
    set detach $loopframe.f1.detach
    set $this-scrollbar $scrollbar

    button $detach.open -text "Open small control window" -command "$this OpenSmall"
    pack $detach.open 

    # load the VCR button bitmaps
    set image_dir [netedit getenv SCIRUN_SRCDIR]/pixmaps
    set rewind [image create photo -file ${image_dir}/rewind-icon.ppm]
    set stepb [image create photo -file ${image_dir}/step-back-icon.ppm]
    set pause [image create photo -file ${image_dir}/pause-icon.ppm]
    set play [image create photo -file ${image_dir}/play-icon.ppm]
    set stepf [image create photo -file ${image_dir}/step-forward-icon.ppm]
    set fforward [image create photo -file ${image_dir}/fast-forward-icon.ppm]

    # Create and pack the Row of Column frame
    frame $roc.r 
    pack $roc.r -fill both -expand yes
    
    if [expr {[set $this-dimension] == 2}] then {
      radiobutton $roc.r.row -text "Row" -variable $this-row_or_col -value row -command "set $this-execmode update; $this-c needexecute"
      radiobutton $roc.r.col -text "Column" -variable $this-row_or_col -value column -command "set $this-execmode update; $this-c needexecute"
      pack $roc.r.row $roc.r.col -side left -expand yes -fill both
    }

    if [expr {[set $this-dimension] == 3}] then {
      radiobutton $roc.r.x -text "X" -variable $this-row_or_col -value x -command "set $this-execmode update; $this-c needexecute"
      radiobutton $roc.r.y -text "Y" -variable $this-row_or_col -value y -command "set $this-execmode update; $this-c needexecute"
      radiobutton $roc.r.z -text "Z" -variable $this-row_or_col -value z -command "set $this-execmode update; $this-c needexecute"
      pack $roc.r.x $roc.r.y $roc.r.y -side left -expand yes -fill both
    }

    if [expr {[set $this-dimension] == 4}] then {
      radiobutton $roc.r.x -text "X" -variable $this-row_or_col -value x -command "set $this-execmode update; $this-c needexecute"
      radiobutton $roc.r.y -text "Y" -variable $this-row_or_col -value y -command "set $this-execmode update; $this-c needexecute"
      radiobutton $roc.r.z -text "Z" -variable $this-row_or_col -value z -command "set $this-execmode update; $this-c needexecute"
      radiobutton $roc.r.t -text "T" -variable $this-row_or_col -value t -command "set $this-execmode update; $this-c needexecute"
      pack $roc.r.x $roc.r.y $roc.r.y $roc.r.t -side left -expand yes -fill both
    }
    
    
    # Create and pack the VCR buttons frame
    button $vcr.rewind -image $rewind -command "set $this-execmode rewind;   $this-c needexecute"
    button $vcr.stepb -image $stepb -command "set $this-execmode stepb;    $this-c needexecute"
    button $vcr.pause -image $pause -command "set $this-execmode stop;     $this-c needexecute"
    button $vcr.play  -image $play  -command "set $this-execmode play;     $this-c needexecute"
    button $vcr.stepf -image $stepf -command "set $this-execmode step;     $this-c needexecute"
    button $vcr.fforward -image $fforward -command "set $this-execmode fforward; $this-c needexecute"
		
    pack $vcr.rewind $vcr.stepb $vcr.pause $vcr.play $vcr.stepf $vcr.fforward -side left -fill both -expand 1
    global ToolTipText
    Tooltip $vcr.rewind $ToolTipText(VCRrewind)
    Tooltip $vcr.stepb $ToolTipText(VCRstepback)
    Tooltip $vcr.pause $ToolTipText(VCRpause)
    Tooltip $vcr.play $ToolTipText(VCRplay)
    Tooltip $vcr.stepf $ToolTipText(VCRstepforward)
    Tooltip $vcr.fforward $ToolTipText(VCRfastforward)

    # Save range, creating the scale resets it to defaults.
    set rmin [set $this-range_min]
    set rmax [set $this-range_max]

    # Create the various range sliders
    scale $scrollbar.min -variable $this-range_min -length 200 -showvalue true -orient horizontal -relief groove -command "$this maybeRestart"
    scale $scrollbar.cur -variable $this-current -length 200 -showvalue true -orient horizontal -relief groove -command "$this maybeRestart"
    bind $scrollbar.cur <ButtonRelease> "$this Restart"
    scale $scrollbar.max -variable $this-range_max -length 200 -showvalue true -orient horizontal -relief groove -command "$this maybeRestart"
    scale $scrollbar.inc -variable $this-inc-amount -length 200 -showvalue true -orient horizontal -relief groove -command "$this maybeRestart"
    pack $scrollbar.min $scrollbar.cur $scrollbar.max $scrollbar.inc -side top -anchor w -fill x -expand yes
    update_range

    # Restore range to pre-loaded value
    set $this-range_min $rmin
    set $this-range_max $rmax


    # Create and pack the play mode frame
    label $playmode.label -text "Play Mode"
    radiobutton $playmode.once -text "Once" -variable $this-playmode -value once -command "$this maybeRestart"
    radiobutton $playmode.loop -text "Loop" -variable $this-playmode -value loop -command "$this maybeRestart"
    radiobutton $playmode.bounce1 -text "Bounce" -variable $this-playmode -value bounce1 -command "$this maybeRestart"
    radiobutton $playmode.bounce2 -text "Bounce with repeating endpoints" -variable $this-playmode -value bounce2 -command "$this maybeRestart"
    iwidgets::spinint $playmode.delay -labeltext {Step Delay (ms)} -range {0 86400000} -justify right -width 5 -step 10 -textvariable $this-delay -repeatdelay 300 -repeatinterval 10
    trace variable $this-delay w "$this maybeRestart;\#"

    pack $playmode.label -side top -expand yes -fill both
    pack $playmode.once $playmode.loop $playmode.bounce1 $playmode.bounce2 $playmode.delay -side top -anchor w

    makeSciButtonPanel $w $w $this 

    moveToCursor $w
    }


	method OpenSmall {} {
	
		global $this-current
		global $this-cur
	
		# Create a unique name for the file selection window
		set w [format "%s-control" .ui[modname]]

		# if the file selector is open, bring it to the front
		# in case it is iconified, deiconify
		if { [winfo exists $w] } {
   		if { [winfo ismapped $w] == 1} {
        raise $w
   		} else {
        wm deiconify $w
   		}
   		return
		}

    upvar \#0 $this-selectable_min min $this-selectable_max max 
    upvar \#0 $this-selectable_units units 

		toplevel $w -class TkFDialog
				
		frame $w.vcr -relief groove -borderwidth 2		
		frame $w.cur -relief groove -borderwidth 2		
		set vcr $w.vcr
		set cur $w.cur
		
		pack $w.vcr $w.cur -side top
		
		# load the VCR button bitmaps
		set image_dir [netedit getenv SCIRUN_SRCDIR]/pixmaps
		set rewind [image create photo -file ${image_dir}/rewind-icon.ppm]
		set stepb [image create photo -file ${image_dir}/step-back-icon.ppm]
		set pause [image create photo -file ${image_dir}/pause-icon.ppm]
		set play [image create photo -file ${image_dir}/play-icon.ppm]
		set stepf [image create photo -file ${image_dir}/step-forward-icon.ppm]
		set fforward [image create photo -file ${image_dir}/fast-forward-icon.ppm]

		# Create and pack the VCR buttons frame
    button $vcr.rewind -image $rewind -command "set $this-execmode rewind;   $this-c needexecute"
    button $vcr.stepb -image $stepb -command "set $this-execmode stepb;    $this-c needexecute"
    button $vcr.pause -image $pause -command "set $this-execmode stop;     $this-c needexecute"
    button $vcr.play  -image $play  -command "set $this-execmode play;     $this-c needexecute"
    button $vcr.stepf -image $stepf -command "set $this-execmode step;     $this-c needexecute"
    button $vcr.fforward -image $fforward -command "set $this-execmode fforward; $this-c needexecute"
		
		pack $vcr.rewind $vcr.stepb $vcr.pause $vcr.play $vcr.stepf $vcr.fforward -side left -fill both -expand 1
		global ToolTipText
		Tooltip $vcr.rewind $ToolTipText(VCRrewind)
		Tooltip $vcr.stepb $ToolTipText(VCRstepback)
		Tooltip $vcr.pause $ToolTipText(VCRpause)
		Tooltip $vcr.play $ToolTipText(VCRplay)
		Tooltip $vcr.stepf $ToolTipText(VCRstepforward)
		Tooltip $vcr.fforward $ToolTipText(VCRfastforward)
		
		set $this-cur $cur
		scale $cur.cur -variable $this-current -length 200 -showvalue true -orient horizontal -relief groove -command "$this MaybeRestart"
    bind $cur.cur <ButtonRelease> "$this Restart"
		pack $cur.cur	
	
		moveToCursor $w
	}


	method ChooseTimeDataFile { } {

		global env
		global $this-filename
		global $this-filename-set

		# Create a unique name for the file selection window
		set w [format "%s-filebox" .ui[modname]]

		# if the file selector is open, bring it to the front
		# in case it is iconified, deiconify
		if { [winfo exists $w] } {
	    		if { [winfo ismapped $w] == 1} {
				raise $w
	    		} else {
				wm deiconify $w
	    		}
	    		return
		}

		toplevel $w -class TkFDialog

		set initdir ""
	
		# place to put preferred data directory
		# it's used if $this-filename is empty
	
		# Use the standard data dirs
		# I guess there is no .mat files in there
		# at least not yet

		if {[info exists env(SCIRUN_DATA)]} {
	    		set initdir $env(SCIRUN_DATA)
		} elseif {[info exists env(SCI_DATA)]} {
	    		set initdir $env(SCI_DATA)
		} elseif {[info exists env(PSE_DATA)]} {
	    		set initdir $env(PSE_DATA)
		}
	
		makeOpenFilebox \
			-parent $w \
			-filevar $this-filename-set \
			-command "wm withdraw $w;  $this OpenNewTimeDatafile" \
 			-commandname "Open" \
			-cancel "wm withdraw $w" \
			-title "Select timeseries data file" \
			-filetypes {{ "Header file for raw data" "*.nhdr" } }\
			-initialdir $initdir \
			-defaultextension "*.*" \
			-selectedfiletype 0

		wm deiconify $w	
	}
	
	method OpenTimeDatafile {} {

		global $this-filename
		global $this-filename-entry
		
		set $this-filename [[set $this-filename-entry] get] 
		
	}

	method OpenNewTimeDatafile {} {

		global $this-filename
		global $this-filename-set
		global $this-filename-entry
		
		set $this-filename [set $this-filename-set] 
		
	}

}


	
	
	
	
	
	

	