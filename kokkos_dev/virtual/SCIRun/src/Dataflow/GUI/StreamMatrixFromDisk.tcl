itcl_class SCIRun_DataIO_StreamMatrixFromDisk {
    inherit Module
    constructor {config} {
        set name StreamMatrixFromDisk
        set_defaults
    }

    method set_defaults {} {
    
    # Variables for file selector
    setGlobal $this-filename	""
     
    # Variables for wrapper creation
    
    # Variables for nd matrix information
    
    # Variables for play buttons
		
    setGlobal $this-row_or_col        column
    setGlobal $this-slider_min        0
    setGlobal $this-slider_max        100
    setGlobal $this-range_min         0
    setGlobal $this-range_max         100
    setGlobal $this-playmode          once
    setGlobal $this-current           0
    setGlobal $this-execmode          init
    setGlobal $this-delay             0
    setGlobal $this-inc-amount        1
    setGlobal $this-send-amount       1
    setGlobal $this-scrollbar         ""
    setGlobal $this-cur               ""
    setGlobal $this-filename          ""
    setGlobal $this-filename-set      ""
    setGlobal $this-filename-entry    ""		
    setGlobal $this-scrollbar         ""
    
    }

    method maybeRestart { args } {
      upvar \#0 $this-execmode execmode
      if [string equal $execmode play] return
      $this-c needexecute
    }


    method update_range { args } {
    
      set w .ui[modname]
      if {[winfo exists $w]} {
      
      
        set scrollbar [set $this-scrollbar]
        upvar \#0 $this-slider_min min $this-slider_max max 
        upvar \#0 $this-row_or_col roc	
        
        $scrollbar.min.slider configure -label "Start $roc:" -from $min -to $max
        $scrollbar.cur.slider config -label "Current $roc:" -from $min -to $max
        $scrollbar.max.slider config -label "End $roc:" -from $min -to $max
        $scrollbar.inc.slider config -label "Increment current $roc by:" -from 1 -to [expr $max-$min]

      }
      
      set w [format "%s-control" .ui[modname]]
      if {[winfo exists $w]} {
      
        set scrollbar [set $this-cur]
        upvar \#0 $this-slider_min min $this-slider_max max 
        set pre $roc		
        
        $scrollbar.cur config -label "Current $pre:" -from $min -to $max	
      }
    }
	


    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w

        frame $w.fileframe  -relief groove -borderwidth 2
        set fileframe $w.fileframe 
        pack $w.fileframe -fill x

        frame $fileframe.f1
        frame $fileframe.f2
        
        pack $fileframe.f1 $fileframe.f2 -side top -fill x -expand yes

        label $fileframe.f1.label -text "Matrix file"
        entry $fileframe.f1.file -textvariable $this-filename
        button $fileframe.f1.browse -text "Browse" -command "$this ChooseMatrixFile"
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
      
        radiobutton $roc.r.row -text "Row" -variable $this-row_or_col -value row -command "set $this-execmode update; $this-c needexecute"
        radiobutton $roc.r.col -text "Column" -variable $this-row_or_col -value column -command "set $this-execmode update; $this-c needexecute"
        pack $roc.r.row $roc.r.col -side left -expand yes -fill both
    
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
        set rroc [set $this-row_or_col]

        # Create the various range sliders
        frame $scrollbar.min
        frame $scrollbar.cur
        frame $scrollbar.max
        frame $scrollbar.inc
        pack $scrollbar.min $scrollbar.cur $scrollbar.max $scrollbar.inc -side top -anchor w -fill x -expand yes
    
        scale $scrollbar.min.slider -variable $this-range_min -length 200 -showvalue true -orient horizontal -relief groove 
        iwidgets::spinint $scrollbar.min.count -range {0 86400000} -justify right -width 5 -step 1 -textvariable $this-range_min -repeatdelay 300 -repeatinterval 10 -command "$this maybeRestart"
        pack $scrollbar.min.slider -side left -anchor w -fill x -expand yes
        pack $scrollbar.min.count -side left

        scale $scrollbar.cur.slider -variable $this-current -length 200 -showvalue true -orient horizontal -relief groove 
        iwidgets::spinint $scrollbar.cur.count -range {0 86400000} -justify right -width 5 -step 1 -textvariable $this-current -repeatdelay 300 -repeatinterval 10 -command "$this maybeRestart" -decrement "incr $this-current -1; $this maybeRestart" -increment "incr $this-current 1; $this maybeRestart"  
        pack $scrollbar.cur.slider -side left -anchor w -fill x -expand yes
        pack $scrollbar.cur.count -side left
        bind $scrollbar.cur.slider <ButtonRelease> "$this maybeRestart"
        
       scale $scrollbar.max.slider -variable $this-range_max -length 200 -showvalue true -orient horizontal -relief groove
        iwidgets::spinint $scrollbar.max.count -range {0 86400000} -justify right -width 5 -step 1 -textvariable $this-range_max -repeatdelay 300 -repeatinterval 10 -command "$this maybeRestart"
        pack $scrollbar.max.slider -side left -anchor w -fill x -expand yes
        pack $scrollbar.max.count -side left

        scale $scrollbar.inc.slider -variable $this-inc-amount -length 200 -showvalue true -orient horizontal -relief groove
        iwidgets::spinint $scrollbar.inc.count -range {0 86400000} -justify right -width 5 -step 1 -textvariable $this-inc-amount -repeatdelay 300 -repeatinterval 10 -command "$this maybeRestart"
        pack $scrollbar.inc.slider -side left -anchor w -fill x -expand yes
        pack $scrollbar.inc.count -side left

        $scrollbar.min.slider configure -label "Start $rroc:" -from $rmin -to $rmax 
        $scrollbar.cur.slider configure -label "Current $rroc:" -from $rmin -to $rmax
        $scrollbar.max.slider configure -label "End $rroc:" -from $rmin -to $rmax
        $scrollbar.inc.slider configure -label "Increment current $rroc by:" -from 1 -to [expr $rmax-$rmin]
         
        # Create and pack the play mode frame
        label $playmode.label -text "Play Mode"
        radiobutton $playmode.once -text "Once" -variable $this-playmode -value once -command "$this maybeRestart"
        radiobutton $playmode.loop -text "Loop" -variable $this-playmode -value loop -command "$this maybeRestart"
        radiobutton $playmode.bounce1 -text "Bounce" -variable $this-playmode -value bounce1 -command "$this maybeRestart"
        radiobutton $playmode.bounce2 -text "Bounce with repeating endpoints" -variable $this-playmode -value bounce2 -command "$this maybeRestart"
        iwidgets::spinint $playmode.delay -labeltext {Step Delay (ms)} -range {0 86400000} -justify right -width 5 -step 10 -textvariable $this-delay -repeatdelay 300 -repeatinterval 10

        pack $playmode.label -side top -expand yes -fill both
        pack $playmode.once $playmode.loop $playmode.bounce1 $playmode.bounce2 $playmode.delay -side top -anchor w

        trace variable $this-current w "update idletasks;\#"
        trace variable $this-delay w "$this maybeRestart;\#"

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
		
		pack $w.vcr 
    pack $w.cur -side top -expand yes -fill x
		
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
		
		pack $vcr.rewind $vcr.stepb $vcr.pause $vcr.play $vcr.stepf $vcr.fforward -side left -fill both -expand yes
		global ToolTipText
		Tooltip $vcr.rewind $ToolTipText(VCRrewind)
		Tooltip $vcr.stepb $ToolTipText(VCRstepback)
		Tooltip $vcr.pause $ToolTipText(VCRpause)
		Tooltip $vcr.play $ToolTipText(VCRplay)
		Tooltip $vcr.stepf $ToolTipText(VCRstepforward)
		Tooltip $vcr.fforward $ToolTipText(VCRfastforward)
		
		set $this-cur $cur
		scale $cur.cur -variable $this-current -length 200 -showvalue true -orient horizontal -relief groove 
    iwidgets::spinint $cur.count -range {0 86400000} -justify right -width 5 -step 1 -textvariable $this-current -repeatdelay 300 -repeatinterval 10 -command "$this maybeRestart" -decrement "incr $this-current -1; $this maybeRestart" -increment "incr $this-current 1; $this maybeRestart" 
    bind $cur.cur <ButtonRelease> "$this maybeRestart"
		pack $cur.cur	 -fill x -expand yes -side left
    pack $cur.count -side left
    $this update_range
  
		moveToCursor $w
	}


	method ChooseMatrixFile { } {

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
			-command "wm withdraw $w;  $this OpenNewMatrixfile" \
 			-commandname "Open" \
			-cancel "wm withdraw $w" \
			-title "Select timeseries data file" \
			-filetypes {{ "Header file for matrix data" "*.nhdr" } }\
			-initialdir $initdir \
			-defaultextension "*.*" \
			-selectedfiletype 0

		wm deiconify $w	
	}
	
	method OpenMatrixfile {} {

		global $this-filename
		global $this-filename-entry
		
		set $this-filename [[set $this-filename-entry] get] 
		
	}

	method OpenNewMatrixfile {} {

		global $this-filename
		global $this-filename-set
		global $this-filename-entry
		
		set $this-filename [set $this-filename-set] 
		
	}   
    
    
    
    
}


