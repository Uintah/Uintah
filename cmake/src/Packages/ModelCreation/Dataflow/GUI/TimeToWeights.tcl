itcl_class ModelCreation_Time_TimeToWeights {
    inherit Module
    constructor {config} {
        set name TimeToWeights
        set_defaults
    }

    method set_defaults {} {
    
    # Variables for file selector
    setGlobal $this-filename	""
     
    # Variables for wrapper creation
    
    # Variables for nd matrix information
    
    # Variables for play buttons
		
    setGlobal $this-slider_min         0.0
    setGlobal $this-slider_max         2.0
    setGlobal $this-range_min          0.0
    setGlobal $this-range_max          2.0
    setGlobal $this-playmode          once
    setGlobal $this-current           0.5
    setGlobal $this-execmode          init
    setGlobal $this-delay             0
    setGlobal $this-inc-amount        0.1
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
        
        set step [expr ( $max - $min)/10000]
        
        $scrollbar.min.slider configure -label "Start time:" -from $min -to $max -resolution [set $this-inc-amount]
        $scrollbar.cur.slider config -label "Current time:" -from $min -to $max -resolution [set $this-inc-amount]
        $scrollbar.max.slider config -label "End time:" -from $min -to $max -resolution [set $this-inc-amount]
        $scrollbar.inc.slider config -label "Increment current time by:" -from 0 -to [expr $max-$min] -resolution 0.00001

      }
      
    }


    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w

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

        # Save range, creating the scale resets it to defaults.
        set rmin [set $this-slider_min]
        set rmax [set $this-slider_max]
    
        set step [expr ( $rmax - $rmin)/10000]
    

        # Create the various range sliders
        frame $scrollbar.min
        frame $scrollbar.cur
        frame $scrollbar.max
        frame $scrollbar.inc
        pack $scrollbar.min $scrollbar.cur $scrollbar.max $scrollbar.inc -side top -anchor w -fill x -expand yes
    
        scale $scrollbar.min.slider -variable $this-range_min -length 200 -showvalue true -orient horizontal -relief groove -resolution [set $this-inc-amount]
        iwidgets::entryfield $scrollbar.min.count -width 10 -textvariable $this-range_min -command "$this maybeRestart"
        pack $scrollbar.min.slider -side left -anchor w -fill x -expand yes
        pack $scrollbar.min.count -side left

        scale $scrollbar.cur.slider -variable $this-current -length 200 -showvalue true -orient horizontal -relief groove -resolution [set $this-inc-amount]
        iwidgets::entryfield $scrollbar.cur.count -width 10 -textvariable $this-current -command "$this maybeRestart"
        pack $scrollbar.cur.slider -side left -anchor w -fill x -expand yes
        pack $scrollbar.cur.count -side left
        bind $scrollbar.cur.slider <ButtonRelease> "$this maybeRestart"
        
        scale $scrollbar.max.slider -variable $this-range_max -length 200 -showvalue true -orient horizontal -relief groove -resolution [set $this-inc-amount]
        iwidgets::entryfield $scrollbar.max.count -width 10 -textvariable $this-range_max -command "$this maybeRestart"
        pack $scrollbar.max.slider -side left -anchor w -fill x -expand yes
        pack $scrollbar.max.count -side left

        scale $scrollbar.inc.slider -variable $this-inc-amount -length 200 -showvalue true -orient horizontal -relief groove -resolution $step
        iwidgets::entryfield $scrollbar.inc.count -width 10 -textvariable $this-inc-amount -command "$this update_range ; $this maybeRestart"
        pack $scrollbar.inc.slider -side left -anchor w -fill x -expand yes
        pack $scrollbar.inc.count -side left
        bind $scrollbar.inc.slider <ButtonRelease> "$this update_range ; $this maybeRestart"

        $scrollbar.min.slider configure -label "Start time:" -from $rmin -to $rmax 
        $scrollbar.cur.slider configure -label "Current time:" -from $rmin -to $rmax
        $scrollbar.max.slider configure -label "End time:" -from $rmin -to $rmax 
        $scrollbar.inc.slider configure -label "Increment current time by:" -from 0 -to [expr $rmax-$rmin] -resolution 0.00001
         
         
         
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
    
    
}


