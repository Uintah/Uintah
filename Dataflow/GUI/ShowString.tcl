itcl_class SCIRun_Visualization_ShowString {
    inherit Module

    constructor {config} {
        set name ShowString
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w
 
        # Style
        iwidgets::labeledframe $w.style -labeltext "String Style"
        set style [$w.style childsite]

        # Until we have a good way to ask for the screen resolution
        # this one should be switched off

        # Style - box
        #frame $style.bbox

        #checkbutton $style.bbox.button -variable $this-bbox \
        #  -command "$this-c needexecute"
        #label $style.bbox.label -text "Box" -width 4 \
        #  -anchor w -just left
	
        #pack $style.bbox.button $style.bbox.label -side left
        #pack $style.bbox -side left

        # Style - color
        frame $style.color
        addColorSelection $style.color "Color" $this-color "color_change"
        pack $style.color -side left -padx 5

        pack $w.style  -fill x -expand yes -side top 

	frame $w.twocol
	
# Size
	iwidgets::labeledframe $w.twocol.size -labeltext "String Size"
	set size [$w.twocol.size childsite]

        # Size - tiny
        frame $size.tiny

        radiobutton $size.tiny.button -variable $this-size -value 0 \
            -command "$this-c needexecute"
        label $size.tiny.label -text "Tiny" -width 6 \
            -anchor w -just left
        
        pack $size.tiny.button $size.tiny.label -side left
        pack $size.tiny -side top -padx 5
        
        # Size - small
        frame $size.small

        radiobutton $size.small.button -variable $this-size -value 1 \
            -command "$this-c needexecute"
        label $size.small.label -text "Small" -width 6 \
            -anchor w -just left
        
        pack $size.small.button $size.small.label -side left
        pack $size.small -side top -padx 5

      # Size - medium
        frame $size.medium

        radiobutton $size.medium.button -variable $this-size -value 2 \
            -command "$this-c needexecute"
        label $size.medium.label -text "Medium" -width 6 \
            -anchor w -just left
        
        pack $size.medium.button $size.medium.label -side left
        pack $size.medium -side top -padx 5

      # Size - large
        frame $size.large

        radiobutton $size.large.button -variable $this-size -value 3 \
            -command "$this-c needexecute"
        label $size.large.label -text "Large" -width 6 \
            -anchor w -just left
        
        pack $size.large.button $size.large.label -side left
        pack $size.large -side top -padx 5

      # Size - huge
        frame $size.huge

        radiobutton $size.huge.button -variable $this-size -value 4 \
            -command "$this-c needexecute"
        label $size.huge.label -text "Huge" -width 6 \
            -anchor w -just left
        
        pack $size.huge.button $size.huge.label -side left
        pack $size.huge -side top -padx 5
        
# Location
	iwidgets::labeledframe $w.twocol.location -labeltext "String Location"
	set location [$w.twocol.location childsite]

	set locator [makeStickyLocator $location.gui \
			 $this-location-x $this-location-y \
			 100 100]

	$locator bind movable <ButtonRelease> "$this-c needexecute"

	pack $location.gui -fill x -expand yes -side top


	pack $w.twocol.size $w.twocol.location -fill both -expand yes -side left

	pack $w.twocol -fill x -expand yes -side top

                               
        makeSciButtonPanel $w $w $this
        moveToCursor $w
    }

    method raiseColor {col color colMsg} {
       global $color
       set window .ui[modname]
       if {[winfo exists $window.color]} {
           SciRaise $window.color
           return
       } else {
           makeColorPicker $window.color $color \
             "$this setColor $col $color $colMsg" \
             "destroy $window.color"
       }
   }

    method setColor {col color colMsg} {
       global $color
       global $color-r
       global $color-g
       global $color-b
       set ir [expr int([set $color-r] * 65535)]
       set ig [expr int([set $color-g] * 65535)]
       set ib [expr int([set $color-b] * 65535)]

       set window .ui[modname]
       $col config -background [format #%04x%04x%04x $ir $ig $ib]
       $this-c $colMsg

      # The above works for only the geometry not for the text so execute.
       $this-c needexecute
    }

    method addColorSelection {frame text color colMsg} {
       #add node color picking 
       global $color
       global $color-r
       global $color-g
       global $color-b
       set ir [expr int([set $color-r] * 65535)]
       set ig [expr int([set $color-g] * 65535)]
       set ib [expr int([set $color-b] * 65535)]
       
       frame $frame.colorFrame
       frame $frame.colorFrame.col -relief ridge -borderwidth \
         4 -height 0.8c -width 1.0c \
         -background [format #%04x%04x%04x $ir $ig $ib]
       
       set cmmd "$this raiseColor $frame.colorFrame.col $color $colMsg"
       button $frame.colorFrame.set_color \
         -text $text -command $cmmd
       
       #pack the node color frame
       pack $frame.colorFrame.set_color $frame.colorFrame.col -side left -padx 2
       pack $frame.colorFrame -side left
    }



}


