itcl_class SCIRun_Visualization_ShowString {
    inherit Module
    constructor {config} {
        set name ShowString
        set_defaults
    }

    method set_defaults {} {
      global $this-bbox
      global $this-size
      global $this-location
      global $this-color-r
      global $this-color-g
      global $this-color-b

      set $this-bbox 0
      set $this-size 2
      set $this-location "Top Left"
      set $this-color-r 1.0
      set $this-color-g 1.0
      set $this-color-b 1.0
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w
 
        # Style
        iwidgets::labeledframe $w.style -labeltext "Title Style"
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

        # Size
        iwidgets::labeledframe $w.size -labeltext "Title Size"
        set size [$w.size childsite]
        pack $w.style  -fill x -expand yes -side top 
        pack $w.size -fill x -expand yes -side top

        # Size - tiny
        frame $size.tiny

        radiobutton $size.tiny.button -variable $this-size -value 0 \
            -command "$this-c needexecute"
        label $size.tiny.label -text "Tiny" -width 6 \
            -anchor w -just left
        
        pack $size.tiny.button $size.tiny.label -side left
        pack $size.tiny -side left -padx 5
        
        # Size - small
        frame $size.small

        radiobutton $size.small.button -variable $this-size -value 1 \
            -command "$this-c needexecute"
        label $size.small.label -text "Small" -width 6 \
            -anchor w -just left
        
        pack $size.small.button $size.small.label -side left
        pack $size.small -side left -padx 5

      # Size - medium
        frame $size.medium

        radiobutton $size.medium.button -variable $this-size -value 2 \
            -command "$this-c needexecute"
        label $size.medium.label -text "Medium" -width 6 \
            -anchor w -just left
        
        pack $size.medium.button $size.medium.label -side left
        pack $size.medium -side left -padx 5

      # Size - large
        frame $size.large

        radiobutton $size.large.button -variable $this-size -value 3 \
            -command "$this-c needexecute"
        label $size.large.label -text "Large" -width 6 \
            -anchor w -just left
        
        pack $size.large.button $size.large.label -side left
        pack $size.large -side left -padx 5

      # Size - huge
        frame $size.huge

        radiobutton $size.huge.button -variable $this-size -value 4 \
            -command "$this-c needexecute"
        label $size.huge.label -text "Huge" -width 6 \
            -anchor w -just left
        
        pack $size.huge.button $size.huge.label -side left
        pack $size.huge -side left -padx 5
        
      #	pack $w.size -fill x -expand yes -side top



      # Location
        iwidgets::labeledframe $w.location -labeltext "Title Location"
        set location [$w.location childsite]

      # Location - top left
        frame $location.top_left

        radiobutton $location.top_left.button -variable $this-location \
            -value "Top Left" -command "$this-c needexecute"
        label $location.top_left.label -text "Top Left" -width 9 \
            -anchor w -just left
        
        pack $location.top_left.button $location.top_left.label -side left

      # Location - top center
        frame $location.top_center

        radiobutton $location.top_center.button -variable $this-location \
            -value "Top Center" -command "$this-c needexecute"
        label $location.top_center.label -text "Top Center" -width 10 \
            -anchor w -just left
        
        pack $location.top_center.button $location.top_center.label -side left

      # Location - bottom left
        frame $location.bottom_left

        radiobutton $location.bottom_left.button -variable $this-location \
            -value "Bottom Left" -command "$this-c needexecute"
        label $location.bottom_left.label -text "Bottom Left" -width 12 \
            -anchor w -just left
        
        pack $location.bottom_left.button $location.bottom_left.label -side left

      # Location - bottom center
        frame $location.bottom_center

        radiobutton $location.bottom_center.button -variable $this-location \
            -value "Bottom Center" -command "$this-c needexecute"
        label $location.bottom_center.label -text "Bottom Center" -width 13 \
            -anchor w -just center
        
        pack $location.bottom_center.button $location.bottom_center.label -side left


        pack $location.top_left $location.top_center \
            $location.bottom_left $location.bottom_center -side left
        
        pack $w.location -fill x -expand yes -side top       
                               
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


