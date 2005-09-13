itcl_class SCIRun_Math_CreateMatrix {
    inherit Module
    constructor {config} {
        set name CreateMatrix
        set_defaults
    }

    method set_defaults {} {
        global $this-rows
        global $this-cols
        global $this-crows
        global $this-ccols
        global $this-data
        global $this-update-data
        
        set $this-rows 1
        set $this-cols 1
        set $this-crows 0
        set $this-ccols 0
        set $this-data { {0.0} } 
        set $this-update-data "$this update_matrixdata"
    }

    method ui {} {
        global $this-rows
        global $this-cols
        global $this-data

        set $this-update-data "$this update_matrixdata"
            
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w
 
        iwidgets::labeledframe $w.dimensions -labeltext "MATRIX DIMENSIONS"
        set dimensions [$w.dimensions childsite]        
        
        iwidgets::entryfield $dimensions.rf \
          -labeltext "# Rows" \
          -validate numeric \
          -invalid {showMsg "Use numbers only!"} \
          -textvariable $this-rows \
          -command "$this update_contents"

        iwidgets::entryfield $dimensions.cf \
          -labeltext "# Cols" \
          -validate numeric \
          -invalid {showMsg "Use numbers only!"} \
          -textvariable $this-cols \
          -command "$this update_contents"

        pack $dimensions.rf $dimensions.cf -padx 5 -pady 5 -side left

        trace variable $this-rows w "$this update_contents"
        trace variable $this-cols w "$this update_contents"
  
        iwidgets::labeledframe $w.contents -labeltext "MATRIX CONTENTS"
        set contents [$w.contents childsite]        
        
        iwidgets::scrolledframe $contents.d \
          -vscrollmode dynamic \
          -hscrollmode dynamic
        pack $contents.d
        
        pack $w.dimensions -fill x -pady 5 -side top
        pack $w.contents -fill both -expand yes -pady 5 -side top

        update_contents
        resize_data

        makeSciButtonPanel $w $w $this
        moveToCursor $w
    }
    
    method resize_data {} {
    
        global $this-data   
        global $this-rows
        global $this-cols    
    
        set nrows [set $this-rows]
        set ncols [set $this-cols]
        
        set newdata {}
        for {set c 0} {$c < $ncols } {incr c} {
        
            if { $c < [llength [set $this-data]]} {
              set rowdata [lindex [set $this-data] $c]
              set newrowdata {}
            } else {
              set rowdata {}
              set newrowdata {}
            }
            for {set r 0} {$r < $nrows } {incr r} {

              if { $r < [llength [set rowdata]]} {
                set data [lindex [set rowdata] $r]
              } else {
                set data 0.0
              }
              lappend newrowdata $data
            }
            lappend newdata $newrowdata
        }
        
        set $this-data $newdata
    }
    
    method update_contents {} {
        global $this-data
        global $this-rows
        global $this-cols    
        global $this-crows
        global $this-ccols    
       
        if {[set $this-rows] == [set $this-crows] && [set $this-cols] == [set $this-ccols]} {
          return
        }
       
        update_matrixdata
       
        set $this-crows [set $this-rows]
        set $this-ccols [set $this-cols]
       
        resize_data
       
        set w .ui[modname]
        set contents [$w.contents childsite]     
        set nrows [set $this-rows]
        set ncols [set $this-cols]
        
        pack forget $contents.d
        destroy $contents.d
        iwidgets::scrolledframe $contents.d \
          -vscrollmode dynamic \
          -hscrollmode dynamic
        pack $contents.d -fill both -expand yes
      
        set d [$contents.d childsite]   

        for {set c 0} {$c < $ncols } {incr c} {
          label $d.clabel-$c -bd 2 -text [format "%d" $c] \
          -bg blue -fg white -relief raised
          grid $d.clabel-$c -row 0 -column [expr $c + 1 ] -sticky nsew
        }

        for {set r 0} {$r < $nrows } {incr r} {
          label $d.rlabel-$r -bd 2 -text [format "%d" $r] \
          -bg blue -fg white -relief raised
          grid $d.rlabel-$r -column 0 -row [expr $r + 1 ] -sticky nsew
        }


        for {set c 0} {$c < $ncols } {incr c} {
            set rowdata [lindex [set $this-data] $c]
            for {set r 0} {$r < $nrows } {incr r} {
                  
            iwidgets::entryfield $d.data-$r-$c \
                      -validate real \
                      -command "$this update_data $r $c" \
                      -width 8                     
    
            set data [lindex $rowdata $r]
            $d.data-$r-$c insert 0 $data
            grid $d.data-$r-$c -row [expr $r + 1] -column [expr $c + 1] -sticky nsew
            }
        }
    }
    
    
    method update_matrixdata {} {
        global $this-data
        global $this-crows
        global $this-ccols  
        
        set nrows [set $this-crows]
        set ncols [set $this-ccols]
            
        set w .ui[modname]

            
        set w .ui[modname]
        if {[winfo exists $w]} {
          set contents [$w.contents childsite] 
          set d [$contents.d childsite]     
          set newdata {}
          
          for {set c 0} {$c < $ncols } {incr c} {
              set newrowdata {}   
              for {set r 0} {$r < $nrows } {incr r} {
                  set data [$d.data-$r-$c get]
                  lappend newrowdata $data
              }
              lappend newdata $newrowdata
          }
                  
          set $this-data $newdata
        }

    }
    
    method update_data {r c} {
        global $this-data
            
        set w .ui[modname]
        set contents [$w.contents childsite] 
        set d [$contents.d childsite]     
        
        set data [$d.data-$r-$c get]
        set rowdata [lindex [set $this-data] $c]
        set rowdata [lreplace $rowdata $r $r $data]
        set $this-data [lreplace [set $this-data] $c $c $rowdata]    
    }
}



