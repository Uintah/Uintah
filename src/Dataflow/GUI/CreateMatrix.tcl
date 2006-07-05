itcl_class SCIRun_Math_CreateMatrix {
    inherit Module

    constructor {config} {
        set name CreateMatrix
        
        initGlobal $this-crows 0
        initGlobal $this-ccols 0
        initGlobal $this-loaddata 1
    }

    method ui {} {
        global $this-rows
        global $this-cols
        global $this-data
        global $this-rlabel
        global $this-clabel
        global $this-loaddata

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
          -textvariable $this-rows \
          -command "$this update_contents"

        iwidgets::entryfield $dimensions.cf \
          -labeltext "# Cols" \
          -validate numeric \
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
        global $this-rlabel
        global $this-clabel

        set nrows [set $this-rows]
        set ncols [set $this-cols]

        set newrlabel {}
        set newclabel {}

        for {set c 0} {$c < $ncols } {incr c} {
          if { $c < [llength [set $this-clabel]]} {
              set data [lindex [set $this-clabel] $c]
            } else {
              set data $c
            }
            lappend newclabel $data
        }

        set $this-clabel $newclabel      

        for {set r 0} {$r < $nrows } {incr r} {
          if { $c < [llength [set $this-rlabel]]} {
              set data [lindex [set $this-rlabel] $r]
            } else {
              set data $r
            }
            lappend newrlabel $data
        }
              
        set $this-rlabel $newrlabel      
              
        
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
        global $this-clabel
        global $this-rlabel
        global $this-rows
        global $this-cols    
        global $this-crows
        global $this-ccols 
        global $this-loaddata

        if {[set $this-rows] == [set $this-crows] && [set $this-cols] == [set $this-ccols] && [set $this-loaddata] == 0} {
          return
        }
        
        if {[set $this-loaddata] == 0} {
          $this update_matrixdata
        }

        set [set $this-loaddata] 0
    
        set $this-crows [set $this-rows]
        set $this-ccols [set $this-cols]

        $this resize_data
       
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
          set labeldata [set $this-clabel]
          iwidgets::entryfield $d.clabel-$c -borderwidth 2 -background blue -foreground white -textbackground blue -relief raised -command "$this update_clabel $c" -width 8 -justify center
          bind $d.clabel-$c <Leave> "$this update_clabel $c"
          set data [lindex $labeldata $c]
          $d.clabel-$c insert 0 $data
          grid $d.clabel-$c -row 0 -column [expr $c + 1 ] -sticky nsew
        }

        for {set r 0} {$r < $nrows } {incr r} {
          set labeldata [set $this-rlabel]
          iwidgets::entryfield $d.rlabel-$r -borderwidth 2 -background blue -foreground white -textbackground blue -relief raised -command "$this update_rlabel $r" -width 12 -justify center
          bind $d.rlabel-$r <Leave> "$this update_rlabel $r"
          set data [lindex $labeldata $r]
          $d.rlabel-$r insert 0 $data
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
        global $this-clabel
        global $this-rlabel
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

          set newdata {}
          for {set r 0} {$r < $nrows } {incr r} {
              set data [$d.rlabel-$r get]
              lappend newdata $data
          }
          set $this-rlabel $newdata
          
          set newdata {}
          for {set c 0} {$c < $ncols } {incr c} {
              set data [$d.clabel-$c get]
              lappend newdata $data
          }
          set $this-clabel $newdata

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

    method update_rlabel {r} {
        global $this-rlabel
            
        set w .ui[modname]
        set contents [$w.contents childsite] 
        set d [$contents.d childsite]     
        
        set data [$d.rlabel-$r get]
        set $this-rlabel [lreplace [set $this-rlabel] $r $r $data]    
    }

    method update_clabel {c} {
        global $this-clabel
            
        set w .ui[modname]
        set contents [$w.contents childsite] 
        set d [$contents.d childsite]     
        
        set data [$d.clabel-$c get]
        set $this-clabel [lreplace [set $this-clabel] $c $c $data]    
    }


}



