itcl_class ModelCreation_Script_ParameterList {
    inherit Module
    constructor {config} {
        set name ParameterList
        set_defaults
    }

    method set_defaults {} {
        global $this-data
        global $this-new_field_count
        global $this-update_all
        global $this-use-global
        
        set $this-data {{0 "example field" string "example" ""} {0 "example scalar" scalar 1.0 ""} }    
        set $this-new_field_count 1
        set $this-update_all "$this update_all_data"
        set $this-use-global 0
    }

    method ui {} {

        global $this-update_all
        set $this-update_all "$this update_all_data"

        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        
        toplevel $w
        
        wm minsize $w 600 100
        
        iwidgets::labeledframe $w.paramlist
        set paramlist [$w.paramlist childsite]       
        
        iwidgets::scrolledframe $paramlist.d \
          -vscrollmode dynamic \
          -hscrollmode dynamic
        pack $paramlist.d        
        pack $w.paramlist -fill both -expand yes -side top
        
        update_gui

        frame $w.paramedit 
        pack $w.paramedit -anchor e -side top
        set paramedit $w.paramedit       

        button $paramedit.delete -text "Delete Parameters" -command "$this delete_parameters"
        button $paramedit.add -text "Add Parameter" -command "$this add_parameter"
        
        grid $paramedit.delete -row 0 -column 0 -sticky news
        grid $paramedit.add -row 0 -column 1 -sticky news        
        
        makeSciButtonPanel $w $w $this
        moveToCursor $w
    }
    
    method update_gui {} {

        global $this-data

        set boolean_color  #ffd0f0
        set string_color   #e0ffe0
        set scalar_color   #e0e0ff
        set vector_color   #ffe0e0        
        set tensor_color   #ffffd5   
        set array_color    #d5ffff   
        set filename_color #ffe7d0   
        set header_color #dcdcdc
        set partypes {boolean string scalar vector tensor array filename}     

        font create paramlist_font -family Helvetica -weight bold
        
        set w .ui[modname]
        set paramlist [$w.paramlist childsite]     
        
        pack forget $paramlist.d
        destroy $paramlist.d
        iwidgets::scrolledframe $paramlist.d \
          -vscrollmode dynamic \
          -hscrollmode dynamic -background $header_color
        pack $paramlist.d -fill both -expand yes
      
        set len [llength [set $this-data]]
        set d [$paramlist.d childsite]   

        frame $d.h -relief raised -background $header_color
        grid $d.h -row 0 -column 0 -sticky we

        frame $d.h.f -relief raised -background $header_color
        pack $d.h.f -anchor w

        label $d.h.f.check -width 4 -text "  " -background $header_color
        grid  $d.h.f.check -row 0 -column 0 

        if {[expr [set $this-use-global] == 1]} {
          label $d.h.f.arg -width 10 -text "GLOBAL" -background $header_color
          grid  $d.h.f.arg -row 0 -column 1 
        }
       
        label $d.h.f.type -width 10 -text "FIELDTYPE" -background $header_color
        grid  $d.h.f.type -row 0 -column 2 

        label $d.h.f.name -width 25 -text "FIELDNAME" -background $header_color
        grid  $d.h.f.name -row 0 -column 3 

        label $d.h.f.data -width 20 -text "FIELDDATA" -background $header_color
        grid  $d.h.f.data -row 0 -column 4 
        

        for {set p 0} {$p < $len} {incr p} {
          set data [lindex [set $this-data] $p]
          
          set parcheck [lindex $data 0]
          set parname  [lindex $data 1]
          set partype  [lindex $data 2]
          set pardata  [lindex $data 3]
          set parscript [lindex $data 4]
          
          if {[string equal $partype "boolean"]} {

            frame $d.$p -relief raised -background $boolean_color
            grid $d.$p -row [expr $p + 1] -column 0 -sticky we

            frame $d.$p.f -relief raised -background $boolean_color
            pack $d.$p.f -anchor w
                      
            global $this-cb-$p          
            checkbutton $d.$p.f.check -foreground black -background $boolean_color -variable $this-cb-$p -command "$this update_data $p"
            grid $d.$p.f.check -row 0 -column 0 
            $d.$p.f.check deselect
            if {[expr $parcheck == 1]} {
              $d.$p.f.check select
            }
            
            iwidgets::entryfield $d.$p.f.name -width 25 -command "$this update_data $p" -background $boolean_color -textbackground $boolean_color -textfont paramlist_font
            grid $d.$p.f.name -row 0 -column 3 -sticky snew
            $d.$p.f.name insert 0 $parname
            
            iwidgets::optionmenu $d.$p.f.type -command "$this update_type $p" -foreground black -background $boolean_color
            foreach q $partypes {
              $d.$p.f.type insert end $q
            }
            grid $d.$p.f.type -row 0 -column 2 -sticky snew
            $d.$p.f.type select $partype
            
            iwidgets::optionmenu $d.$p.f.data -command "$this update_data $p"  -background $boolean_color 
            grid $d.$p.f.data -row 0 -column 4 -sticky snew            
            foreach q {true false} {
              $d.$p.f.data insert 0 $q
            }
            $d.$p.f.data select $pardata

            if {[expr [set $this-use-global] == 1]} {
              iwidgets::entryfield $d.$p.f.script -width 12 -command "$this update_data $p" -background $boolean_color -textbackground $boolean_color -textfont paramlist_font
              grid $d.$p.f.script -row 0 -column 1 -sticky snew
              $d.$p.f.script insert 0 $parscript            
            }
          }

          if {[string equal $partype "string"]} {

            frame $d.$p -relief raised -background $string_color
            grid $d.$p -row [expr $p + 1] -column 0 -sticky we

            frame $d.$p.f -relief raised -background $string_color
            pack $d.$p.f -anchor w
                      
            global $this-cb-$p          
            checkbutton $d.$p.f.check -foreground black -background $string_color -variable $this-cb-$p -command "$this update_data $p"
            grid $d.$p.f.check -row 0 -column 0 
            $d.$p.f.check deselect
            if {[expr $parcheck == 1]} {
              $d.$p.f.check select
            }
            
            iwidgets::entryfield $d.$p.f.name -width 25 -command "$this update_data $p" -background $string_color -textbackground $string_color -textfont paramlist_font
            grid $d.$p.f.name -row 0 -column 3 -sticky snew
            $d.$p.f.name insert 0 $parname
            
            iwidgets::optionmenu $d.$p.f.type -command "$this update_type $p" -foreground black -background $string_color
            foreach q $partypes {
              $d.$p.f.type insert end $q
            }
            grid $d.$p.f.type -row 0 -column 2 -sticky snew
            $d.$p.f.type select $partype
            
            iwidgets::entryfield $d.$p.f.data -width 50 -command "$this update_data $p" -foreground black -background $string_color -textbackground $string_color  -textfont paramlist_font
            grid $d.$p.f.data -row 0 -column 4 -sticky snew            
            $d.$p.f.data insert 0 $pardata

            if {[expr [set $this-use-global] == 1]} {
              iwidgets::entryfield $d.$p.f.script -width 12 -command "$this update_data $p" -background $string_color -textbackground $string_color -textfont paramlist_font
              grid $d.$p.f.script -row 0 -column 1 -sticky snew
              $d.$p.f.script insert 0 $parscript            
            }
          }

          if {[string equal $partype "filename"]} {

            frame $d.$p -relief raised -background $filename_color
            grid $d.$p -row [expr $p + 1] -column 0 -sticky we

            frame $d.$p.f -relief raised -background $filename_color
            pack $d.$p.f -anchor w
                      
            global $this-cb-$p          
            checkbutton $d.$p.f.check -foreground black -background $filename_color -variable $this-cb-$p -command "$this update_data $p"
            grid $d.$p.f.check -row 0 -column 0 
            $d.$p.f.check deselect
            if {[expr $parcheck == 1]} {
              $d.$p.f.check select
            }
            
            iwidgets::entryfield $d.$p.f.name -width 25 -command "$this update_data $p" -background $filename_color -textbackground $filename_color -textfont paramlist_font
            grid $d.$p.f.name -row 0 -column 3 -sticky snew
            $d.$p.f.name insert 0 $parname
            
            iwidgets::optionmenu $d.$p.f.type -command "$this update_type $p" -foreground black -background $filename_color
            foreach q $partypes {
              $d.$p.f.type insert end $q
            }
            grid $d.$p.f.type -row 0 -column 2 -sticky snew
            $d.$p.f.type select $partype
            
            frame $d.$p.f.data -width 30 -background $filename_color -bd 0
            grid $d.$p.f.data -row 0 -column 4 -sticky snew   

            button $d.$p.f.data.browse -text "Browse" -command "$this browse_filename $p" -background $filename_color 
            grid $d.$p.f.data.browse -row 0 -column 1 -sticky snew            

            iwidgets::entryfield $d.$p.f.data.filename -width 42 -command "$this update_data $p" -foreground black -background $filename_color -textbackground $filename_color  -textfont paramlist_font
            grid $d.$p.f.data.filename -row 0 -column 0 -sticky snew            
            $d.$p.f.data.filename insert 0 $pardata

            if {[expr [set $this-use-global] == 1]} {
              iwidgets::entryfield $d.$p.f.script -width 12 -command "$this update_data $p" -background $filename_color -textbackground $filename_color -textfont paramlist_font
              grid $d.$p.f.script -row 0 -column 1 -sticky snew
              $d.$p.f.script insert 0 $parscript            
            }
          }

          if {[string equal $partype "scalar"]} {

            frame $d.$p -relief raised -background $scalar_color
            grid $d.$p -row [expr $p + 1] -column 0 -sticky we

            frame $d.$p.f -relief raised -background $scalar_color
            pack $d.$p.f -anchor w
                        
            global $this-cb-$p            
            checkbutton $d.$p.f.check -background $scalar_color -variable $this-cb-$p -command "$this update_data $p"
            grid $d.$p.f.check -row 0 -column 0 
            $d.$p.f.check deselect
            if {[expr $parcheck == 1]} {
              $d.$p.f.check select
            }
                      
            iwidgets::entryfield $d.$p.f.name -width 25 -command "$this update_data $p" -foreground black -background $scalar_color -textbackground $scalar_color  -textfont paramlist_font
            grid $d.$p.f.name -row 0 -column 3 -sticky snew
            $d.$p.f.name insert 0 $parname
            
            iwidgets::optionmenu $d.$p.f.type -command "$this update_type $p" -foreground black -background $scalar_color
            foreach q $partypes {
              $d.$p.f.type insert end $q
            }
            grid $d.$p.f.type -row 0 -column 2 -sticky snew
            $d.$p.f.type select $partype
            
            iwidgets::entryfield $d.$p.f.data -width 10 -command "$this update_data $p" -validate real -foreground black -background $scalar_color   -textbackground $scalar_color  -textfont paramlist_font
            grid $d.$p.f.data -row 0 -column 4 -sticky snew          
            $d.$p.f.data insert 0 $pardata

            if {[expr [set $this-use-global] == 1]} {
              iwidgets::entryfield $d.$p.f.script -width 12 -command "$this update_data $p" -background $scalar_color -textbackground $scalar_color -textfont paramlist_font
              grid $d.$p.f.script -row 0 -column 1 -sticky snew
              $d.$p.f.script insert 0 $parscript            
            }
          }
        
          if {[string equal $partype "vector"]} {

            frame $d.$p -relief raised -background $vector_color
            grid $d.$p -row [expr $p + 1] -column 0 -sticky we          

            frame $d.$p.f -relief raised -background $vector_color
            pack $d.$p.f -anchor w
                   
            global $this-cb-$p                 
            checkbutton $d.$p.f.check  -background $vector_color -variable $this-cb-$p -command "$this update_data $p"
            grid $d.$p.f.check -row 0 -column 0           
            $d.$p.f.check deselect
            if {[expr $parcheck == 1]} {
              $d.$p.f.check select
            }
            
            iwidgets::entryfield $d.$p.f.name -width 25 -command "$this update_data $p" -foreground black -background $vector_color -textbackground $vector_color  -textfont paramlist_font
            grid $d.$p.f.name -row 0 -column 3 -sticky snew
            $d.$p.f.name insert 0 $parname
            
            iwidgets::optionmenu $d.$p.f.type -command "$this update_type $p" -foreground black -background $vector_color 
            foreach q $partypes {
              $d.$p.f.type insert end $q
            }
            grid $d.$p.f.type -row 0 -column 2 -sticky snew
            $d.$p.f.type select $partype
            
            frame $d.$p.f.data -width 30 -background $vector_color -bd 0
            grid $d.$p.f.data -row 0 -column 4 -sticky snew   
            
            for {set r 0} {$r < 3} {incr r} {
              iwidgets::entryfield $d.$p.f.data.$r -command "$this update_data $p" -width 10 -foreground black -background $vector_color -textbackground $vector_color  -textfont paramlist_font
              grid $d.$p.f.data.$r -row 0 -column $r -sticky snew             
              
              set subdata [lindex $pardata $r]
              $d.$p.f.data.$r insert 0 $subdata
            }

            if {[expr [set $this-use-global] == 1]} {
              iwidgets::entryfield $d.$p.f.script -width 12 -command "$this update_data $p" -background $vector_color -textbackground $vector_color -textfont paramlist_font
              grid $d.$p.f.script -row 0 -column 1 -sticky snew
              $d.$p.f.script insert 0 $parscript            
            }
          }
        
          if {[string equal $partype "tensor"]} {

            frame $d.$p -relief raised -background $tensor_color
            grid $d.$p -row [expr $p + 1] -column 0 -sticky we          

            frame $d.$p.f -relief raised -background $tensor_color
            pack $d.$p.f -anchor w
                   
            global $this-cb-$p                 
            checkbutton $d.$p.f.check  -background $tensor_color -variable $this-cb-$p -command "$this update_data $p"
            grid $d.$p.f.check -row 0 -column 0           
            $d.$p.f.check deselect
            if {[expr $parcheck == 1]} {
              $d.$p.f.check select
            }
            
            iwidgets::entryfield $d.$p.f.name -width 25 -command "$this update_data $p" -foreground black -background $tensor_color -textbackground $tensor_color  -textfont paramlist_font
            grid $d.$p.f.name -row 0 -column 3 -sticky snew
            $d.$p.f.name insert 0 $parname
            
            iwidgets::optionmenu $d.$p.f.type -command "$this update_type $p" -foreground black -background $tensor_color 
            foreach q $partypes {
              $d.$p.f.type insert end $q
            }
            grid $d.$p.f.type -row 0 -column 2 -sticky snew
            $d.$p.f.type select $partype
            
            frame $d.$p.f.data -width 30 -background $tensor_color -bd 0
            grid $d.$p.f.data -row 0 -column 4 -sticky snew   
            
            for {set r 0} {$r < 6} {incr r} {
              iwidgets::entryfield $d.$p.f.data.$r -command "$this update_data $p" -width 10 -foreground black -background $tensor_color -textbackground $tensor_color  -textfont paramlist_font
              grid $d.$p.f.data.$r -row 0 -column $r -sticky snew             
              
              set subdata [lindex $pardata $r]
              $d.$p.f.data.$r insert 0 $subdata
            }

            if {[expr [set $this-use-global] == 1]} {
              iwidgets::entryfield $d.$p.f.script -width 12 -command "$this update_data $p" -background $tensor_color -textbackground $tensor_color -textfont paramlist_font
              grid $d.$p.f.script -row 0 -column 1 -sticky snew
              $d.$p.f.script insert 0 $parscript                        
            }
          }

          if {[string equal $partype "array"]} {

            frame $d.$p -relief raised -background $array_color
            grid $d.$p -row [expr $p + 1] -column 0 -sticky we          

            frame $d.$p.f -relief raised -background $array_color
            pack $d.$p.f -anchor w
                   
            global $this-cb-$p                 
            checkbutton $d.$p.f.check  -background $array_color -variable $this-cb-$p -command "$this update_data $p"
            grid $d.$p.f.check -row 0 -column 0           
            $d.$p.f.check deselect
            if {[expr $parcheck == 1]} {
              $d.$p.f.check select
            }
            
            iwidgets::entryfield $d.$p.f.name -width 25 -command "$this update_data $p" -foreground black -background $array_color -textbackground $array_color  -textfont paramlist_font
            grid $d.$p.f.name -row 0 -column 3 -sticky snew
            $d.$p.f.name insert 0 $parname
            
            iwidgets::optionmenu $d.$p.f.type -command "$this update_type $p" -foreground black -background $array_color 
            foreach q $partypes {
              $d.$p.f.type insert end $q
            }
            grid $d.$p.f.type -row 0 -column 2 -sticky snew
            $d.$p.f.type select $partype
            
            frame $d.$p.f.data -width 30 -background $array_color -bd 0
            grid $d.$p.f.data -row 0 -column 4 -sticky snew   
            
            set alen [llength $pardata]
            iwidgets::entryfield $d.$p.f.data.len -command "$this update_data $p" -labeltext "#" -width 3 -foreground black -background $array_color -textbackground $array_color  -textfont paramlist_font -labelfont paramlist_font
            grid $d.$p.f.data.len -row 0 -column 0 -sticky snew -padx 8            
            $d.$p.f.data.len insert 0 $alen
            
            for {set r 0} {$r < $alen} {incr r} {
              iwidgets::entryfield $d.$p.f.data.$r -command "$this update_data $p" -width 10 -foreground black -background $array_color -textbackground $array_color  -textfont paramlist_font
              grid $d.$p.f.data.$r -row 0 -column [expr $r + 1] -sticky snew             
              
              set subdata [lindex $pardata $r]
              $d.$p.f.data.$r insert 0 $subdata
            }

            if {[expr [set $this-use-global] == 1]} {
              iwidgets::entryfield $d.$p.f.script -width 12 -command "$this update_data $p" -background $array_color -textbackground $array_color -textfont paramlist_font
              grid $d.$p.f.script -row 0 -column 1 -sticky snew
              $d.$p.f.script insert 0 $parscript                        
            }
          }
       }   
       font delete paramlist_font    
    }

    method update_all_data {} {

      global $this-data
      set w .ui[modname]
      if {[winfo exists $w]} {

        set paramlist [$w.paramlist childsite]  
        set d [$paramlist.d childsite]  
        set datalist ""
        set len [llength [set $this-data]]
        for {set p 0} {$p < $len} {incr p} {

          set data [lindex [set $this-data] $p]
          set partype  [lindex $data 2]

          set newdata ""
          lappend newdata [set $this-cb-$p]
          lappend newdata [$d.$p.f.name get]
          lappend newdata $partype

          if {[string equal $partype "boolean"]} {
            lappend newdata [$d.$p.f.data get]
          }                    
          
          if {[string equal $partype "string"]} {
            lappend newdata [$d.$p.f.data get]
          }                    
          
          if {[string equal $partype "filename"]} {
            lappend newdata [$d.$p.f.data.filename get]
          }                    
          
          if {[string equal $partype "scalar"]} {
            lappend newdata [$d.$p.f.data get]
          }                    
          
          if {[string equal $partype "vector"]} {
             set subdata "" 
             for {set r 0} {$r < 3} {incr r} {
                lappend subdata [$d.$p.f.data.$r get]
              }          
            lappend newdata $subdata
          }
                      
          if {[string equal $partype "tensor"]} {
             set subdata "" 
             for {set r 0} {$r < 6} {incr r} {
                lappend subdata [$d.$p.f.data.$r get]
              }          
            lappend newdata $subdata
          }

          if {[string equal $partype "array"]} {
            set pardata  [lindex $data 3]
            set newlen [$d.$p.f.data.len get]
            set oldlen [llength $pardata]
           
            set subdata ""
            if {[expr $newlen <= $oldlen]} {
              for {set r 0} {$r < $newlen} {incr r} {
                lappend subdata [$d.$p.f.data.$r get]
              }          
              lappend newdata $subdata
            }
           
            if {[expr $newlen > $oldlen]} {
              for {set r 0} {$r < $oldlen} {incr r} {
                lappend subdata [$d.$p.f.data.$r get]
              }
              for {set r $oldlen} {$r < $newlen} {incr r} {
                lappend subdata 0.0
              }          
              lappend newdata $subdata
            }
          }

          if {[expr [set $this-use-global] == 1]} {
            lappend newdata [$d.$p.f.script get]
          } else {
            lappend newdata ""
          }
          lappend datalist $newdata
        }
        
        set $this-data $datalist
      }
    }


    method update_data {p} {
      
      global $this-data
      
      set w .ui[modname]
      if {[winfo exists $w]} {
        set paramlist [$w.paramlist childsite]  
        set d [$paramlist.d childsite]  
     
        set data [lindex [set $this-data] $p]
        set partype  [lindex $data 2]

        lappend newdata [set $this-cb-$p]
        lappend newdata [$d.$p.f.name get]
        lappend newdata $partype

        if {[string equal $partype "boolean"]} {
          lappend newdata [$d.$p.f.data get]
        }                    
        
        if {[string equal $partype "string"]} {
          lappend newdata [$d.$p.f.data get]
        }                    
        
        if {[string equal $partype "filename"]} {
          lappend newdata [$d.$p.f.data.filename get]
        }                    
        
        if {[string equal $partype "scalar"]} {
          lappend newdata [$d.$p.f.data get]
        }                    
        
        if {[string equal $partype "vector"]} {
           for {set r 0} {$r < 3} {incr r} {
              lappend subdata [$d.$p.f.data.$r get]
            }          
          lappend newdata $subdata
        }
                    
        if {[string equal $partype "tensor"]} {
           for {set r 0} {$r < 6} {incr r} {
              lappend subdata [$d.$p.f.data.$r get]
            }          
          lappend newdata $subdata
        }

        if {[string equal $partype "array"]} {
          set pardata  [lindex $data 3]
          set newlen [$d.$p.f.data.len get]
          set oldlen [llength $pardata]
         
          if {[expr $newlen <= $oldlen]} {
            for {set r 0} {$r < $newlen} {incr r} {
              lappend subdata [$d.$p.f.data.$r get]
            }          
            lappend newdata $subdata
          }
         
          if {[expr $newlen > $oldlen]} {
            for {set r 0} {$r < $oldlen} {incr r} {
              lappend subdata [$d.$p.f.data.$r get]
            }
            for {set r $oldlen} {$r < $newlen} {incr r} {
              lappend subdata 0.0
            }          
            lappend newdata $subdata
          }
        
          if {[expr $newlen != $oldlen]} {
            if {[expr [set $this-use-global] == 1]} {
              lappend newdata [$d.$p.f.script get]
            }
            else
            {
              lappend newdata ""
            }
            set $this-data [lreplace [set $this-data] $p $p $newdata]
            update_gui
            return
          }
        }
        if {[expr [set $this-use-global] == 1]} {
          lappend newdata [$d.$p.f.script get]
        } else {
          lappend newdata ""
        }
        set $this-data [lreplace [set $this-data] $p $p $newdata]            
      }
    }
    
    method update_type {p} {
       global $this-data
      
      set w .ui[modname]
      if {[winfo exists $w]} {

	update_all_data

        set paramlist [$w.paramlist childsite]  
        set d [$paramlist.d childsite]  
     
        set data [lindex [set $this-data] $p]
        set parcheck [lindex $data 0]
        set parname  [lindex $data 1]
        set partype  [lindex $data 2]
        set parscript  [lindex $data 4]
    
        set newpartype [$d.$p.f.type get]
        
        if {[string equal $partype $newpartype]} {        
          return
        }
        
        lappend newdata $parcheck
        lappend newdata $parname
        lappend newdata $newpartype

        if {[string equal $newpartype "boolean"]} {
          lappend newdata "true"
        }                    
         
        if {[string equal $newpartype "string"]} {
          lappend newdata "new string"
        }                    

        if {[string equal $newpartype "filename"]} {
          lappend newdata ""
        }                    
        
        if {[string equal $newpartype "scalar"]} {
          lappend newdata 0.0
        }                
        
        if {[string equal $newpartype "vector"]} {
           for {set r 0} {$r < 3} {incr r} {
              lappend subdata 0.0
            }          
          lappend newdata $subdata
        }
                       
        if {[string equal $newpartype "tensor"]} {
           for {set r 0} {$r < 6} {incr r} {
              lappend subdata 0.0
            }          
          lappend newdata $subdata
        }
        
        if {[string equal $newpartype "array"]} {
           for {set r 0} {$r < 1} {incr r} {
              lappend subdata 0.0
            }          
          lappend newdata $subdata
        }
        
        lappend newdata $parscript
        set $this-data [lreplace [set $this-data] $p $p $newdata]         
        update_gui
      }
    }
    
    method delete_parameters {} {
      global $this-data
      
      set len [llength [set $this-data]]
      
      set newlist ""
      for {set p 0} {$p < $len} {incr p} {
          set data [lindex [set $this-data] $p]
          
          set parcheck  [lindex $data 0]
          set parname   [lindex $data 1]
          set partype   [lindex $data 2]
          set pardata   [lindex $data 3]
          set parscript [lindex $data 4]
          
          if {[expr $parcheck == 0]} {
              set newdata ""
              lappend newdata $parcheck
              lappend newdata $parname
              lappend newdata $partype
              lappend newdata $pardata
              lappend newdata $parscript
              
              lappend newlist $newdata
          }
      }
      
      set $this-data $newlist
      update_gui
    }
    
    method add_parameter {} {
      global $this-data
      global $this-new_field_count

      update_all_data

      set len [llength [set $this-data]]
      
      lappend newentry 0
      lappend newentry [format "new_datafield_%d" [set $this-new_field_count]]
      lappend newentry string
      lappend newentry ""
      lappend newentry ""
      incr $this-new_field_count
      
      for {set p 0} {$p < $len} {incr p} {
          set data [lindex [set $this-data] $p]
          set parcheck [lindex $data 0]   
          if {[expr $parcheck == 1]} {
             set $this-data [linsert [set $this-data] $p $newentry]
             update_gui
             return
          }
      }
      
      lappend $this-data $newentry
      update_gui
    }

    method browse_filename {p } {

      global env
      global $this-filename-$p
      global $this-data

      set data [lindex [set $this-data] $p]
      set $this-filename-$p [lindex $data 3]

      # Create a unique name for the file selection window
      set w [format "%s-fileselector-%d" .ui[modname] $p] 

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

      if {[info exists env(SCIRUN_DATA)]} {
              set initdir $env(SCIRUN_DATA)
      } elseif {[info exists env(SCI_DATA)]} {
              set initdir $env(SCI_DATA)
      } elseif {[info exists env(PSE_DATA)]} {
              set initdir $env(PSE_DATA)
      }

      makeOpenFilebox \
              -parent $w \
              -filevar $this-filename-$p \
              -command "wm withdraw $w;  $this set_filename $p" \
              -commandname "Select" \
              -cancel "wm withdraw $w" \
              -title "SELECT FILENAME" \
              -filetypes { { "All files"  "*" } }\
              -initialdir $initdir \
              -defaultextension "*.*" \
              -selectedfiletype 0
      
      wm deiconify $w	
    }

    method set_filename {p} {
      global $this-filename-$p
      global $this-data    
    
      set data [lindex [set $this-data] $p]
      set partype [lindex $data 2]
      if {[string equal $partype "filename"]} {
        set data [lreplace $data 3 3 [set $this-filename-$p]]
        set $this-data [lreplace [set $this-data] $p $p $data]
        update_gui
      }
    }

}


