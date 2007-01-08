itcl_class ModelCreation_Script_ManageParameterString {
    inherit Module
    constructor {config} {
        set name ManageParameterString
        set_defaults
    }

    method set_defaults {} {
        global $this-string-name
        global $this-string-listbox
        global $this-string-selection
        global $this-string-entry

        set $this-string-name "string"
        set $this-string-selection ""
        set $this-string-listbox ""
        set $this-string-entry ""
    }

    
    method choose_field {} {
        global $this-string-name
        global $this-string-selection
        
        set w .ui[modname]
        if {[winfo exists $w]} {
          set stringnum [$w.sel.listbox curselection]
          if [expr [string equal $stringnum ""] == 0] {
            set $this-string-name  [lindex [set $this-string-selection] $stringnum] 
          }
        }
    }    


    method ui {} {
        global $this-string-name
        global $this-string-listbox
        global $this-string-selection
        global $this-string-entry    
    
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w
        wm minsize $w 100 50        
  
        frame $w.name
        frame $w.sel

        pack $w.name -side top -fill x -padx 5p
        pack $w.sel -side top -fill both -expand yes -padx 5p
        
        label $w.name.label -text "String Name"
        entry $w.name.entry -textvariable $this-string-name
        pack $w.name.label -side left 
        pack $w.name.entry -side left -fill x -expand yes

        iwidgets::scrolledlistbox $w.sel.listbox -selectioncommand "$this choose_field"
        $w.sel.listbox component listbox configure -listvariable $this-string-selection -selectmode browse 
        pack $w.sel.listbox -fill both -expand yes
               
        makeSciButtonPanel $w $w $this
        moveToCursor $w
    }

}


