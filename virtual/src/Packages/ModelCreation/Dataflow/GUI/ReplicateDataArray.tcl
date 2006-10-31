itcl_class ModelCreation_DataArrayMath_ReplicateDataArray {
    inherit Module
    constructor {config} {
        set name ReplicateDataArray
        set_defaults
    }

    method set_defaults {} {
    	global $this-size
      set $this-size 1
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w

        iwidgets::entryfield $w.size \
          -labeltext "Number of times to replicate template array:" \
          -command "$this update_size" \
          -validate numeric 
			     
        $w.size insert end [set $this-size]         
        pack $w.size
    
        makeSciButtonPanel $w $w $this
        moveToCursor $w
    }
    
  method update_size {} {

        set w .ui[modname]
        if {[winfo exists $w]} {
          global $this-size
          set $this-size [$w.size get]
        }
    }  
}


