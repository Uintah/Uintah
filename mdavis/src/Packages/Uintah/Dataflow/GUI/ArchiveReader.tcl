itcl_class Uintah_DataIO_ArchiveReader { 

    inherit Module 
    
    constructor {config} { 
        set name ArchiveReader 
        set_defaults
    } 
  	
    method set_defaults {} { 
	global $this-filebase 
	set $this-filebase ""
    } 
  

    method ui {} { 

	set w .ui[modname]
	if {[winfo exists $w]} {
	    return
	}

	global $this-filebase
	toplevel $w
	set n "$this-c needexecute"

	set len [string length [set $this-filebase]]
	if { $len == 0 } { set len 40 }
	frame $w.f 

	pack $w.f -padx 5 -pady 5 -expand yes -fill both

	label $w.f.l -text "Current Archive: "
	entry $w.f.e -textvariable $this-filebase -width $len
	bind $w.f.e <Return> $n 

	pack $w.f.l $w.f.e -side left -expand yes -fill x -anchor w

	TooltipMultiline $w.f.l \
            "This field displays the currently selected\n" \
	    "UDA (Uintah Data Archive)."

	TooltipMultiline "$w.f.e" "Click the 'Select UDA' button to choose an UDA,\nor enter the full path yourself."

	makeSciButtonPanel $w $w $this "\" Select UDA \" \"$this get_uda\" \"Press to select an UDA directory.\""
	moveToCursor $w
    }
    method get_uda {} {
	set w .ui[modname]
	global $this-filebase
	
	set filebase [tk_chooseDirectory -parent $w \
			-mustexist 1 -initialdir [set $this-filebase] ]
	if { $filebase != "" } {
	    set len [string length $filebase]
	    incr len
	    $w.f.e configure -width $len
	    set $this-filebase $filebase
	    $this-c needexecute
	}
    }
	
}
