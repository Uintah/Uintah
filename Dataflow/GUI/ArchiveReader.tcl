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
	    return;
	}

	global $this-filebase
	toplevel $w
	set n "$this-c needexecute"

	set len [string length [set $this-filebase]]
	if { $len == 0 } { set len 40 }
	frame $w.f 
	pack $w.f 
	label $w.f.l -text "Choose Archive"
	entry $w.f.e -textvariable $this-filebase -width $len
	frame $w.f.b
	pack $w.f.l $w.f.e $w.f.b -side top -pady 2 -expand yes -fill x

	button $w.f.b.b1 -text "Browse" -command "$this get_uda"
	button $w.f.b.b2 -text "Execute" \
	    -command $n
	pack $w.f.b.b1 $w.f.b.b2 -side left -padx 2

	bind $w.f.e <Return> $n 

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
	}
	$this-c needexecute
    }
	
}
