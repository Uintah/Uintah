itcl_class Kurt_Vis_ArchiveReader { 

    inherit Module 
    
    protected filedir
    protected filename

    constructor {config} { 
        set name ArchiveReader 
        set_defaults
    } 
  
    method filedir { filebase } {
	set n [string last "/" "$filebase"]
	if { $n != -1} {
	    return [ string range $filebase 0 $n ]
	} else {
	    return ""
	}
    }

    method filename { filebase } {
	set n [string last "/" "$filebase"]
	if { $n != -1} {
	    return [ string range $filebase [eval $n + 1] end]
	} else {
	    return ""
	}
    }
	
    method set_defaults {} { 
	global $this-filebase 
	set $this-filebase ""
    } 
  

    method ui {} { 
        set w .ui[modname] 
	if {[winfo exists $w]} {
	    wm deiconify $w
	    raise $w
	    return;
	}
	global $this-startFrame
	global $this-endFrame
	global $this-animate
	global $this-increment
	global $this-tcl_status
	global $this-filebase
	global env

        toplevel $w 
        wm minsize $w 100 50 
  
        set n "$this-c needexecute " 
  
        frame $w.f1 -relief groove -borderwidth 2
	pack $w.f1 -in $w -side left
	
	
	if { [string compare [set $this-filebase] ""] == 0 } {
	    if { [info exists env(PSE_DATA)] } {
		set dir $env(PSE_DATA)
	    } else {
		set dir $env(PWD)
	    }
	    iwidgets::fileselectionbox $w.f1.fb \
		-directory $dir
	    #-dblfilescommand  "$this selectfile"
	    $w.f1.fb.filter delete 0 end
	    $w.f1.fb.filter insert 0 "$dir\/*"
	    $w.f1.fb filter
	    pack $w.f1.fb -padx 2 -pady 2 -side top
	} else {
	    iwidgets::fileselectionbox $w.f1.fb \
	    -directory [filedir [set $this-filebase ] ] 
	    #-dblfilescommand  "$this selectfile"


	    $w.f1.fb.filter delete 0 end
	    $w.f1.fb.filter insert 0 [filedir [set $this-filebase]]/*
	    $w.f1.fb filter
	    $w.f1.fb.selection delete 0 end
	    $w.f1.fb.selection insert 0 [set $this-filebase]
	    pack $w.f1.fb -padx 2 -pady 2 -side top
	}
	
	frame $w.f1.f -relief flat
	pack $w.f1.f -side top -padx 2 -pady 2 -expand yes -fill x
	
	button $w.f1.select -text Select -command "$this selectfile"
	pack $w.f1.select -side left -padx 2 -pady 2
	button $w.f1.close -text Close -command "wm withdraw $w"
	pack $w.f1.close -side left -padx 2 -pady 2

    } 

    method selectfile {} {
	set w .ui[modname]
	set $this-filebase [$w.f1.fb get]
	$this-c needexecute
    }
}
