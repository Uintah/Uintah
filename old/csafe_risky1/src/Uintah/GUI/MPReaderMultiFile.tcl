itcl_class Uintah_Readers_MPReaderMultiFile { 

    inherit Module 
    
    protected filedir
    protected filename

    constructor {config} { 
        set name MPReaderMultiFile 
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
	global $this-dirbase
	global $this-timestep
	set $this-filebase ""
	set $this-dirbase ""
	set $this-timestep ""
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
	global $this-dirbase
	global $this-timestep
	global env

        toplevel $w 
        wm minsize $w 100 50 
  
        set n "$this-c needexecute " 
  
        frame $w.f1 -relief groove -borderwidth 2
	pack $w.f1 -in $w -side left -expand yes -fill both
	
	
	frame $w.f2 -relief groove -borderwidth 2
	pack $w.f2 -in $w -side right -expand yes -fill both

	label $w.f2.title -text "Animation control" \
	    -font "-*-*-*-*-*-*-14-*-*-*-*-*-*-*"
	pack $w.f2.title -side top 
	
	label $w.f2.l -text "\nAnimation begins with\nthe selected file.\n"
	pack $w.f2.l -side top

	checkbutton $w.f2.rb -text "Animate" -state disabled \
	    -command $n -variable $this-animate -offvalue 0 -onvalue 1
	pack $w.f2.rb -side top
	
	set color [$w.f2.rb cget -disabledforeground]
	frame $w.f2.f1 -relief flat
	pack $w.f2.f1 -side top -anchor w
	entry $w.f2.f1.entry -width 4 -textvariable $this-startFrame \
	     -state disabled -fg $color
	pack $w.f2.f1.entry -side left
	label $w.f2.f1.label -text "start frame" -fg $color 
	pack $w.f2.f1.label -side right

	frame $w.f2.f2 -relief flat
	pack $w.f2.f2 -side top -anchor w
	entry $w.f2.f2.entry -width 4 -textvariable $this-endFrame \
	     -state disabled -fg $color
	pack $w.f2.f2.entry -side left
	label $w.f2.f2.label -text "end frame" -fg $color 
	pack $w.f2.f2.label -side right
	
	frame $w.f2.f3 -relief flat
	pack $w.f2.f3 -side top -anchor w
	entry $w.f2.f3.entry -width 4 -textvariable $this-increment \
	     -state disabled -fg $color
	pack $w.f2.f3.entry -side left
	label $w.f2.f3.label -text "increment" -fg $color 
	pack $w.f2.f3.label -side right

	frame $w.f2.f4 -relief flat
        label $w.f2.f4.label -text "Status:" 
        entry $w.f2.f4.t -width 15  -relief sunken -bd 2 \
	    -textvariable $this-tcl_status 
        pack $w.f2.f4.label $w.f2.f4.t -expand yes -fill x -side left -pady 2m 
	pack $w.f2.f4 -side bottom -anchor w


	frame $w.f1.dirbase -relief flat
	pack $w.f1.dirbase -side top -padx 2 -pady 2 -expand yes -fill x
	
	frame $w.f1.namebase -relief flat 
	pack $w.f1.namebase -side top -padx 2 -pady 2 -expand yes -fill x

	frame $w.f1.timestep  -relief flat
	pack $w.f1.timestep -side top -padx 2 -pady 2 -expand yes -fill x
	
	label $w.f1.dirbase.l -text "Directory Base:  "
	entry $w.f1.dirbase.e -textvariable $this-dirbase
	pack $w.f1.dirbase.e $w.f1.dirbase.l -side right

	label $w.f1.namebase.l -text "Filename base:  "
	entry $w.f1.namebase.e -textvariable $this-namebase
	pack $w.f1.namebase.e $w.f1.namebase.l -side right

	label $w.f1.timestep.l -text "Timestep \#:  "
	entry $w.f1.timestep.e -textvariable $this-timestep
	pack $w.f1.timestep.e $w.f1.timestep.l -side right

	frame $w.f1.f -relief flat
	pack $w.f1.f -side top -padx 2 -pady 2 -expand yes -fill x
	
	button $w.f1.select -text Select -command "$this selectfiles"
	pack $w.f1.select -side left -padx 2 -pady 2
	button $w.f1.close -text Close -command "destroy $w"
	pack $w.f1.close -side left -padx 2 -pady 2
#  	makeFilebox $w.f1 $this-filebase $n "wm withdraw $w"
## Replace !! when tcl is upgraded.

    } 

    method selectfiles {} {
	set w .ui[modname]
	set dirnames [ glob [set $this-dirbase]???? ]
	set nprocessors [llength dirnames ]
	set filenames { }
	set ts [formatTimestep]
	set curdir [pwd]
	for {set i 0 } { $i < [llength $dirnames] } { incr i} {
	    set dirbase [lindex $dirnames $i]
	    cd $dirbase
	    set fbase [set $this-namebase]$ts\*
	    set fnames [glob $fbase]
	    for { set j 0} { $j < [llength $fnames] } { incr j} {
		set fullname [pwd]/[lindex $fnames $j]
		lappend filenames $fullname
	    }
	    cd $curdir
	}

	$this-c combineFiles $filenames

	$this-c needexecute
    }

    method formatTimestep {} {
	global $this-timestep
	set ts  [set $this-timestep]
	if { [string length $ts] == 0 } {
	    return 0000
	} elseif {[string length $ts] == 1 } {
	    return 000$ts
	} elseif {[string length $ts] == 2 } {
	    return 00$ts
	} elseif {[string length $ts] == 3 } {
	    return 0$ts
	} else {
	    return $ts
	}
    }
	    
	
    
    method activate {} {
	set w .ui[modname]
	
	if [winfo exists $w] {
	    $w.f2.rb configure -state active
#	    $w.f2.rb deselect
	    set color [$w.f2.rb cget -fg]
	    $w.f2.f1.entry configure -state normal
	    $w.f2.f2.entry configure -state normal
	    $w.f2.f3.entry configure -state normal
	    $w.f2.f1.entry configure -fg $color
	    $w.f2.f2.entry configure -fg $color
	    $w.f2.f3.entry configure -fg $color
	    $w.f2.f1.label configure -fg $color
	    $w.f2.f2.label configure -fg $color
	    $w.f2.f3.label configure -fg $color
	}
    }

    method deselect {} {
	set w .ui[modname]
	$w.f2.rb deselect
    }

#     method errorDialog { args } {

# 	set w .error[modname]
# 	set errorString [join $args]
# 	set button [tk_dialog $w "File Opening Error" \
# 			"Error: $errorString"  error 0 "Ok"]
#     }	
	
}
