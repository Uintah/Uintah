itcl_class Kurt_Vis_KurtScalarFieldReader { 

    inherit Module 
    
    protected filedir
    protected filename

    constructor {config} { 
        set name KurtScalarFieldReader 
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
	    $this activate
	}
	
	frame $w.f1.f -relief flat
	pack $w.f1.f -side top -padx 2 -pady 2 -expand yes -fill x
	
	button $w.f1.select -text Select -command "$this selectfile"
	pack $w.f1.select -side left -padx 2 -pady 2
	button $w.f1.close -text Close -command "destroy $w"
	pack $w.f1.close -side left -padx 2 -pady 2
#  	makeFilebox $w.f1 $this-filebase $n "wm withdraw $w"
## Replace !! when tcl is upgraded.

    } 

    method selectfile {} {
	set w .ui[modname]
	puts "third [$w.f1.fb get]"
	set $this-filebase [$w.f1.fb get]
	puts "fourth [set $this-filebase]"
	$this-c needexecute
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
