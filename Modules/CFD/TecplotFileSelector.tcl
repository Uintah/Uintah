itcl_class TecplotFileSelector { 

    inherit Module 
    constructor {config} { 
        set name TecplotFileSelector 
        set_defaults 
    } 
  

    method set_defaults {} { 
        global $this-tcl_status 
	global .ui$this-path
	set .ui$this-path "/home/sci/data19/kuzimmer/data/tecplot/2D"
  
    } 
  

    method ui {} { 
        set w .ui$this 
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
        pack $w.f2.f4.label $w.f2.f4.t  -side left -pady 2m 
	pack $w.f2.f4 -side bottom -anchor w

  	makeFilebox $w.f1 $this-filebase $n "destroy $w"
## Replace !! when tcl is upgraded.
	
    } 

    method activate {} {
	set w .ui$this
	
	$w.f2.rb configure -state active
#	$w.f2.rb deselect
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

    method deselect {} {
	set w .ui$this
	$w.f2.rb deselect
    }

#     method errorDialog { args } {

# 	set w .error$this
# 	set errorString [join $args]
# 	set button [tk_dialog $w "File Opening Error" \
# 			"Error: $errorString"  error 0 "Ok"]
#     }	
	
}
