itcl_class cConjGrad {
    inherit Module
    constructor {config} {
	set name cConjGrad
	set_defaults
    }


    method set_defaults {} {	
        global $this-tcl_max_it
	global $this-tcl_it
	global $this-tcl_max_err
	global $this-tcl_err
	global $this-tcl_precond
	global $this-tcl_status

        set $this-tcl_precond 0
	set $this-tcl_max_it 300
	set $this-tcl_max_err 0.001
        set $this-tcl_it 0
	set $this-tcl_err 1


   }


    method ui {} {
	set w .ui$this
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}

	
	global $this-tcl_max_it
	global $this-tcl_it
	global $this-tcl_max_err
	global $this-tcl_err
	global $this-tcl_precond
	global $this-tcl_status
      

        toplevel $w
	wm minsize $w 100 50

	set n "$this-c needexecute "
        

	frame $w.fs -relief groove -borderwidth 1
        label $w.fs.label -text "Settings:"

	frame $w.fs.f1
 	label $w.fs.f1.label -text "Maximum Iterations:"
	entry $w.fs.f1.t -width 10  -relief sunken -bd 2 -textvariable $this-tcl_max_it
	pack $w.fs.f1.label $w.fs.f1.t -side left -pady 2m
	
	frame $w.fs.f2
	label $w.fs.f2.label -text "Maximum Error:"
	entry $w.fs.f2.t -width 10  -relief sunken -bd 2 -textvariable $this-tcl_max_err
	pack $w.fs.f2.label $w.fs.f2.t -side left -pady 2m 
  
        pack $w.fs.label $w.fs.f1 $w.fs.f2 -fill x

	frame $w.fb -relief groove -borderwidth 1
	label $w.fb.label -text "Preconditioner:"
	radiobutton $w.fb.rb1 -text "No" -variable $this-tcl_precond -value 0  -anchor w
	radiobutton $w.fb.rb2 -text "Jacoby (Diagonal) " -variable $this-tcl_precond -value 1  -anchor w
        radiobutton $w.fb.rb3 -text "SSOR" -variable $this-tcl_precond -value 2  -anchor w

	pack $w.fb.label $w.fb.rb1 $w.fb.rb2 $w.fb.rb3 -fill x


        frame $w.fc -relief groove -borderwidth 1         
        label $w.fc.label -text "Convergence:"

	frame $w.fc.f4
	label $w.fc.f4.label -text "Iteration:"
	entry $w.fc.f4.t -width 10  -relief sunken -bd 2 -textvariable $this-tcl_it
	pack $w.fc.f4.label $w.fc.f4.t -side left -pady 2m 
	
	frame $w.fc.f5
	label $w.fc.f5.label -text "Error:"
	entry $w.fc.f5.t -width 10  -relief sunken -bd 2 -textvariable $this-tcl_err
	pack $w.fc.f5.label $w.fc.f5.t -side left -pady 2m

        frame $w.fc.f6
	label $w.fc.f6.label -text "Status:"
	entry $w.fc.f6.t -width 10  -relief sunken -bd 2 -textvariable $this-tcl_status
	pack $w.fc.f6.label $w.fc.f6.t  -side left -pady 2m         

        pack $w.fc.label $w.fc.f4 $w.fc.f5 $w.fc.f6
        button $w.stop -text "Stop" -relief raised -command "$this-c stop"
        button $w.go -text "Execute" -relief raised -command $n 

 	pack $w.fs $w.fb $w.fc  $w.stop $w.go  -fill x 
 
	     
    }
}

