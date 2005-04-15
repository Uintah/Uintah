catch {rename DaveW_ISL_Downhill_Simplex3 ""}

itcl_class DaveW_ISL_Downhill_Simplex3 {
    inherit Module

    constructor {config} {
	set name Downhill_Simplex3
	set_defaults
    }


    method set_defaults {} {	
        global $this-tcl_status
	global $this-methodTCL
	global $this-useCacheTCL
	set $this-methodTCL downhill
	set $this-useCacheTCL 1
    }


    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}

	
	
        toplevel $w
	wm minsize $w 100 50
	
	set n "$this-c needexecute "
        
	frame $w.f1 -relief groove -borderwidth 1
	label $w.f1.label -text "Network:"
	
        frame $w.f
	label $w.f.label -text "Status:"
	entry $w.f.t -width 15  -relief sunken -bd 2 -textvariable $this-tcl_status
	pack $w.f.label $w.f.t  -side left -pady 2m         
	
	frame $w.g
        button $w.g.go -text "Execute" -relief raised -command $n 
        button $w.g.p -text "Pause" -relief raised -command "$this-c pause"
        button $w.g.np -text "Unpause" -relief raised -command "$this-c unpause"
	button $w.g.print -text "Print" -relief raised -command "$this-c print"
	button $w.g.stop -text "Stop" -relief raised -command "$this-c stop"
	pack $w.g.go $w.g.p $w.g.np $w.g.print $w.g.stop -side left -fill x
	global $this-methodTCL
        make_labeled_radio $w.m "Method:" "" \
                left $this-methodTCL \
                {{"Downhill Simplex" "downhill"} \
                {"Protozoa" "protozoa"} \
                {"Simulated Annealing" "anneal"}}
	global $this-useCacheTCL
	checkbutton $w.b -text "UseCache" -variable $this-useCacheTCL
	pack $w.f1 $w.f $w.g $w.m $w.b -side top -fill x
    }
	
}

