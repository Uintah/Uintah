global CoreTCL

itcl_class Packages/Phil_Tbon_ElapsedTime {

    inherit Module

    protected min
    protected sec
    protected hsec
    protected w
    protected thefont

    constructor {config} {
        set name ElapsedTime
        set_defaults
    }


    method set_defaults {} {
	set min 0
	set sec 0
	set hsec 0
    }

    method ui {} {
	set thefont *-courier-bold-r-normal--*-250-*-*-*-*-*-*

        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }


        toplevel $w
        wm minsize $w 100 50

	frame $w.header
	frame $w.fmin
	frame $w.fsec
	frame $w.fhsec
	frame $w.start

	set n "$this-c needexecute"

	label $w.header.lab -text " Elapsed Time: " -font $thefont
	label $w.fmin.lab -text "   $min :" -font $thefont
	label $w.fsec.lab -text "0$sec :" -font $thefont
	label $w.fhsec.lab -text "0$hsec " -font $thefont
	button $w.start.bstart -text "Start" -font $thefont \
		-command "set $this-stop 0; $n"
	button $w.start.bstop -text "Stop" -font $thefont \
		-command "set $this-stop 1"

	pack $w.header.lab -in $w.header -side left -padx 2 -pady 2\
		-ipadx 2 -ipady 2
	pack $w.fmin.lab -in $w.fmin -side right -padx 2 -pady 2 \
		-ipadx 2 -ipady 2
	pack $w.fsec.lab -in $w.fsec -side right -padx 2 -pady 2 \
		-ipadx 2 -ipady 2
	pack $w.fhsec.lab -in $w.fhsec -side right -padx 2 -pady 2 \
		-ipadx 2 -ipady 2
	pack $w.start.bstop $w.start.bstart -in $w.start -side right \
		-padx 2 -pady 2 -ipadx 2 -ipady 2

	pack $w.header $w.fmin $w.fsec $w.fhsec $w.start -in $w -side left \
		-ipadx 2 -ipady 2
    }    

    method update_elapsed_time {} {
	set vars [$this-c getVars]
	set varlist [split $vars]

	set min [lindex $varlist 0]
	set sec [lindex $varlist 1]
	set hsec [lindex $varlist 2]

	if { $min < 10 } {
	    $w.fmin.lab configure -text "   $min :" -font $thefont
	} elseif { $min < 100 } {
	    $w.fmin.lab configure -text "  $min :" -font $thefont
	} elseif { $min < 1000 } {
	    $w.fmin.lab configure -text " $min :" -font $thefont
	} else {
	    $w.fmin.lab configure -text "$min :" -font $thefont
	}
	
	if { $sec < 10 } {
	    $w.fsec.lab configure -text "0$sec :" -font $thefont
	} else {
	    $w.fsec.lab configure -text "$sec :" -font $thefont
	}

	if { $hsec < 10 } {
	    $w.fhsec.lab configure -text "0$hsec " -font $thefont
	} else {
	    $w.fhsec.lab configure -text "$hsec " -font $thefont
	}
	update idletasks
    }
}


