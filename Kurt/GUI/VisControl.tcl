########################################
#CLASS
#    VizControl
#
#    Visualization control for simulation data that contains
#    information on both a regular grid in particle sets.
#
#OVERVIEW TEXT
#    This module receives a ParticleGridReader object.  The user
#    interface is dynamically created based information provided by the
#    ParticleGridReader.  The user can then select which variables he/she
#    wishes to view in a visualization.
#
#
#
#KEYWORDS
#    ParticleGridReader, Material/Particle Method
#
#AUTHOR
#    Kurt Zimmerman
#    Department of Computer Science
#    University of Utah
#    January 1999
#
#    Copyright (C) 1999 SCI Group
#
#LOG
#    Created January 5, 1999
########################################

catch {rename VisControl ""}

itcl_class Kurt_Vis_VisControl { 
    inherit Module 

    protected psVarList ""
    protected pvVarList ""
    protected gsVarList ""
    protected gvVarList ""
    protected gf ""
    protected pf ""

    constructor {config} { 
        set name VisControl 
        set_defaults

    } 

    method set_defaults {} { 
        global $this-tcl_status 
	global $this-gsVar;
	global $this-gvVar;
	global $this-psVar;
	global $this-pvVar;
	global $this-time;

	set $this-gsVar ""
	set $this-gvVar ""
	set $this-psVar ""
	set $this-pvVar ""
	set $this-pName ""
	set $this-gName ""
	set $this-time 0
    } 
    
    method ui {} { 
        set w .ui[modname] 

        if {[winfo exists $w]} { 
	    wm deiconify $w
            raise $w 
            return;
        } 
	
        toplevel $w 
        wm minsize $w 100 50 
	
	#       set n "$this-c needexecute" 
 	frame $w.f -relief flat
 	pack $w.f -side top -expand yes -fill both
	

	scale $w.f.time -orient horizontal -from 0 -to 1.0 \
	    -resolution 0.01 -bd 2
	
	button $w.b -text "Close" -command "wm withdraw $w"
	pack $w.f.time $w.b -side top -fill x -padx 2 -pady 2

	makeFrames $w.f
	
    }


    method Visible {} {
	if {[winfo exists .ui[modname]]} {
	    return 1
	} else {
	    return 0
	}
    }
    
    method Rebuild {} {
	set w .ui[modname]

	$this destroyFrames
	$this makeFrames $w.f
    }

    method build {} {
	set w .ui[modname]

	$this makeFrames $w.f
    }

    method makeFrames { parent } {
	global $this-gsVar;
	global $this-gvVar;
	global $this-psVar;
	global $this-pvVar;
	set Labels [list ""  "Grid Vars" "Particle Vars"]
	
	for { set i 1 } { $i < [llength $Labels] } { incr i } {
	    frame $parent.f$i -relief groove -borderwidth 2 
	    label $parent.f$i.label -text [lindex $Labels $i] 
	    pack $parent.f$i -side left -expand yes -fill both -padx 2
	    pack $parent.f$i.label -side top
	    
	    frame $parent.f$i.names -relief groove -borderwidth 2
	    pack $parent.f$i.names -side top -expand yes -fill both 

	    frame $parent.f$i.1 -relief flat -borderwidth 2
	    pack $parent.f$i.1 -side top -expand yes -fill both -padx 2

	    buildControlFrame $parent.f$i.1
	    

	}


	set w .ui[modname] 
	if { [winfo exists $w.b] } {
	    destroy $w.b
	}
	if { [winfo exists $w.time] } {
	    destroy $w.time
	}
	
	scale $w.time -orient horizontal -from 0 -to 1.0 \
	    -variable $this-time -resolution 0.01 -bd 2 -command "$this-c needexecute"
	#-text "Time"
	
	button $w.b -text "Close" -command "wm withdraw $w"
	pack $w.time $w.b -side top -fill x -padx 2 -pady 2

	
	set gf $parent.f1
	set pf $parent.f2
    }
    

    method buildControlFrame { name } {

	if { [winfo exists $name.1] } {
	    destroy $name.1
	}
	if { [winfo exists $name.2] } {
	    destroy $name.2
	}

	frame $name.1 -relief flat -borderwidth 2
	label $name.1.s -text "Scalars"
	pack $name.1 -side left  -expand yes -fill both -padx 2
	pack $name.1.s -side top
	frame $name.2 -relief flat -borderwidth 2
	label $name.2.v -text "Vectors"
	pack $name.2 -side left -expand yes -fill both -padx 2
	pack $name.2.v -side top
    }

    method destroyFrames {} {
	set w .ui[modname] 

	destroy $gf
	destroy $pf
	if { [winfo exists $w.b] } {
	    destroy $w.b
	}
	set gNameList ""
	set pNameList ""
	set psVarList ""
	set pvVarList ""
	set gsVarList ""
	set gvVarList ""
	set gf ""
	set pf ""
#	button $w.b -text "Make Frames" -command "$this makeFrames $w.f"
#	pack $w.b -side top -fill x -padx 2 -pady 2

    }

    method SetTimeRange { min max timesteps } {
	set w .ui[modname] 
	if { [winfo exists $w.time] } {
	    
	    set r [expr ($max - $min)/double($timesteps)]
	    set res "$r"

	    $w.time configure -from $min
	    $w.time configure -to $max
	    $w.time configure -resolution $res 
	}
    }

    method setGridScalars { args } {
	set gsVarList $args;
    }
    method setGridVectors { args } {
	set gvVarList $args
    }
    method setParticleScalars { args } {
	set psVarList $args;
	puts "psVarList is now $psVarList";
    }
    method setParticleVectors { args } {
	set pvVarList $args;
	puts "pvVarList is now $pvVarList";
    }    
    
    method buildVarList { type } {
	puts "buildVarList $type $name"
	global $this-pName
	global $this-gName
	set sv ""
	set vv ""
	set c "$this-c needexecute"
	if { $type == "particleSet"} {
	    puts "... buildControlFrame $pf.1"
	    buildControlFrame $pf.1
	    #set varlist [split $psVarList]
	    for {set i 0} { $i < [llength $psVarList] } { incr i } {
		set newvar [lindex $psVarList $i]
		if { $i == 0 && [set $this-psVar] == ""} {
		    set $this-psVar $newvar
		}
		set lvar [string tolower $newvar]
		regsub \\. $lvar _ lvar
		puts "button $pf.1.1.$lvar"
		radiobutton $pf.1.1.$lvar -text $newvar \
		    -variable $this-psVar -command $c -value $newvar
		pack $pf.1.1.$lvar -side top -anchor w
		if { $newvar == [set $this-psVar] } {
		    $pf.1.1.$lvar invoke
		}
	    }
	    ## set the Particle Vector Variables
	    #set varlist [split $pvVarList]
	    for {set i 0} { $i < [llength $pvVarList] } { incr i } {
		set newvar [lindex $pvVarList $i]
		if { $i == 0 && [set $this-pvVar] == ""} {
		    set $this-pvVar $newvar
		}
		set lvar [string tolower $newvar]
		regsub \\. $lvar _ lvar
		radiobutton $pf.1.2.$lvar -text $newvar \
		    -variable $this-pvVar -command $c -value $newvar
		pack $pf.1.2.$lvar -side top -anchor w
		if { $newvar == [set $this-pvVar] } {
		    $pf.1.2.$lvar invoke
		}
	    }
	} else {

	    buildControlFrame $gf.1
	    #set varlist [split $gsVarList]
	    for {set i 0} { $i < [llength $gsVarList] } { incr i } {
		set newvar [lindex $gsVarList $i]
		if { $i == 0 && [set $this-gsVar] == ""} {
		    set $this-gsVar $newvar
		}
		set lvar [string tolower $newvar]
		regsub \\. $lvar _ lvar
		radiobutton $gf.1.1.$lvar -text $newvar \
		    -variable $this-gsVar -command $c -value $newvar
		pack $gf.1.1.$lvar -side top -anchor w
		if { $newvar == [set $this-gsVar] } {
		    $gf.1.1.$lvar invoke
		}
	    }
	    ## set the Particle Vector Variables
	    #set varlist [split $gvVarList]
	    for {set i 0} { $i < [llength $gvVarList] } { incr i } {
		set newvar [lindex $gvVarList $i]
		if { $i == 0 && [set $this-gvVar] == ""} {
		    set $this-gvVar $newvar
		}
		set lvar [string tolower $newvar]
		regsub \\. $lvar _ lvar
		radiobutton $gf.1.2.$lvar -text $newvar \
		    -variable $this-gvVar -command $c -value $newvar
		pack $gf.1.2.$lvar -side top -anchor w
		if { $newvar == [set $this-gvVar] } {
		    $gf.1.2.$lvar invoke
		}
	    }
	}   
    }    
	
    method graph { id var args } {
	global $this-psVar
	set w .graph[modname]$id
        if {[winfo exists $w]} { 
            destroy $w 
	}
	toplevel $w
#	wm minsize $w 300 300

	button $w.close -text "Close" -command "destroy $w"
	pack $w.close -side bottom -anchor s -expand yes -fill x
	
	blt::graph $w.graph -title "Particle Value" -height 250 \
		-plotbackground gray99

	set max 1e-10
	set min 1e+10

	for { set i 1 } { $i < [llength $args] } { set i [expr $i + 2]} {
	    set val [lindex $args $i]
	    if { $max < $val } { set max $val }
	    if { $min > $val } { set min $val }
	}
	if { ($max - $min) > 1000 || ($max - $min) < 1e-3 } {
	    $w.graph yaxis configure -logscale true -title $var
	} else {
	    $w.graph yaxis configure -title $var
	}

	$w.graph xaxis configure -title "Timestep" \
		-loose true
	$w.graph element create "Particle Value" -linewidth 2 -color blue \
	    -pixels 3

	pack $w.graph
	if { $args != "" } {
	    $w.graph element configure "Particle Value" -data "$args"
	}
    }
    method infoFrame { id } {
	set w .info[modname]$id
        if {[winfo exists $w]} { 
            destroy $w 
        } 

	toplevel $w
	button $w.close -text "Close" -command "destroy $w"
	pack $w.close -side bottom -expand yes -fill x
	
    }

    method infoAdd { id addbutton args } {
	set w .info[modname]$id
        if { ![winfo exists $w]} { 
            $this infoFrame $id 
        }
	

	set text [join $args]
	set var [lindex $args 0]
	set args1 [string tolower $var ]
	frame $w.$args1 -relief flat 
	pack $w.$args1 -side top -expand yes -fill x
	label $w.$args1.l -text $text
	pack $w.$args1.l -anchor w -side left   -expand yes -fill x
	if { $addbutton } {
	    button $w.$args1.b -text "Graph $var over time" \
		-command "$this-c graph $id $var"
	    pack $w.$args1.b -side left -expand yes -fill x
	}
    }
	    	    
}
