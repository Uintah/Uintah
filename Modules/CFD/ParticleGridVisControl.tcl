########################################
#CLASS
#    ParticleGridVisControl
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

itcl_class ParticleGridVisControl { 
    inherit Module 

    protected sf ""
    protected vf ""
    protected pf ""

    constructor {config} { 
        set name ParticleGridVisControl 
        set_defaults

    } 

    method set_defaults {} { 
        global $this-tcl_status 
	global $this-tcl_status;
	global $this-sVar;
	global $this-vVar;
	global $this-psVar;
	global $this-pvVar;
	global $this-sFluid;
	global $this-vFluid;
	global $this-pFluid;
    } 
    
    method ui {} { 
        set w .ui$this 

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
	
	#button $w.b -text "close" -command "destroy $w"
	#button $w.b -text "Make Frames" -command "$this makeFrames $w.f"
	#pack $w.b -side top -fill x -padx 2 -pady 2

	makeFrames $w.f
	
    }


    method Visible {} {
	if {[winfo exists .ui$this]} {
	    return 1
	} else {
	    return 0
	}
    }
    
    method Rebuild {} {
	set w .ui$this

	$this destroyFrames
	$this makeFrames $w.f
    }

    method makeFrames { parent } {
	global $this-sVar;
	global $this-vVar;
	global $this-psVar;
	global $this-pvVar;
	set Labels [list ""  "Scalars" "Vectors" "Particles"]
	
	for { set i 1 } { $i < 4 } { incr i } {
	    frame $parent.f$i -relief groove -borderwidth 2 
	    label $parent.f$i.label -text [lindex $Labels $i] 
	    pack $parent.f$i -side left -expand yes -fill both -padx 2
	    pack $parent.f$i.label -side top
	    
	    frame $parent.f$i.1 -relief flat -borderwidth 2
	    pack $parent.f$i.1 -expand yes -fill both -padx 2
	    
	    frame $parent.f$i.1.1 -relief flat -borderwidth 2
	    pack $parent.f$i.1.1 -side left  -expand yes -fill both -padx 2
	    frame $parent.f$i.1.2 -relief flat -borderwidth 2
	    pack $parent.f$i.1.2 -side left -expand yes -fill both -padx 2
	}

	set sf $parent.f1
	set vf $parent.f2
	set pf $parent.f3

	frame $pf.1.1.1 -relief groove -borderwidth 2
	frame $pf.1.1.2 -relief groove -borderwidth 2
	pack $pf.1.1.1 $pf.1.1.2 -side top -expand yes -fill both -padx 2

	set w .ui$this 
	if { [winfo exists $w.b] } {
	    destroy $w.b
	}
	button $w.b -text "Close" -command "destroy $w"
	pack $w.b -side top -fill x -padx 2 -pady 2

	puts "adding scalarvars"
  	$this addScalarVars
	puts "adding vector vars"
  	$this addVectorVars
	puts "adding fluidvars"
  	$this addFluids

    }

    method destroyFrames {} {
	set w .ui$this 

	destroy $sf
	destroy $vf
	destroy $pf
	if { [winfo exists $w.b] } {
	    destroy $w.b
	}
	set sf ""
	set vf ""
	set pf ""

#	button $w.b -text "Make Frames" -command "$this makeFrames $w.f"
#	pack $w.b -side top -fill x -padx 2 -pady 2

    }

    method addVar { vartype var } { # vartype is scalar or vector
	global $this-sVar;
	global $this-vVar;
	global $this-psVar;
	global $this-pvVar;

	set lvar [string tolower $var]

	set c "$this-c needexecute"
	if { $vartype == "scalar"} {
	    if { ($sf != "") && ($pf != "")  } {
		radiobutton $sf.1.1.$lvar -text $var \
		    -variable $this-sVar -command $c -value $var
		pack $sf.1.1.$lvar -side top -anchor w
		
		radiobutton $pf.1.1.1.$lvar -text $var \
		    -variable $this-psVar -command $c -value $var
		pack $pf.1.1.1.$lvar -side top -anchor w
	    } else {
		return
	    }
	} elseif { $vartype == "vector" } {
	    if { ( $vf != "" ) && ( $pf != "") } {
		radiobutton $vf.1.1.$lvar -text $var \
		    -variable $this-vVar -command $c -value $var
		pack $vf.1.1.$lvar -side top  -anchor w

		radiobutton $pf.1.1.2.$lvar -text $var \
		    -variable $this-pvVar -command $c -value $var
		pack $pf.1.1.2.$lvar -side top -anchor w
	    } else {
		return
	    }
	}
    }

    method addFluid { fluidNum } {
	global $this-sFluid;
	global $this-vFluid;
	global $this-pFluid;

	set c "$this-c needexecute"
	radiobutton $sf.1.2.$fluidNum -text "fluid$fluidNum" \
	    -variable $this-sFluid -command $c -value $fluidNum
	pack $sf.1.2.$fluidNum -side top

	radiobutton $vf.1.2.$fluidNum -text "fluid$fluidNum" \
	    -variable $this-vFluid -command $c -value $fluidNum
	pack $vf.1.2.$fluidNum -side top

	if { [$this-c hasParticles $fluidNum]} {
	    radiobutton $pf.1.2.$fluidNum -text "fluid$fluidNum" \
		-variable $this-pFluid -command $c -value $fluidNum
	    pack $pf.1.2.$fluidNum -side top
	}
    }
    
    method addScalarVars {} {
	set vars [$this-c getScalarVars]
	set varlist [split $vars]
	for {set i 0} { $i < [llength $varlist] } { incr i } {
	    $this addVar "scalar" [lindex $varlist $i]
	}
    }

    method addVectorVars {} {
	set vars [$this-c getVectorVars]
	set varlist [split $vars]
	for {set i 0} { $i < [llength $varlist] } { incr i } {
	    $this addVar "vector" [lindex $varlist $i]
	}
    }

    method addFluids {} {
	set nfluids [$this-c getNFluids]
	for {set i 1} { $i <= $nfluids } { incr i } {
	    $this addFluid $i
	}
    }

    
    method infoFrame { id } {
	set w .info$this$id
        if {[winfo exists $w]} { 
            destroy $w 
        } 

	toplevel $w
	button $w.close -text "Close" -command "destroy $w"
	pack $w.close -side bottom -expand yes -fill x
	
    }

    method infoAdd { id addbutton args } {
	set w .info$this$id
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

    method graph { id  args } {
	global $this-psVar
	set w .graph$this$id
        if {[winfo exists $w]} { 
            destroy $w 
	}
	toplevel $w
#	wm minsize $w 300 300

	button $w.close -text "Close" -command "destroy $w"
	pack $w.close -side bottom -anchor s -expand yes -fill x
	
	blt_graph $w.graph -title "Particle Value" -height 250 \
		-plotbackground gray99

	set max 1e-10
	set min 1e+10

	for { set i 1 } { $i < [llength $args] } { set i [expr $i + 2]} {
	    set val [lindex $args $i]
	    if { $max < $val } { set max $val }
	    if { $min > $val } { set min $val }
	}
	if { ($max - $min) > 1000 } {
	    $w.graph yaxis configure -logscale true -title [set $this-psVar]
	} else {
	    $w.graph yaxis configure -title [set $this-psVar]
	}

	$w.graph xaxis configure -title "Timestep" \
		-loose true
	$w.graph element create "Particle Value" -linewidth 2 -foreground blue
	set startTime [lindex $args 0]
	set value [lindex $args 1]
	$w.graph element configure "Particle Value" -data "$startTime $value"

	pack $w.graph
	if { $args != "" } {
	    $w.graph element append "Particle Value" "$args"
	}
    }
	    
}
