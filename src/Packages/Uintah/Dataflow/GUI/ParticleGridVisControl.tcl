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

itcl_class Uintah_MPMViz_ParticleGridVisControl { 
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
	global $this-sVar;
	global $this-vVar;
	global $this-psVar;
	global $this-pvVar;
	global $this-sMaterial;
	global $this-vMaterial;
	global $this-pMaterial;

	set $this-sVar ""
	set $this-vVar ""
	set $this-pVar ""
	set $this-sMaterial 0
	set $this-vMaterial 0
	set $this-pMaterial 0
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
	
	#button $w.b -text "close" -command "destroy $w"
	#button $w.b -text "Make Frames" -command "$this makeFrames $w.f"
	#pack $w.b -side top -fill x -padx 2 -pady 2

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

	set w .ui[modname] 
	if { [winfo exists $w.b] } {
	    destroy $w.b
	}
	button $w.b -text "Close" -command "destroy $w"
	pack $w.b -side top -fill x -padx 2 -pady 2

	puts "adding scalarnames"
  	$this addScalarNames
	puts "adding vector names"
  	$this addVectorNames
	puts "adding materialnames"
  	$this addMaterials

    }

    method destroyFrames {} {
	set w .ui[modname] 

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

    method addMaterial { materialNum } {
	global $this-sMaterial;
	global $this-vMaterial;
	global $this-pMaterial;

	set c "$this-c needexecute"
	radiobutton $sf.1.2.$materialNum -text "material$materialNum" \
	    -variable $this-sMaterial -command $c -value $materialNum
	pack $sf.1.2.$materialNum -side top

	radiobutton $vf.1.2.$materialNum -text "material$materialNum" \
	    -variable $this-vMaterial -command $c -value $materialNum
	pack $vf.1.2.$materialNum -side top

	if { [$this-c hasParticles $materialNum]} {
	    radiobutton $pf.1.2.$materialNum -text "material$materialNum" \
		-variable $this-pMaterial -command $c -value $materialNum
	    pack $pf.1.2.$materialNum -side top
	}
    }
    
    method addScalarNames {} {
	set names [$this-c getScalarNames]
	set varlist [split $names]
	for {set i 0} { $i < [llength $varlist] } { incr i } {
	    $this addVar "scalar" [lindex $varlist $i]
	}
    }

    method addVectorNames {} {
	set names [$this-c getVectorNames]
	set varlist [split $names]
	for {set i 0} { $i < [llength $varlist] } { incr i } {
	    $this addVar "vector" [lindex $varlist $i]
	}
    }

    method addMaterials {} {
	set nmaterials [$this-c getNMaterials]
	for {set i 1} { $i <= $nmaterials } { incr i } {
	    $this addMaterial $i
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
	    
}
