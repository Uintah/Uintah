########################################
#CLASS
#    VizControl
#    Visualization control for simulation data that contains
#    information on both a regular grid in particle sets.
#OVERVIEW TEXT
#    This module receives a ParticleGridReader object.  The user
#    interface is dynamically created based information provided by the
#    ParticleGridReader.  The user can then select which variables he/she
#    wishes to view in a visualization.
#KEYWORDS
#    ParticleGridReader, Material/Particle Method
#AUTHOR
#    Kurt Zimmerman
#    Department of Computer Science
#    University of Utah
#    January 1999
#    Copyright (C) 1999 SCI Group
#LOG
#    Created January 5, 1999
########################################

catch {rename FieldExtractor ""}

itcl_class Uintah_Selectors_FieldExtractor { 
    inherit Module 

    protected varList ""
    protected gf ""
    protected label_text ""

    constructor {config} { 
    } 

    method set_defaults {} { 
	global $this-tcl_status
	global $this-sVar;
	global $this-sMatNum;
	global $this-level
        global $this-remove_boundary_cells
        global $this-want_all_levels
	set $this-sVar ""
	set $this-sMatNum 0
	set $this-level 0
	set $this-tcl_status "Idle"
        set $this-remove_boundary_cells 0
        set $this-want_all_levels 0
    } 
    
    method ui {} { 
        set w .ui[modname] 

        if {[winfo exists $w]} {
            return
        }
    }

    method buildTopLevel {} {
        set w .ui[modname] 

        if {[winfo exists $w]} { 
            return
        } 
	
        toplevel $w 
	wm withdraw $w
	
	set n "$this-c needexecute"
	frame $w.f -relief flat
 	pack $w.f -side top -expand yes -fill both

	makeFrames $w.f
       
        # add frame for SCI Button Panel
        frame $w.control -relief flat
        pack $w.control -side top -expand yes -fill both
	makeSciButtonPanel $w.control $w $this
	moveToCursor $w
    }


    method isVisible {} {
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
	global $this-sVar;

	frame $parent.f0 -relief groove -borderwidth 2 
	label $parent.f0.label -text $label_text
	pack $parent.f0 -side left -expand yes -fill both -padx 2
	pack $parent.f0.label -side top
	    
	frame $parent.f0.1 -relief flat -borderwidth 2
	pack $parent.f0.1 -side top -expand yes -fill both -padx 2
	buildControlFrame $parent.f0.1

        checkbutton $parent.f0.cb -text "Remove Boundary Cells" \
            -variable $this-remove_boundary_cells \
            -command "$this-c needexecute"
        pack $parent.f0.cb -side bottom
	    
	set gf $parent.f0
    }
    

    method buildControlFrame { name } {

	if { [winfo exists $name.1] } {
	    destroy $name.1
	}

	frame $name.1 -relief groove -borderwidth 2
	pack $name.1 -side left  -expand yes -fill both -padx 2
    }

    method destroyFrames {} {
	set w .ui[modname] 

	destroy $gf
	set varList ""
	set gf ""
    }
    method setVars { args } {
	set varList $args;
    }


    method buildLevels { levels } {
	set w $gf
	set buttontype radiobutton
	set c "$this-c needexecute"
	frame $w.lf -relief flat -borderwidth 2
	pack $w.lf -side top
	label $w.lf.l -text Levels
	pack $w.lf.l -side top
	frame $w.lf.bf -relief flat
	pack $w.lf.bf -side top -expand yes -fill both
        for {set j 0} { $j < $levels } {incr j} {
            $buttontype $w.lf.bf.b$j -text $j \
                -variable $this-level  -value $j -command $c
            pack $w.lf.bf.b$j -side left
            bind $w.lf.bf.b$j <ButtonRelease> "set $this-want_all_levels 0"
        }
 
       if {$levels > 1} {
            $buttontype $w.lf.bf.b$levels -text all \
                -variable $this-level -value $levels -command $c
            pack $w.lf.bf.b$levels -side left
            bind $w.lf.bf.b$j <ButtonRelease> "set $this-want_all_levels 1"
        }
        
        if { [set $this-want_all_levels] == 1 } {
            set $this-level $levels
        } elseif { [set $this-level] > $levels } {
	    set $this-level [expr $levels -1]
	}
    }
	
 
    method buildMaterials { args } {
        set parent $gf
        set buttontype radiobutton
        set c "$this-c needexecute"
        frame $parent.m -relief flat -borderwidth 2
        pack $parent.m -side top
        label $parent.m.l -text Materials
        pack $parent.m.l -side top
        frame $parent.m.m -relief flat 
        pack $parent.m.m  -side top -expand yes -fill both
        for {set j 0} { $j < [llength $args] } {incr j} {
            set i [lindex $args $j]
            $buttontype $parent.m.m.b$i -text $i \
                -variable $this-sMatNum -command $c -value $i
            pack $parent.m.m.b$i -side left
        }

        if { [set $this-sMatNum] != "" } {
            if { [lsearch $args [set $this-sMatNum]] == -1 } {
                if { [llength $args] == 0 } {
                    set $this-sMatNum ""
                } else {
                    set $this-sMatNum [lindex $args 0]
                }
            }
        }
    }

    method isOn { bval } {
	return  [set $this-$bval]
    }

    method buildVarList {} {
	set sv ""
	set c "$this-c needexecute"

	buildControlFrame $gf.1
	for {set i 0} { $i < [llength $varList] } { incr i } {
	    set newvar [lindex $varList $i]
	    if { $i == 0 && [set $this-sVar] == ""} {
		set $this-sVar $newvar
	    }
	    set lvar [string tolower $newvar]
	    regsub \\. $lvar _ lvar
	    radiobutton $gf.1.1.$lvar -text $newvar \
		-variable $this-sVar -command $c -value $newvar
	    pack $gf.1.1.$lvar -side top -anchor w
#	    if { $newvar == [set $this-sVar] } {
#		$gf.1.1.$lvar invoke
#	    }
	}
    }
}    
	    	    
