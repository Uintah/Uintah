#
#  The contents of this file are subject to the University of Utah Public
#  License (the "License"); you may not use this file except in compliance
#  with the License.
#  
#  Software distributed under the License is distributed on an "AS IS"
#  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
#  License for the specific language governing rights and limitations under
#  the License.
#  
#  The Original Source Code is SCIRun, released March 12, 2001.
#  
#  The Original Source Code was developed by the University of Utah.
#  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
#  University of Utah. All Rights Reserved.
#


catch {rename HarvardVis ""}

itcl_class Kurt_Visualization_HarvardVis {
    inherit Module
    constructor {config} {
	set name HarvardVis
	set_defaults
    }
    method set_defaults {} {
	global $this-file_name_
	set $this-file_name_ /home/sci/bigler/data/harvard/sci30.2em1.1p0.1em5.1e6
#	set $this-file_name_ /home/sci/bigler/data/harvard/sci30.small

	global $this-which_I_var_
	set $this-which_I_var_ 0
	global $this-num_timesteps_
	set $this-num_timesteps_ 0
	global $this-which_timestep_
	set $this-which_timestep_ 0
    }
    

    method add_wiv { frame var text value command } {
	radiobutton $frame.$var -text $text -relief flat \
	    -variable $this-which_I_var_ -value $value \
	    -anchor w -command $command

	pack $frame.$var -side top -fill x
    }

    method update_slider {} {
	set w .ui[modname]
#	global $this-num_timesteps_
	if [ expr [winfo exists $w] ] {
	    $w.time.s configure -from 0 -to [set $this-num_timesteps_]
	    $w.time.s configure -tickinterval [expr [set $this-num_timesteps_]/3] \
	}
    }
    
    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    wm deiconify $w
	    raise $w
	    return;
	}
	toplevel $w
	wm minsize $w 250 300
	set n "$this-c needexecute "

	frame $w.time
	pack $w.time -padx 2 -pady 2 -fill x

	scale $w.time.s -from 0 -to [set $this-num_timesteps_] -length 200 \
	    -tickinterval [expr [set $this-num_timesteps_]/3] \
	    -variable $this-which_timestep_ \
	    -orient horizontal -showvalue false \
	    -command $n -resolution 1
	pack $w.time.s -side top -fill x
	
	frame $w.wiv
	pack $w.wiv -padx 2 -pady 2 -fill x

	label $w.wiv.l -text "Value Used"
	pack $w.wiv.l -side top -fill x

	add_wiv $w.wiv "n" "Number of particles" 0 $n
	add_wiv $w.wiv "ncum" "Cumulative number of particles starting from \n the most massive bin" 1 $n
	add_wiv $w.wiv "ecc" "Orbital eccentricity of particles" 2 $n
	add_wiv $w.wiv "h_scl" "Scaled height above the disk midplane" 3 $n
	add_wiv $w.wiv "mtot" "Total mass in mass bin (grams)" 4 $n
	add_wiv $w.wiv "mi" "Average mass of particles in a mass bin (grams)" 5 $n

	add_wiv $w.wiv "radius" "Radius" 10 $n
	add_wiv $w.wiv "kepvel" "Velocity of a circular orbit" 11 $n
	add_wiv $w.wiv "massinzone" "Total mass in planetasimals" 12 $n
	add_wiv $w.wiv "massloss" "Amount of mass lost to particles smaller \n than the smallest bin" 13 $n
	add_wiv $w.wiv "masslostrate" "Rate of mass loss" 14 $n

	

    }
}
