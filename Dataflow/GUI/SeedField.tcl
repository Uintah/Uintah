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

package require Iwidgets 3.0

itcl_class SCIRun_Fields_SeedField {
    inherit Module
    constructor {config} {
        set name SeedField

        set_defaults
    }

    method set_defaults {} {
	global $this-wtype
	global $this-maxseeds
	global $this-dist
	global $this-numseeds
	global $this-rngseed
	global $this-whichtab
	set $this-wtype rake
	set $this-maxseeds 15
	set $this-dist importance
	set $this-numseeds 10
	set $this-rngseed 1234
	set $this-whichtab widget
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }
        toplevel $w

	iwidgets::tabnotebook  $w.tabs -raiseselect true -width 350
	pack $w.tabs -side top
	set wtab [$w.tabs add -label "Widget" \
		  -command "set $this-whichtab widget"]
	set rtab [$w.tabs add -label "Random" \
		  -command "set $this-whichtab random"]
	$w.tabs view "Widget"

	iwidgets::Labeledframe $wtab.type -labelpos nw -labeltext "Widget type"
	pack $wtab.type 
	set type [$wtab.type childsite]
	radiobutton $type.rake -var $this-wtype -value rake -text "Rake" 
	radiobutton $type.ring -var $this-wtype -value ring -text "Ring" 
	radiobutton $type.frame -var $this-wtype -value frame \
		    -text "Frame"
	pack $type.rake $type.ring $type.frame -side left -padx 5 -pady 5
	frame $wtab.f1 
	pack $wtab.f1 -side top 
	label $wtab.f1.maxseeds_l -text "Maximum number of seeds" -width 23 \
              -anchor w
	entry $wtab.f1.maxseeds -text $this-maxseeds -width 10
	pack $wtab.f1.maxseeds_l $wtab.f1.maxseeds -side left

	iwidgets::Labeledframe $rtab.dist -labelpos nw \
		               -labeltext "Distribution"
	pack $rtab.dist
	set dist [$rtab.dist childsite]
	radiobutton $dist.uniwi -var $this-dist -value importance \
		    -text "Importance" 
	radiobutton $dist.uniwoi -var $this-dist -value uniform \
	            -text "Uniform" 
	radiobutton $dist.scat -var $this-dist -value scattered \
	            -text "Scattered" 
	pack $dist.uniwi $dist.uniwoi $dist.scat -side left -padx 5 -pady 5
	frame $rtab.f1 
	pack $rtab.f1 -side top 
	label $rtab.f1.rngseed_l -text "Seed value for RNG" -width 23 \
              -anchor w
	entry $rtab.f1.rngseed -text $this-rngseed -width 10
	pack $rtab.f1.rngseed_l $rtab.f1.rngseed -side left -anchor w
	frame $rtab.f2
	pack $rtab.f2 -side top
	label $rtab.f2.numseeds_l -text "number of seeds" -width 23 \
              -anchor w
	entry $rtab.f2.numseeds -text $this-numseeds -width 10
	pack $rtab.f2.numseeds_l $rtab.f2.numseeds -side left

	

	button $w.execute -text "Execute" -command "$this-c needexecute"
	pack $w.execute -padx 5 -pady 5
    }
}


