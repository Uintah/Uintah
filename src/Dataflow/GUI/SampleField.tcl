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

itcl_class SCIRun_FieldsCreate_SampleField {
    inherit Module
    constructor {config} {
        set name SampleField

        set_defaults
    }

    method set_defaults {} {
	global $this-wtype
	global $this-maxseeds
	global $this-dist
	global $this-numseeds
	global $this-rngseed
	global $this-rnginc
	global $this-whichtab
        global $this-clamp
        global $this-autoexecute
	set $this-wtype rake
	set $this-maxseeds 15
	set $this-dist uniuni
	set $this-numseeds 10
	set $this-rngseed 1
	set $this-rnginc 1
	set $this-whichtab Widget
        set $this-clamp 0
        set $this-autoexecute 1
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
	    wm deiconify $w
            raise $w
            return
        }
        toplevel $w

	iwidgets::tabnotebook  $w.tabs -raiseselect true \
		               -width 350 -height 200
	pack $w.tabs -side top
	set wtab [$w.tabs add -label "Widget" \
		  -command "set $this-whichtab Widget"]
	set rtab [$w.tabs add -label "Random" \
		  -command "set $this-whichtab Random"]
	if {"[set $this-whichtab]"=="Widget"} {
	    $w.tabs view 0
	} else {
	    $w.tabs view 1
	}

	iwidgets::Labeledframe $wtab.type -labelpos nw \
		               -labeltext "Widget type"
	set type [$wtab.type childsite]
	radiobutton $type.rake -var $this-wtype -value rake -text "Rake" 
	radiobutton $type.ring -var $this-wtype -value ring -text "Ring" \
		    -state disabled -dis #444
	radiobutton $type.frame -var $this-wtype -value frame \
		    -text "Frame" -state disabled -dis #444
	pack $type.rake $type.ring $type.frame -side left -padx 5 -pady 5 -fill both -expand yes

	frame $wtab.f1 
	label $wtab.f1.maxseeds_l -text "Maximum number of samples" -width 25 \
              -anchor w
	entry $wtab.f1.maxseeds -text $this-maxseeds -width 10
	pack $wtab.f1.maxseeds_l $wtab.f1.maxseeds -side left

	checkbutton $wtab.auto -text "Execute automatically" \
		-variable $this-autoexecute

	pack $wtab.type $wtab.f1 $wtab.auto -side top -fill x -pady 5 -anchor w


	frame $rtab.f2
	pack $rtab.f2 -side top -anchor w -padx 8
	label $rtab.f2.numseeds_l -text "Number of samples" -width 23 -anchor w
	entry $rtab.f2.numseeds -text $this-numseeds -width 10
	pack $rtab.f2.numseeds_l $rtab.f2.numseeds -side left -anchor w

	iwidgets::Labeledframe $rtab.dist -labelpos nw \
		               -labeltext "Distribution"
	pack $rtab.dist -fill x -e y
	set dist [$rtab.dist childsite]
	frame $dist.imp 
	frame $dist.uni 
	pack $dist.uni $dist.imp -side left -f both -e y

	label $dist.imp.label -text "Importance Weighted"
	radiobutton $dist.imp.uni -var $this-dist -value impuni \
		    -text "Uniform" 
	radiobutton $dist.imp.scat -var $this-dist -value impscat \
		    -text "Scattered"
	label $dist.uni.label -text "Not Weighted"
	radiobutton $dist.uni.uni -var $this-dist -value uniuni \
	            -text "Uniform" 
	radiobutton $dist.uni.scat -var $this-dist -value uniscat \
	            -text "Scattered" 
	pack $dist.imp.label $dist.imp.uni $dist.imp.scat \
	     $dist.uni.label $dist.uni.uni $dist.uni.scat \
	     -side top -padx 5 -pady 2 -anchor w

	checkbutton $rtab.rnginc -text "Increment RNG seed on execute" \
	    -var $this-rnginc
	pack $rtab.rnginc -side top -anchor w -padx 8

	frame $rtab.f1 
	pack $rtab.f1 -side top  -anchor w
	label $rtab.f1.rngseed_l -text "Seed value for RNG" -width 23 -anchor w
	entry $rtab.f1.rngseed -text $this-rngseed -width 10
	pack $rtab.f1.rngseed_l $rtab.f1.rngseed -side left -anchor w -padx 8

        frame $rtab.f3
        pack $rtab.f3 -side top -anchor w
        checkbutton $rtab.f3.clamp -text "Clamp to nodes" -var $this-clamp
        pack $rtab.f3.clamp -anchor w -padx 8

	makeSciButtonPanel $w $w $this
	moveToCursor $w
    }
}


