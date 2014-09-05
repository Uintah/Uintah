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


catch {rename Isosurface ""}

package require Iwidgets 3.0   

itcl_class SCIRun_Visualization_Isosurface {
    inherit Module

    constructor {config} {
	set name Isosurface
	set_defaults
    }

    method set_defaults {} {
	global $this-isoval-min 
	global $this-isoval-max 
	global $this-continuous
	global $this-extract-from-new-field
	global $this-algorithm
	global $this-type
	global $this-gen
	global $this-build_trisurf
	
	set $this-isoval-min 0
	set $this-isoval-max 4095
	set $this-continuous 0
	set $this-extract-from-new-field 0
	set $this-algorithm 0
	set $this-type ""
	set $this-gen 0
	set $this-build_trisurf 0

	# SAGE vars
	global $this-visibility $this-value $this-scan
	global $this-bbox
	global $this-cutoff_depth 
	global $this-reduce
	global $this-all
	global $this-rebuild
	global $this-min_size
	global $this-poll

	set $this-visiblilty 0
	set $this-value 1
	set $this-scan 1
	set $this-bbox 1
	set $this-reduce 1
	set $this-all 0
	set $this-rebuild 0
	set $this-min_size 1
	set $this-poll 0

#	global $this-update
#	set $this-update 0
    }

    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}

	toplevel $w
	frame $w.f 
	pack $w.f -padx 2 -pady 2 -expand 1 -fill x
	set n "$this-c needexecute "

	scale $w.f.isoval -label "Iso Value:" \
	    -variable $this-isoval \
	    -from [set $this-isoval-min] -to [set $this-isoval-max] \
	    -length 5c \
	    -showvalue true \
	    -orient horizontal  \
	    -digits 5 \
	    -resolution 0.001 \
	    -command "$this change_isoval"

	bind $w.f.isoval <ButtonRelease> "$this set-isoval"
	
	button $w.f.extract -text "Extract" -command "$this-c needexecute"
	pack $w.f.isoval  -fill x
	pack $w.f.extract

	#  Info
	
	global $this-type
	global $this-gen

	iwidgets::Labeledframe $w.f.info -labelpos nw -labeltext "Info"
	set info [$w.f.info childsite]
	
	label $info.type_label -text "File Type: " 
	label $info.type -text [set $this-type]
	label $info.gen_label -text "Generation: "
	label $info.gen -text [set $this-gen]

	pack $info.type_label $info.type $info.gen_label $info.gen -side left
	pack $w.f.info -side top -anchor w

	#  Options

	iwidgets::Labeledframe $w.f.opt -labelpos nw -labeltext "Options"
	set opt [$w.f.opt childsite]
	
	iwidgets::optionmenu $opt.update -labeltext "Update:" \
	    -labelpos w -command "$this update-type $opt.update"
	
	$opt.update insert end "on release" Manual Auto

	global $this-update
	set $this-update $opt.update

	global $this-build_trisurf
	checkbutton $opt.buildsurf -text "Build TriSurf" \
		-variable $this-build_trisurf

	checkbutton $opt.aefnf -text "Auto Extract from New Field" \
	    -relief flat -variable $this-extract-from-new-field 

	pack $opt.update $opt.aefnf $opt.buildsurf -side top -anchor w
	pack $w.f.opt -side top -anchor w


	#  Methods
	iwidgets::Labeledframe $w.f.method -labelpos nw -labeltext "Methods"
	set mf [$w.f.method childsite]
	
	iwidgets::tabnotebook  $mf.tabs -raiseselect true 
#-fill both
	pack $mf.tabs -side top

	#  Method:

	set alg [$mf.tabs add -label "MC" -command "$this select-alg 0"]
	set alg [$mf.tabs add -label "NOISE"  -command "$this select-alg 1"]
	set alg [$mf.tabs add -label "SAGE"  -command "$this select-alg 2"]

# 	iwidgets::checkbox $alg.prune -labeltext "Prune"
# 	$alg.prune add value  -text "Value"  -variable $this-value
# 	$alg.prune add bbox   -text "BBox"   -variable $this-bbox
# 	$alg.prune add scan   -text "Scan"   -variable $this-scan
# 	$alg.prune add points -text "Points" -variable $this-reduce

# 	iwidgets::checkbox $alg.opt -labeltext "Options"
# 	$alg.opt add poll   -text "Poll"   -variable $this-poll
# 	$alg.opt add size   -text "Size"   -variable $this-min_size
#  	$alg.opt add all    -text "All"    -variable $this-all

# 	pack $alg.prune $alg.opt -side left -anchor n
	
#	set alg [$mf.tabs add -label "Opt"]

#	iwidgets::radiobox $alg.orient -labeltext "Tabs position:" \
#	    -command "$this orient $mf $alg" 

# 	$alg.orient add n -text "n"
# 	$alg.orient add w -text "w"
# 	$alg.orient add e -text "e"
# 	$alg.orient add s -text "s"

# 	$alg.orient select n


# 	pack $alg.orient -padx 4 -pady 4 -anchor w

	$mf.tabs view "MC"
	$mf.tabs configure -tabpos "n"
#[$alg.orient get]

	
	pack $mf.tabs -side top

	pack $w.f.method -side top


# 	button $w.f.reset_view -text "Reset View" -relief raised -command $n





# 	pack $w.f.prune -side bottom

# 	pack $w.f.prune.reduce $w.f.prune.skip \
# 	    $w.f.prune.scan $w.f.prune.value $w.f.prune.visibility \
# 	    $w.f.prune.all $w.f.prune.size $w.f.prune.poll \
# 	    -padx 2 -pady 3 -expand 1 -fill x  -side left
	
# #	raiseGL
#     }
    
    method change_isoval { n } {
        global $this-continuous
	
	if { [set $this-continuous] == 1.0 } {
	    eval "$this-c needexecute"
	}
    }
	
    method set-isoval {} {
	global $this-update

	set type [[set $this-update] get]
	if { $type == "on release" } {
	    eval "$this-c needexecute"
	}
    }
	
    method orient { tab page { val 4 }} {
	global $page
	global $tab
	
	$tab.tabs configure -tabpos [$page.orient get]
    }

    method select-alg { alg } {
	global $this-algorithm
	
	if { [set $this-algorithm] != $alg } {
	    set $this-algorithm $alg
	    if { [set $this-continuous] == 1.0 } {
		eval "$this-c needexecute"
	    }
	}
    }

    method update-type { w } {
	global $w
	global $this-continuous

	set type [$w get]
	puts "update to $type current is [set $this-continuous]"
	if { $type == "Auto" } {
	    set $this-continuous 1
	} else {
	    set $this-continuous 0
	}
    }


    method set_info { type generation } {
	global $this-type
	global $this-gen

	set $this-type $type
	set $this-gen $generation

	set w .ui[modname]    
	if [ expr [winfo exists $w] ] {
	    set info [$w.f.info childsite]
	    
	    $info.type configure -text $type 
	    $info.gen  configure -text $generation
	}
    }

    method set_minmax {min max} {
	set w .ui[modname]

	global $this-isoval-min $this-isoval-max
	set $this-isoval-min $min
	set $this-isoval-max $max
	if [ expr [winfo exists $w] ] {
	    $w.f.isoval configure -from $min -to $max
	}
    }
}
