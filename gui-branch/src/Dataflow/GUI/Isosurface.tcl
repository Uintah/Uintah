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

itcl::class SCIRun_Visualization_Isosurface {
    inherit ModuleGui

    public variable isoval      0
    public variable isoval_min  0
    public variable isoval_max  4095
    public variable continuous  0
    public variable extract_from_new_field 0
    public variable algorithm   0
    public variable type        ""
    public variable gen         0
    public variable build_trisurf 0
    public variable np          1
    public variable active_tab  "MC"
    public variable update_type Release
    public variable opt

    # SAGE vars
    public variable visibility   0
    public variable value        1
    public variable scan         1
    public variable bbox         1
    public variable cutoff_depth 8
    public variable reduce       1
    public variable all          0
    public variable rebuild      0
    public variable min_size     1
    public variable poll         0

    constructor {} {
	set name Isosurface

	trace variable [scope active_tab] w "$this switch_to_active_tab"
	trace variable [scope update_type] w "$this set_update_type"
	auto-var-set [scope active_tab]
    }

    method switch_to_active_tab {name1 name2 op}
    method ui {} 
    method change_isoval { n } 
    method set-isoval {} 
    method orient { tab page { val 4 }} 
    method select-alg { alg } 
    method set_update_type { name1 name2 op } 
    method update-type { w } 
    method set_info { type_ generation } 
    method set_minmax {min max} 
}


body SCIRun_Visualization_Isosurface::switch_to_active_tab {name1 name2 op} {
    set window .ui[modname]
    if {[winfo exists $window]} {
	set mf [$window.f.meth childsite]
	$mf.tabs view $active_tab
    }
}

body SCIRun_Visualization_Isosurface::ui {} {
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
	-variable [scope isoval] \
	-from $isoval_min -to $isoval_max \
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
    
    iwidgets::labeledframe $w.f.info -labelpos nw -labeltext "Info"
    set info [$w.f.info childsite]
    
    label $info.type_label -text "File Type: " 
    label $info.type -text $type
    label $info.gen_label -text "Generation: "
    label $info.gen -text $gen
    
    pack $info.type_label $info.type $info.gen_label $info.gen -side left
    pack $w.f.info -side top -anchor w
    
    #  Options
    
    iwidgets::labeledframe $w.f.opt -labelpos nw -labeltext "Options"
    set opt [$w.f.opt childsite]
    
    iwidgets::optionmenu $opt.update -labeltext "Update:" \
	-labelpos w -command "$this update-type $opt.update"
    
    $opt.update insert end Release Manual Auto
    $opt.update select Manual
    #$update_type
    
    set update $opt.update
    
    checkbutton $opt.buildsurf -text "Build TriSurf" \
	-variable [scope build_trisurf]
    
    checkbutton $opt.aefnf -text "Auto Extract from New Field" \
	-relief flat -variable [scope extract_from_new_field]
    
    pack $opt.update $opt.aefnf $opt.buildsurf -side top -anchor w
    pack $w.f.opt -side top -anchor w
    
    
    #  Methods
    iwidgets::labeledframe $w.f.meth -labelpos nw -labeltext "Methods"
    set mf [$w.f.meth childsite]
    
    iwidgets::tabnotebook  $mf.tabs -raiseselect true 
    #-fill both
    pack $mf.tabs -side top
    
    #  Method:
    
    set alg [$mf.tabs add -label "MC" -command "$this select-alg 0"]
    
    scale $alg.np -label "np:" \
	-variable [scope np] \
	-from 1 -to 8 \
	-showvalue true \
	-orient horizontal
    
    pack $alg.np -side left -fill x
    
    set alg [$mf.tabs add -label "NOISE"  -command "$this select-alg 1"]
    
    $mf.tabs view $active_tab
    $mf.tabs configure -tabpos "n"
    
    pack $mf.tabs -side top
    pack $w.f.meth -side top
}

body SCIRun_Visualization_Isosurface::change_isoval { n } {
    if { $continuous == 1.0 } {
	eval "$this-c needexecute"
    }
}

body SCIRun_Visualization_Isosurface::set-isoval {} {
    set type [$opt.update get]
    if { $type == "Release" } {
	eval "$this-c needexecute"
    }
}

body SCIRun_Visualization_Isosurface::orient { tab page { val 4 }} {
    $tab.tabs configure -tabpos [$page.orient get]
}

body SCIRun_Visualization_Isosurface::select-alg { alg } {
    if { $alg == 0 } {
	set active_tab "MC"
    } else {
	set active_tab "NOISE"
    }
    if { $algorithm != $alg } {
	set algorithm $alg
	if { $continuous == 1.0 } {
	    eval "$this-c needexecute"
	}
    }
}

body SCIRun_Visualization_Isosurface::set_update_type { name1 name2 op } {
    puts stdout "set update type"
    puts stdout $name1
    puts stdout $name2
    puts stdout $op
    puts stdout $update_type
    set window .ui[modname]
    if {[winfo exists $window]} {
	set opt [$window.f.opt childsite]
	$opt.update select $update_type
    }
}

body SCIRun_Visualization_Isosurface::update-type { w } {
    set update_type [$w get]
    puts "update to update_type current is $continuous"
    if { $update_type == "Auto" } {
	set continuous 1
    } else {
	set continuous 0
    }
}

body SCIRun_Visualization_Isosurface::set_info { type_ generation } {
    set type $type_
    set gen $generation
    
    set w .ui[modname]    
    if [ expr [winfo exists $w] ] {
	set info [$w.f.info childsite]
	
	$info.type configure -text $type 
	$info.gen  configure -text $generation
    }
}

body SCIRun_Visualization_Isosurface::set_minmax {min max} {
    set w .ui[modname]
    
    set isoval_min $min
    set isoval_max $max
    if [ expr [winfo exists $w] ] {
	$w.f.isoval configure -from $min -to $max
    }
}
