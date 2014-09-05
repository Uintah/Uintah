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

#  BaseWidget.tcl
#  Written by:
#   James Purciful
#   Department of Computer Science
#   University of Utah
#   Mar. 1995
#  Copyright (C) 1995 SCI Group


if {[info commands BaseWidget] != ""} {
	return
}

itcl_class BaseWidget {
    inherit Module
    constructor {config} {
	set name BaseWidget
	global $this-material
	initMaterial $this-material
    }

    method config {config} {
    }

    public name
    public matwin

    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	wm minsize $w 100 100

	frame $w.f
	Base_ui $w.f
	pack $w.f
    }

    method Base_ui {wid} {
	frame $wid.opts
	set opts $wid.opts
	button $opts.close -text "Close" -command "destroy .ui[modname]"
	button $opts.nextmode -text "NextMode" -command "$this-c nextmode"
	button $opts.help -text "Help" -command "$this help"
	pack $opts.close $opts.nextmode $opts.help -side left -padx 2 -pady 2 -anchor e
	
	frame $wid.scale
	label $wid.scale.l -text "Scale"
	button $wid.scale.incr -text "++" -command "$this-c scale 1.25"
	button $wid.scale.incr2 -text "+" -command "$this-c scale 1.05"
	button $wid.scale.decr -text "-" -command "$this-c scale [expr 1.0/1.05]"
	button $wid.scale.decr2 -text "--" -command "$this-c scale [expr 1.0/1.25]"
	pack $wid.scale.l $wid.scale.incr $wid.scale.incr2 $wid.scale.decr $wid.scale.decr2 \
		-side left -anchor w

	frame $wid.mats -relief ridge -borderwidth 4

	frame $wid.mats.mat1 -relief groove -borderwidth 4
	set mat1 $wid.mats.mat1
	label $mat1.label -text "Default Materials"
	listbox $mat1.box -width 10 -height 3 -relief sunken -yscrollcommand "$mat1.boxs set"
	global defaultbox
	set defaultbox $mat1.box
	scrollbar $mat1.boxs -relief sunken -command "$defaultbox yview"
	pack $mat1.label -side top -padx 2 -pady 2 -fill y
	pack $defaultbox -side left -padx 2 -pady 2 -fill both -expand yes
	pack $mat1.boxs -side right -padx 2 -pady 2 -anchor e -fill y

	frame $wid.mats.mat2 -relief groove -borderwidth 4
	set mat2 $wid.mats.mat2
	label $mat2.label -text "Widget Materials"
	listbox $mat2.box -width 10 -height 3 -relief sunken -yscrollcommand "$mat2.boxs set"
	global materialbox
	set materialbox $mat2.box
	scrollbar $mat2.boxs -relief sunken -command "$materialbox yview"
	pack $mat2.label -side top -padx 2 -pady 2 -fill y
	pack $materialbox -side left -padx 2 -pady 2 -fill both -expand yes
	pack $mat2.boxs -side right -padx 2 -pady 2 -anchor e -fill y

	bind $defaultbox <Button-1> "$this bwdefselect %y $wid $defaultbox $materialbox"
	bind $materialbox <Button-1> "$this bwselect %y $wid $materialbox $defaultbox"

	pack $mat1 $mat2 -in $wid.mats -side left -padx 2 -pady 2 -fill both -expand yes
	pack $wid.opts -in $wid -side top -padx 2 -pady 2 -anchor w
	pack $wid.scale -in $wid -side top -padx 2 -pady 2 -anchor w -fill both -expand yes
	pack $wid.mats -in $wid -side top -padx 2 -pady 2 -fill both -expand yes

	$this update_materials
	
	set matwin $wid.matwin
	global $wid.matwin.mati
	global $wid.matwin.mattype
    }

    method update_materials {} {
	set mats [$this-c defmaterials]

	global defaultbox
	$defaultbox delete 0 end
	for {set i 0} {$i < [llength $mats]} {incr i 1} {
	    $defaultbox insert end [lindex $mats $i]
	}

	set mats [$this-c materials]

	global materialbox
	$materialbox delete 0 end
	for {set i 0} {$i < [llength $mats]} {incr i 1} {
	    $materialbox insert end [lindex $mats $i]
	}
    }

    method help {} {
# There's not actually any help extant right now, so make this work some day
# when there is.  Put it in $DataflowTCL/help maybe?
#	global DataflowTCL
#	helpPage $pse_root/help/$name.html
    }

    # This is meant to be a virtual link for inherited widget tcls...
    method scale_changed {newscale} {
    }
    
    method bwdefselect {y wid box obox} {
	set ind [$box nearest $y]
	$box selection set $ind
	$obox selection clear 0 end
	set mat [$box get $ind]

	global $matwin.mattype
	set $matwin.mattype setdefmat
	global $matwin.mati
	set $matwin.mati $ind

	$this-c getdefmat $ind

	if {[winfo exists $matwin]} {
	    meresync $matwin
	    raise $matwin
	} else {
	    toplevel $wid.matwin
	    makeMaterialEditor $wid.matwin $this-material "$this bwcommit $box $obox" "$this bwcancel $box $obox"
	}
    }

    method bwselect {y wid box obox} {
	set ind [$box nearest $y]
	$box selection set $ind
	$obox selection clear 0 end
	set mat [$box get $ind]
	
	global $matwin.mattype
	set $matwin.mattype setmat
	global $matwin.mati
	set $matwin.mati $ind

	$this-c getmat $ind

	if {[winfo exists $matwin]} {
	    meresync $matwin
	    raise $matwin
	} else {
	    toplevel $wid.matwin
	    makeMaterialEditor $wid.matwin $this-material "$this bwcommit $box $obox" "$this bwcancel $box $obox"
	}
    }

    method bwcommit {box obox} {
	global $matwin.mattype $matwin.mati
	$this-c [set $matwin.mattype] [set $matwin.mati]

	bwcancel $box $obox
    }
    method bwcancel {box obox} {
	$box selection clear 0 end
	$obox selection clear 0 end
	destroy $matwin
    }
}

proc initColor {c r g b} {
    global $c-r $c-g $c-b
    set $c-r $r
    set $c-g $g
    set $c-b $b
}

proc initMaterial {matter} {
#     initColor $matter-ambient 0.0 0.0 0.0
#     initColor $matter-diffuse 0.0 0.0 0.0
#     initColor $matter-specular 0.0 0.0 0.0
    initColor $matter-ambient 1.0 1.0 1.0
    initColor $matter-diffuse 1.0 1.0 1.0
    initColor $matter-specular 1.0 1.0 1.0
    global $matter-shininess
    set $matter-shininess 10.0
    initColor $matter-emission 0.0 0.0 0.0
    global $matter-reflectivity
    set $matter-reflectivity 0.5
    global $matter-transparency
    set $matter-transparency 0
    global $matter-refraction_index
    set $matter-refraction_index 1.0
}
