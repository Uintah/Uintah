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

catch {rename Nrrd_Segment_SlicePicker ""}

itcl_class Nrrd_Segment_SlicePicker {
    inherit Module

    constructor {config} {
	set name SlicePicker
	set_defaults
    }

    method set_defaults {} {
	global $this-bias
	global $this-scale
	global $this-type
	global $this-tx
	global $this-ty
	global $this-tz
	set $this-bias 0
	set $this-scale 1
	set $this-tissue 1
	set $this-tx .5
	set $this-ty .5
	set $this-tz .5
	$this-c needexecute
    }
    
    method raiseGL {} {
	set w .ui[modname]
	if {[winfo exists $w.gl]} {
	    raise $w.gl
	} else {
	    toplevel $w.gl
	    wm geometry $w.gl =600x600+300-200
	    wm minsize $w.gl 600 600
	    wm maxsize $w.gl 600 600
	    opengl $w.gl.gl -geometry 600x600 -doublebuffer true -direct true\
		 -rgba true -redsize 1 -greensize 1 -bluesize 1 -depthsize 2
	    bind $w.gl.gl <Expose> "$this-c redraw_all"
	    bind $w.gl.gl <ButtonPress-1> "$this press %x %y"
	    bind $w.gl.gl <Button1-Motion> "$this motion %x %y"
	    bind $w.gl.gl <ButtonRelease-1> "$this release"
	    bind $w.gl.gl <ButtonPress-2> "$this addPoint %x %y"
	    pack $w.gl.gl -fill both -expand 1
	}
    }
	
    method addPoint {wx wy} {
	set w [findWin $wx $wy]
	if {$w=="sag"} {
	    $this-c addPoint $w [expr ($wx-34)/255.0] [expr ($wy-34)/255.0]
	}
	if {$w=="cor"} {
	    $this-c addPoint $w [expr ($wx-310)/255.0] [expr ($wy-34)/255.0]
	} 
	if {$w=="axi"} {
	    $this-c addPoint $w [expr ($wx-310)/255.0] [expr (565-$wy)/255.0]
	}
    }
	
    method findWin {wx wy} {
	if {$wx < 34} {
	    if {$wy < 300} {return "sagzl"}
	    return "null"
	}
	if {$wx < 290} {
	    if {$wy < 34} {return "sagyt"}
	    if {$wy < 290} {return "sag"}
	    if {$wy < 300} {return "sagyb"}
	    return "null"
	}
	if {$wx < 300} {
	    if {$wy < 300} {return "sagzr"}
	    return "null"
	}
	if {$wx < 310} {
	    if {$wy < 300} {return "corzl"}
	    return "axiyl"
	}
	if {$wx < 566} {
	    if {$wy < 34} {return "corxt"}
	    if {$wy < 290} {return "cor"}
	    if {$wy < 300} {return "corxb"}
	    if {$wy < 310} {return "axixt"}
	    if {$wy < 566} {return "axi"}
	    return "axixb"
	}
	if {$wy < 300} {return "corzr"}
	return "axiyr"
    }

    method moveCross {wx wy} {
	global $this-tx
	global $this-ty
	global $this-tz
	global $this-win
	set owin [set $this-win]
#	puts -nonewline "owin = "
#	puts [set $this-win]
	set nwin [findWin $wx $wy]
	if {$owin=="sagzl"||$owin=="sagzr"||$owin=="corzl"||$owin=="corzr"} {
	    set $this-tz [expr ($wy-34)/255.0]
	    if {[set $this-tz] < 0} {set $this-tz 0}
	    if {[set $this-tz] > 1} {set $this-tz 1}
	    return
	}
	if {$owin == "sagyb" || $owin == "sagyt"} {
	    set $this-ty [expr ($wx-34)/255.0]
	    if {[set $this-ty] < 0} {set $this-ty 0}
	    if {[set $this-ty] > 1} {set $this-ty 1}
	    return
	}
	if {$owin =="corxt"||$owin=="corxb"||$owin=="axixt"||$owin=="axixb"} {
	    set $this-tx [expr ($wx-310)/255.0]
	    if {[set $this-tx] < 0} {set $this-tx 0}
	    if {[set $this-tx] > 1} {set $this-tx 1}
	    return
	}
	if {$owin == "axiyl" || $owin == "axiyr"} {
	    set $this-ty [expr (565-$wy)/255.0]
	    if {[set $this-ty] < 0} {set $this-ty 0}
	    if {[set $this-ty] > 1} {set $this-ty 1}
	    return
	}
	if {$owin=="sag"} {
	    set $this-ty [expr ($wx-34)/255.0]
	    set $this-tz [expr ($wy-34)/255.0]
	}
	if {$owin=="cor"} {
	    set $this-tx [expr ($wx-310)/255.0]
	    set $this-tz [expr ($wy-34)/255.0]
	}
	if {$owin=="axi"} {
	    set $this-tx [expr ($wx-310)/255.0]
	    set $this-ty [expr (565-$wy)/255.0]
	}
	return
    }

    method press {wx wy} {
	global $this-win
	set $this-win [findWin $wx $wy]
	moveCross $wx $wy
	$this-c redraw_lines
    }

    method motion {wx wy} {
	moveCross $wx $wy
	$this-c redraw_lines
    }

    method release { } {
	$this-c redraw_all
    }

    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w		    
	    return;
	}
	toplevel $w
	wm geometry $w =290x265+300-200
	wm minsize $w 290 265
#	wm maxsize $w 290 265
	frame $w.f
	frame $w.f.sc -relief sunken -bd 1
	global $this-bias
	global $this-scale
	global $this-tissue
	frame $w.f.sc.b
	label $w.f.sc.b.l -text "Bias: "
	scale $w.f.sc.b.s -variable $this-bias -command "$this-c \
		redraw_all" -from -1.0 -to 1.0 -orient horizontal \
		-tickinterval 0.5 -resolution 0.01
	pack $w.f.sc.b.l -side left
	pack $w.f.sc.b.s -side left -expand 1 -fill x
	frame $w.f.sc.s
	label $w.f.sc.s.l -text "Scale: "
	scale $w.f.sc.s.s -variable $this-scale -command "$this-c \
		redraw_all" -from 0.0 -to 4.0 -orient horizontal \
		-tickinterval 1.0 -resolution 0.01
	pack $w.f.sc.s.l -side left
	pack $w.f.sc.s.s -side left -expand 1 -fill x
	pack $w.f.sc.b $w.f.sc.s -side top -expand 1 -fill x
	frame $w.f.ra
	frame $w.f.ra.l -relief sunken -bd 1 
	label $w.f.ra.l.l -text "Tissue: "
	make_labeled_radio $w.f.ra.l.t "" "" \
                top $this-tissue \
                {{"Skin" 1} \
                {"Bone" 2} \
		{"CSF" 3} \
		{"Grey" 4} \
		{"White" 5} \
		{"Tumor" 6}}
	pack $w.f.ra.l.l $w.f.ra.l.t -side left -fill x -expand 1
	frame $w.f.ra.r -relief sunken -bd 1 
	label $w.f.ra.r.l -text "Move Views:"
	frame $w.f.ra.r.x
	button $w.f.ra.r.x.mm -text " -5 " -command "$this-c xmm"
	button $w.f.ra.r.x.m -text " -1 " -command "$this-c xm"
	label $w.f.ra.r.x.l -text " X "
	button $w.f.ra.r.x.p -text " +1 " -command "$this-c xp"
	button $w.f.ra.r.x.pp -text " +5 " -command "$this-c xpp"
	pack $w.f.ra.r.x.mm $w.f.ra.r.x.m $w.f.ra.r.x.l -side left
	pack $w.f.ra.r.x.pp $w.f.ra.r.x.p -side right
	frame $w.f.ra.r.y
	button $w.f.ra.r.y.mm -text " -5 " -command "$this-c ymm"
	button $w.f.ra.r.y.m -text " -1 " -command "$this-c ym"
	label $w.f.ra.r.y.l -text " Y "
	button $w.f.ra.r.y.p -text " +1 " -command "$this-c yp"
	button $w.f.ra.r.y.pp -text " +5 " -command "$this-c ypp"
	pack $w.f.ra.r.y.mm $w.f.ra.r.y.m $w.f.ra.r.y.l -side left
	pack $w.f.ra.r.y.pp $w.f.ra.r.y.p -side right
	frame $w.f.ra.r.z
	button $w.f.ra.r.z.mm -text " -5 " -command "$this-c zmm"
	button $w.f.ra.r.z.m -text " -1 " -command "$this-c zm"
	label $w.f.ra.r.z.l -text " Z "
	button $w.f.ra.r.z.p -text " +1 " -command "$this-c zp"
	button $w.f.ra.r.z.pp -text " +5 " -command "$this-c zpp"
	pack $w.f.ra.r.z.mm $w.f.ra.r.z.m $w.f.ra.r.z.l -side left
	pack $w.f.ra.r.z.pp $w.f.ra.r.z.p -side right
	pack $w.f.ra.r.l $w.f.ra.r.x $w.f.ra.r.y $w.f.ra.r.y $w.f.ra.r.z \
		-side top -fill y -expand 1
	pack $w.f.ra.l -side left -fill x -expand 1
	pack $w.f.ra.r -side right -fill both -expand 1
	pack $w.f.ra -side left -expand 1 -fill both
	frame $w.f.bu -relief sunken -bd 1
	button $w.f.bu.s -text "Send" -command "$this-c send"
	button $w.f.bu.c -text "Clear All" -command "$this-c clear"
	pack $w.f.bu.s $w.f.bu.c -side left -expand 1
	pack $w.f.sc $w.f.ra $w.f.bu -expand 1 -fill x -side top
	pack $w.f -expand 1 -fill x -side bottom
	raiseGL
    }
}
