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

itcl_class SCIRun_Visualization_SampleColorMap { 
    inherit Module 

    protected  bVar
    protected exposed
    protected cmap
    constructor {config} { 
        set name SampleColorMap 
        set_defaults 
    } 
  
    method set_defaults {} { 
	global $this-min
	global $this-max
	global $this-value
	set exposed 0
	set $this-isFixed 0
	set $this-min 0
	set $this-max 1
	set $this-value 0.5
	set cmap ""
    }   

    method ui {} { 
	global $this-isFixed
	global $this-min
	global $this-max

	set w .ui[modname]
	
	if {[winfo exists $w]} { 
	    wm deiconify $w
	    raise $w 
	    return; 
	} 
	
	toplevel $w 
	wm minsize $w 200 50 
 
	frame $w.f3 -relief flat
	pack $w.f3 -side top -expand yes -fill x -padx 2 -pady 2
	
	label $w.f3.l1 -text "min:  "
	label $w.f3.e1 -textvariable $this-min 
	pack $w.f3.l1 $w.f3.e1 -side left -anchor w

	label $w.f3.l2 -text "min:  "
	label $w.f3.e2 -textvariable $this-max
	pack $w.f3.e2 $w.f3.l2 -side right
 
	frame $w.f4 -relief sunken -height 20  -borderwidth 2 
	pack $w.f4 -side top -padx 2 -pady 2 -expand yes -fill x
	canvas $w.f4.canvas -bg "#ffffff" -height 20 
	pack $w.f4.canvas -anchor w -expand yes -fill x

	frame $w.f5 -relief flat
	pack $w.f5 -side top -padx 2 -pady 2 -expand yes -fill x

	label $w.f5.l -text "value:  "
	entry $w.f5.e -textvariable $this-value
	pack $w.f5.l -side left
	pack $w.f5.e -side left -expand yes -fill x -padx 2 -pady 2


	button $w.close -text Close -command "destroy $w"
	pack $w.close -side bottom -expand yes -fill x


	bind $w.f4.canvas <Expose> "$this canvasExpose"
	bind $w.f4.canvas <Button-1> "$this drawLine %x"
	bind $w.f4.canvas <B1-Motion> "$this drawLine %x"
	bind $w.f4.canvas <ButtonRelease> "$this setvalue %x; $this-c needexecute"

	bind $w.f5.e <Return> "$this moveLine"
    }


    method moveLine { } {
	set x [$this setx]
	drawLine [expr int($x)]
    }

    method drawLine { loc } {
	set w .ui[modname]
	set canvas $w.f4.canvas
	$canvas delete line
	set cw [winfo width $canvas]
	set ch [winfo height $canvas]
	if { $loc < 0 } { 
	    set x 0 
	} elseif { $loc > $cw } { 
	    set x [expr $cw-1] 
	} else { 
	    set x $loc 
	}
	set xm [expr $x - 1]
	set xp [expr $x + 1]
	$canvas create line $xm 0 $xm $ch -tags line -fill yellow
	$canvas create line $x 0 $x $ch -tags line -fill red
	$canvas create line $xp 0 $xp $ch -tags line -fill yellow
	$canvas raise line
    }

    method canvasExpose {} {
	set w .ui[modname]
	
	if { [winfo viewable $w.f4.canvas] } { 
	    if { $exposed } {
		return
	    } else {
		set exposed 1
		set x [$this setx]
		$this drawLine $x
		$this redraw
	    } 
	} else {
	    return
	}
    }

    method setColorMap { args } {
	set cmap [split $args]
    }

    method setx {} {
	set w .ui[modname]
	set canvas $w.f4.canvas
	set cw [winfo width $canvas]
	set val [expr $cw * (([set $this-value]-[set $this-min]) / double([set $this-max] - [set $this-min]))]
	return $val
    }
	
    method setvalue { val } {
	global $this-min
	global $this-max
	set w .ui[modname]
	set canvas $w.f4.canvas
	set cw [winfo width $canvas]
	if { $val > $cw } {
	    set x [expr $cw]
	} elseif { $val < 0 } {
	    set x 0
	} else {
	    set x $val
	}
	set $this-value [expr ([set $this-max] - \
			       [set $this-min]) *(double($x)/$cw) + \
			 [set $this-min]]
    }

    method redraw {} {
	global $this-width
	global $this-height
	set w .ui[modname]

	if {![winfo exist $w]} {
	    return
	}

	set n [expr [llength $cmap]/3]
	set canvas $w.f4.canvas
	$canvas delete map
	set cw [winfo width $canvas]
	set $this-width $cw
	set ch [winfo height $canvas]
	set $this-height $ch
	set dx [expr $cw / double($n)] 
	set x 0
	for {set i 0} {$i < $n} {incr i 1} {
	    set j [expr 3 * $i]
	    set r [expr int([lindex $cmap $j]*255)]
	    set g [expr int([lindex $cmap [expr $j + 1]]*255)]
	    set b [expr int([lindex $cmap [expr $j + 2]]*255)]
	    set c [format "#%02x%02x%02x" $r $g $b]
	    set oldx $x
	    set x [expr ($i+1)*$dx]
	    $canvas create rectangle \
		    $oldx 0 $x $ch -fill $c -outline $c -tags map
	}
	set taglist [$canvas gettags all]
	set i [lsearch $taglist line]
	if { $i != -1 } {
	    $canvas lower map line
	}
    }
}
