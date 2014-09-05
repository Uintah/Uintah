#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  License for the specific language governing rights and limitations under
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
#



catch {rename GLTextureBuilder ""}

itcl_class SCIRun_Visualization_GLTextureBuilder {
    inherit Module
    constructor {config} {
	set name GLTextureBuilder
	set_defaults
    }
    method set_defaults {} {
	global $this-max_brick_dim
	global $this-sel_brick_dim
	global $this-min
	global $this-max
	global $this-is_fixed
	set $this-max_brick_dim 0 
	set $this-sel_brick_dim 0
	set $this-min 0
	set $this-max 1
	set $this-is_fixed 0
    }
    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    return
	}
	toplevel $w
	frame $w.f 
	pack $w.f -padx 2 -pady 2 -fill x
	
	set n "$this-c needexecute"
	
	frame $w.f.dimframe -relief groove -border 2
	label $w.f.dimframe.l -text "Brick Size Cubed"
	pack $w.f.dimframe -side top -padx 2 -pady 2 -fill both
	pack $w.f.dimframe.l -side top -fill x

	if { [set $this-max_brick_dim] != 0 } {
	    $this SetDims [set $this-max_brick_dim]
	}

	global $this-is_fixed
        frame $w.f1 -relief flat
        pack $w.f1 -side top -expand yes -fill x
        radiobutton $w.f1.b -text "Auto Scale"  -variable $this-is_fixed \
		-value 0 -command "$this autoScale"
        pack $w.f1.b -side left

        frame $w.f2 -relief flat
        pack $w.f2 -side top -expand yes -fill x
        radiobutton $w.f2.b -text "Fixed Scale"  -variable $this-is_fixed \
		-value 1 -command "$this fixedScale"
        pack $w.f2.b -side left

        frame $w.f3 -relief flat
        pack $w.f3 -side top -expand yes -fill x
        
        label $w.f3.l1 -text "min:  "
        entry $w.f3.e1 -textvariable $this-min

        label $w.f3.l2 -text "max:  "
        entry $w.f3.e2 -textvariable $this-max
        pack $w.f3.l1 $w.f3.e1 $w.f3.l2 $w.f3.e2 -side left \
            -expand yes -fill x -padx 2 -pady 2

        bind $w.f3.e1 <Return> $n
        bind $w.f3.e2 <Return> $n

	if { [set $this-is_fixed] } {
            $w.f2.b select
            $this fixedScale
        } else {
            $w.f1.b select
            $this autoScale
        }

	makeSciButtonPanel $w $w $this
	moveToCursor $w
    }

    method autoScale { } {
        global $this-is_fixed
        set w .ui[modname]
        
        set $this-is_fixed 0

        set color "#505050"

        $w.f3.l1 configure -foreground $color
        $w.f3.e1 configure -state disabled -foreground $color
        $w.f3.l2 configure -foreground $color
        $w.f3.e2 configure -state disabled -foreground $color


#        $this-c needexecute
    }

    method fixedScale { } {
        global $this-is_fixed
        set w .ui[modname]

        set $this-is_fixed 1


        $w.f3.l1 configure -foreground black
        $w.f3.e1 configure -state normal -foreground black
        $w.f3.l2 configure -foreground black
        $w.f3.e2 configure -state normal -foreground black
        
    }

    method SetDims { val } {
	global $this-max_brick_dim
	global $this-sel_brick_dim
	set $this-max_brick_dim $val
	if {[set $this-sel_brick_dim] == 0} {set $this-sel_brick_dim $val}

	set w .ui[modname]

	set vals  [format "%i %i %i %i" [expr $val/8] [expr $val/4] [expr $val/2] $val] 
	set vals [split $vals]
	if {![winfo exists $w]} {
	    return
	}
	if {[winfo exists $w.f.dimframe.f]} {
	    destroy $w.f.dimframe.f
	}

	frame $w.f.dimframe.f -relief flat
	pack $w.f.dimframe.f -side top -fill x
	set f $w.f.dimframe.f
	set still_exists 0
	for {set i 0} {$i < 4} { incr i} {
	    set v [lindex $vals $i]
	    if {$v == [set $this-sel_brick_dim]} {set still_exists 1}
	    radiobutton $f.brickdim$v -text $v -relief flat \
		-variable $this-sel_brick_dim -value $v \
		-command "$this-c needexecute"
	    pack $f.brickdim$v -side left -padx 2 -fill x
	}
	if {$still_exists == 0} {set $this-sel_brick_dim $val}
    }
}
