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


itcl_class SCIRun_FieldsCreate_SamplePlane {
    inherit Module

    constructor {config} {
        set name SamplePlane
    }

    method update-type { w } {
	global $this-update_type
	set $this-update_type [$w get]
    }

    method position_release { } {
	global $this-update_type

	set type [set $this-update_type]
	if { $type == "On Release" } {
	    $this-c needexecute
	}
    }

    method set_position {v} {
	global $this-update_type

	set type [set $this-update_type]
	if { $type == "Auto" } {
	    $this-c needexecute
	}
    }


    method edit_scale { } {
        set w .ui[modname]
        if { ![winfo exists $w] } {
            return
        }

        if { [set $this-auto_size] == 0 } {
            $w.row4.scale configure -from -1.0 -to 1.0 -resolution .01 \
                -variable $this-pos
            $w.row4.l2 configure -state disabled
            set fgc [$w.row4.l2 cget -disabledforeground]
            $w.row4.entry configure -state disabled -foreground $fgc
            
        } else {
            $w.row4.scale configure -from 0 -to [expr [set $this-sizez] - 1] \
                -resolution 1  -variable $this-z_value
            $w.row4.l2 configure -state normal
            set fgc [$w.row4.l2 cget -foreground]
            $w.row4.entry configure -state normal -foreground $fgc
        }
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w

	frame $w.row1
	frame $w.row2
	frame $w.row21
	frame $w.row3
	frame $w.row31
	frame $w.row32
	frame $w.row4
        frame $w.row5
	frame $w.which -relief groove -borderwidth 2

	pack $w.row1 $w.row2 $w.row21 $w.row3 $w.row31 $w.row32 \
            $w.row4 $w.row5 $w.which \
	    -side top -e y -f both -padx 5 -pady 5
	
	label $w.row1.xsize_label -text "Width    "
	entry $w.row1.xsize -textvariable $this-sizex
        radiobutton $w.row1.auto_size  -text "Manual size" -value 0 \
           -variable $this-auto_size -borderwidth 2 -command "$this edit_scale"
	label $w.row2.ysize_label -text "Height   "
	entry $w.row2.ysize -textvariable $this-sizey
        radiobutton $w.row2.auto_size  -text "Auto size" -value 1 \
           -variable $this-auto_size -borderwidth 2 -command "$this edit_scale"

	pack $w.row1.xsize_label $w.row1.xsize $w.row1.auto_size -side left
	pack $w.row2.ysize_label $w.row2.ysize $w.row2.auto_size -side left

	label $w.row21.zsize_label -text "Pad Percentage"
	entry $w.row21.zsize -textvariable $this-padpercent
	pack $w.row21.zsize_label $w.row21.zsize -side left

	label $w.row3.label -text "Axis: "
	radiobutton $w.row3.x -text "X    " -variable $this-axis -value 0 \
            -command "$this set_position [set $this-z_value]"
	radiobutton $w.row3.y -text "Y    " -variable $this-axis -value 1 \
            -command "$this set_position [set $this-z_value]"
	radiobutton $w.row3.z -text "Z    " -variable $this-axis -value 2 \
            -command "$this set_position [set $this-z_value]"
	radiobutton $w.row3.c -text "Custom" -variable $this-axis -value 3 \
            -command "$this set_position [set $this-z_value]"
	pack $w.row3.label $w.row3.x $w.row3.y $w.row3.z $w.row3.c -side left

        label $w.row31.lab -text "Custom Origin:   "
        entry $w.row31.ex -textvariable $this-corigin-x -width 8
        entry $w.row31.ey -textvariable $this-corigin-y -width 8
        entry $w.row31.ez -textvariable $this-corigin-z -width 8
        pack $w.row31.lab $w.row31.ex $w.row31.ey $w.row31.ez -side left
        label $w.row32.lab -text "Custom Normal: "
        entry $w.row32.ex -textvariable $this-cnormal-x -width 8
        entry $w.row32.ey -textvariable $this-cnormal-y -width 8
        entry $w.row32.ez -textvariable $this-cnormal-z -width 8
        pack $w.row32.lab $w.row32.ex $w.row32.ey $w.row32.ez -side left

	label $w.row4.l1 -text "Position: "
	scale $w.row4.scale -width 10 -orient horizontal \
            -command "$this set_position" 
        label $w.row4.l2 -text " index: "
        entry $w.row4.entry -textvariable $this-z_value  -width 12


	iwidgets::optionmenu $w.row5.update -labeltext "Update:" \
	    -labelpos w -command "$this update-type $w.row5.update"
	$w.row5.update insert end "On Release" Manual Auto
	$w.row5.update select [set $this-update_type]

	bind $w.row3.x <ButtonRelease> "$this position_release"
	bind $w.row3.y <ButtonRelease> "$this position_release"
	bind $w.row3.z <ButtonRelease> "$this position_release"
	bind $w.row3.c <ButtonRelease> "$this position_release"
	bind $w.row4.scale <ButtonRelease> "$this position_release"
        bind $w.row4.entry <Return> "$this position_release"
	
	pack $w.row4.l1 $w.row4.scale $w.row4.l2 $w.row4.entry -side left \
            -anchor s
        pack $w.row5.update -side left

	label $w.which.l -text "Data At Location"
	radiobutton $w.which.node -text "Nodes (linear basis)" \
		-variable $this-data-at -value Nodes
	radiobutton $w.which.face -text "Faces (constant basis)" \
		-variable $this-data-at -value Faces
	radiobutton $w.which.none -text "None" \
		-variable $this-data-at -value None
	pack $w.which.l -side top
	pack $w.which.node $w.which.face $w.which.none -anchor nw

	makeSciButtonPanel $w $w $this
	moveToCursor $w

        $this edit_scale

    }
}


