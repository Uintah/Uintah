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
#    File   : ICUMonitor.tcl
#    Author : Martin Cole
#    Date   : Fri Nov 19 10:42:03 2004

itcl_class VS_Render_ICUMonitor {
    inherit Module

    constructor {config} {
	set name ICUMonitor
	global  $this-time	 
	global  $this-sample_rate 
	global  $this-sweep_speed_ 
	global  $this-dots_per_inch
	global  $this-plot_height
	global  $this-play_mode
	global  $this-plot_count
	global  $this-edit
	global  $this-edit-target

	set $this-edit          0
	set $this-edit-target   0
	set $this-time          0.0
	set $this-sample_rate   200.0 
	set $this-sweep_speed   25.0
	set $this-dots_per_inch 96.7
	set $this-plot_height  75.0
	set $this-play_mode     1
	set $this-plot_count    0
    }

    method bind_events {w} {
	# every time the OpenGL widget is displayed, redraw it
	bind $w <Expose> "$this-c expose"
# 	bind $w <Shift-ButtonPress-1> "$this-c mouse push %x %y %b 0"
# 	bind $w <Shift-ButtonPress-2> "$this-c mouse push %x %y %b 0"
# 	bind $w <Shift-ButtonPress-3> "$this-c mouse push %x %y %b 1"
# 	bind $w <Shift-Button1-Motion> "$this-c mouse motion %x %y"
# 	bind $w <Shift-Button2-Motion> "$this-c mouse motion %x %y"
# 	bind $w <Shift-Button3-Motion> "$this-c mouse motion %x %y"
# 	bind $w <Shift-ButtonRelease-1> "$this-c mouse release %x %y %b"
# 	bind $w <Shift-ButtonRelease-2> "$this-c mouse release %x %y %b"
# 	bind $w <Shift-ButtonRelease-3> "$this-c mouse release %x %y %b"

# 	# controls for pan and zoom and reset
# 	bind $w <ButtonPress-1> "$this-c mouse translate start %x %y"
# 	bind $w <Button1-Motion> "$this-c mouse translate move %x %y"
# 	bind $w <ButtonRelease-1> "$this-c mouse translate end %x %y"
# 	bind $w <ButtonPress-2> "$this-c mouse reset %x %y 1"
# 	bind $w <Button2-Motion> "$this-c mouse reset %x %y 1"
# 	bind $w <ButtonRelease-2> "$this-c mouse reset %x %y 1"
# 	bind $w <ButtonPress-3> "$this-c mouse scale start %x %y"
# 	bind $w <Button3-Motion> "$this-c mouse scale move %x %y"
# 	bind $w <ButtonRelease-3> "$this-c mouse scale end %x %y"

	bind $w <Destroy> "$this-c closewindow"		
    }

    method edit_targ {} {
	global  $this-edit-target
	return [set $this-edit-target]
    }


    method edit_accept {win} {
	global  $this-edit
	global  $this-edit-target
	destroy $win
	set $this-edit 1
	add_plot edit_targ
    }

    method add_accept {win} {
	global $this-plot_count
	global  $this-edit
	if {[set $this-edit] != 1} {
	    incr $this-plot_count
	}
	set $this-edit 0
	$this-c "init"
	destroy $win
    }

    method add_cancel {win} {
	destroy $win
    }

    method del_plot {} {
	global $this-plot_count
	if {[set $this-plot_count] > 0} {
	    incr $this-plot_count -1
	}
	$this-c "init"
    }


    method edit_plot {} {
	set w .ui[modname]
	if {[winfo exists $w.edit]} {
	    SciRaise $w.edit
	    return
	} else {
	    global $this-plot_count

	    if {[set $this-plot_count] == 0} {
		return
	    }

	    toplevel $w.edit

	    set which 0

	    frame $w.edit.f -relief groove -borderwidth 2
	    label $w.edit.f.whichl -text "Which Plot \[0-n\]:" -relief groove
	    entry  $w.edit.f.which -textvariable $this-edit-target -width 8

	    pack $w.edit.f.whichl $w.edit.f.which -side left -fill x -padx 6
	    pack $w.edit.f
	    frame $w.edit.accept -borderwidth 2
	    button $w.edit.accept.accept -text "Accept" \
		-command "$this edit_accept $w.edit"

	    pack $w.edit.accept.accept -side left -fill x -padx 10
	    pack $w.edit.accept -side top -fill x -padx 2 -pady 2
	}
    }
    
    method setColor {col color colMsg} {
	 global $color
	 global $color-r
	 global $color-g
	 global $color-b
	 set ir [expr int([set $color-r] * 65535)]
	 set ig [expr int([set $color-g] * 65535)]
	 set ib [expr int([set $color-b] * 65535)]

	 set window .ui[modname]
	 $col config -background [format #%04x%04x%04x $ir $ig $ib]
	 $this-c $colMsg
    }

    method raiseColor {col color colMsg} {
	 global $color
	 set window .ui[modname]
	 if {[winfo exists $window.color]} {
	     SciRaise $window.color
	     return
	 } else {
	     # makeColorPicker now creates the $window.color toplevel.
	     makeColorPicker $window.color $color \
		     "$this setColor $col $color $colMsg" \
		     "destroy $window.color"
	 }
    }

    method addColorSelection {frame text color colMsg} {
	 #add node color picking 
	 global $color
	 global $color-r
	 global $color-g
	 global $color-b
	 if {! [info exists $color-r]} {
	     set $color-r .5
	 }
	 if {! [info exists $color-g]} {
	     set $color-g .5
	 }
	 if {! [info exists $color-b]} {
	     set $color-b .5
	 }

	 set ir [expr int([set $color-r] * 65535)]
	 set ig [expr int([set $color-g] * 65535)]
	 set ib [expr int([set $color-b] * 65535)]

	 frame $frame.colorFrame
	 frame $frame.colorFrame.col -relief ridge -borderwidth \
		 4 -height 0.8c -width 1.0c \
		 -background [format #%04x%04x%04x $ir $ig $ib]
	 
	 set cmmd "$this raiseColor $frame.colorFrame.col $color $colMsg"
	 button $frame.colorFrame.set_color \
		 -text $text -command $cmmd
	 
	 #pack the node color frame
	 pack $frame.colorFrame.set_color $frame.colorFrame.col -side left -padx 2
	 pack $frame.colorFrame -side left

    }

    method add_plot {fn} {
	set w .ui[modname]
	if {[winfo exists $w.add]} {
	    SciRaise $w.add
	    return
	} else {
	    global $this-plot_count
	    toplevel $w.add

	    set v [$this $fn]

	    frame $w.add.f -relief groove -borderwidth 2
	    label $w.add.f.nwl -text "NW Label:" -relief groove
	    entry $w.add.f.nw -textvariable $this-nw_label-$v \
		-width 8
	    label $w.add.f.swl -text "SW Label:" -relief groove
	    entry $w.add.f.sw -textvariable $this-sw_label-$v \
		-width 8
	    label $w.add.f.pl -text "Plot Label:" -relief groove
	    entry $w.add.f.p -textvariable $this-label-$v \
		-width 8
	    label $w.add.f.minrl -text "Min Label:" -relief groove
	    entry $w.add.f.minr \
		-textvariable $this-min_ref_label-$v -width 8
	    label $w.add.f.maxrl -text "Max Label:" -relief groove
	    entry $w.add.f.maxr \
		-textvariable $this-max_ref_label-$v -width 8
	    label $w.add.f.minl -text "Min Value:" -relief groove
	    entry $w.add.f.min -textvariable $this-min-$v \
		-width 8
	    label $w.add.f.maxl -text "Max Value:" -relief groove
	    entry $w.add.f.max -textvariable $this-max-$v \
		-width 8
	    label $w.add.f.idxl -text "Data Index:" -relief groove
	    entry $w.add.f.idx -textvariable $this-idx-$v \
		-width 8

	    frame $w.add.f.col -borderwidth 2
	    addColorSelection $w.add.f.col "Plot Color" $this-plot_color-$v  \
		"redraw"	   
 
	    pack  $w.add.f.pl $w.add.f.p $w.add.f.nwl $w.add.f.nw \
		$w.add.f.swl $w.add.f.sw $w.add.f.minrl $w.add.f.minr \
		$w.add.f.maxrl $w.add.f.maxr $w.add.f.minl $w.add.f.min \
		$w.add.f.maxl $w.add.f.max $w.add.f.idxl $w.add.f.idx \
		$w.add.f.col -side top -fill x -padx 2 -pady 2

	    pack $w.add.f -side top -fill x -padx 2 -pady 2
	    frame $w.add.accept -borderwidth 2
	    

	    button $w.add.accept.cancel -text "Cancel" \
		-command "$this add_cancel $w.add"
	    button $w.add.accept.accept -text "Accept" \
		-command "$this add_accept $w.add"
	    pack $w.add.accept.cancel $w.add.accept.accept -side left \
		-fill x -padx 10


	    pack $w.add.accept -side top -fill x -padx 2 -pady 2
	}
    }

    method cur_size {} {
	global $this-plot_count
	return [set $this-plot_count]
    }
    
    method create_gl {} {
        set w .ui[modname]
        if {[winfo exists $w.f.gl]} {
            raise $w
        } else {
            set n "$this-c needexecute"
	    
            frame $w.f.gl -relief groove -borderwidth 2
            pack $w.f.gl -padx 2 -pady 2
            # create an OpenGL widget
            opengl $w.f.gl.gl -geometry 640x640 -doublebuffer true \
		-direct true -rgba true -redsize 1 -greensize 1 \
		-bluesize 1 -depthsize 2
	    bind_events $w.f.gl.gl
            # place the widget on the screen
            pack $w.f.gl.gl -fill both -expand 1
	    checkbutton  $w.f.gl.play -text "Continuous Updates" -padx 6 \
		-justify center -relief flat -variable $this-play_mode \
		-onvalue 1 -offvalue 0 -anchor n

	    # time slider
	    scale $w.f.gl.time -variable $this-time \
                 -from 0.0 -to 1.0 -label "Time" \
                 -showvalue true -resolution 0.001 \
                 -orient horizontal -command "$this-c time"


	    frame $w.f.gl.plots -relief groove -borderwidth 2

	    button $w.f.gl.plots.add -text "Add Plot" \
		-command "$this add_plot cur_size"
	    button $w.f.gl.plots.edit -text "Edit Plot" \
		-command "$this edit_plot"
	    button $w.f.gl.plots.del -text "Delete Last" \
		-command "$this del_plot"

	    pack $w.f.gl.plots.add $w.f.gl.plots.edit $w.f.gl.plots.del \
		-side left -fill x -padx 6

	    pack $w.f.gl.play  $w.f.gl.time $w.f.gl.plots -side top \
		-fill x -padx 4

	    frame $w.f.settings -relief groove -borderwidth 2
	    label $w.f.settings.srl -text "Sample Rate:" -relief groove
	    entry $w.f.settings.sr  -textvariable $this-sample_rate -width 8
	    label $w.f.settings.ssl -text "Sweep Speed:" -relief groove
	    entry $w.f.settings.ss  -textvariable $this-sweep_speed -width 8
	    label $w.f.settings.dpil -text "Monitor dpi:" -relief groove
	    entry $w.f.settings.dpi -textvariable $this-dots_per_inch -width 8
	    label $w.f.settings.ghl -text "Plot Height:" -relief groove
	    entry $w.f.settings.gh  -textvariable $this-plot_height -width 8

	    pack $w.f.settings.srl $w.f.settings.sr $w.f.settings.ssl \
		$w.f.settings.ss $w.f.settings.dpil $w.f.settings.dpi \
		$w.f.settings.ghl $w.f.settings.gh \
		-side left -fill x -padx 2 -pady 2

	    pack $w.f.settings -padx 2 -pady 2 -fill x
 
        }
    }
	

    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    create_gl
	    return
	}
	toplevel $w
	frame $w.f
	pack $w.f -padx 2 -pady 2
        create_gl

        pack $w.f 
    }
}
    
