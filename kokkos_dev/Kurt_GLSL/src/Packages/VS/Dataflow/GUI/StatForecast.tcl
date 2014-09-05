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
#    File   : StatForecast.tcl
#    Author : Martin Cole, Alex Ade
#    Date   : Fri Nov 19 10:42:03 2004

itcl_class VS_Render_StatForecast {
    inherit Module

    constructor {config} {
	set name StatForecast
	global  $this-time	 
	global  $this-sample_rate 
	global  $this-sweep_speed_ 
	global  $this-dots_per_inch
	global  $this-plot_height
	global  $this-play_mode
	global  $this-time_markers_mode
	global  $this-plot_count
	global  $this-edit
	global  $this-edit-target
	global  $this-selected_marker
	global  $this-font_scale
	global  $this-show_name
	global  $this-show_alarm
	global  $this-injury_offset
	global  $this-threshold_squ
	global  $this-threshold_sod
	global  $this-threshold_dqu
	global  $this-threshold_ttd
	global  $this-geom

	set $this-edit          0
	set $this-edit-target   0
	set $this-time          0.0
	set $this-sample_rate   200.0 
	set $this-sweep_speed   1.0
	set $this-dots_per_inch 96.7
	set $this-plot_height  75.0
	set $this-play_mode     1
	set $this-time_markers_mode  1
	set $this-plot_count    0
	set $this-selected_marker -1
	set $this-font_scale 1.0
	set $this-show_name 0
	set $this-show_alarm 1
	set $this-injury_offset 0
	set $this-threshold_squ 40
	set $this-threshold_sod 50
	set $this-threshold_dqu 60
	set $this-threshold_ttd 30
	set $this-geom "640x640+0+0"
    }

	method do_expose {} {
	 	if {[winfo exists .ui[modname]] != 0} {
			set w .ui[modname]
			set $this-geom [wm geometry $w]
		}

		$this-c expose
	}

    method bind_events {w} {
	# every time the OpenGL widget is displayed, redraw it
#	bind $w <Expose> "$this-c expose"
	bind $w <Expose> "$this do_expose"
# 	bind $w <Shift-ButtonPress-1> "$this-c mouse push %x %y %b 0"
# 	bind $w <Shift-ButtonPress-2> "$this-c mouse push %x %y %b 0"
# 	bind $w <Shift-ButtonPress-3> "$this-c mouse push %x %y %b 1"
# 	bind $w <Shift-Button1-Motion> "$this-c mouse motion %x %y"
# 	bind $w <Shift-Button2-Motion> "$this-c mouse motion %x %y"
# 	bind $w <Shift-Button3-Motion> "$this-c mouse motion %x %y"
# 	bind $w <Shift-ButtonRelease-1> "$this-c mouse release %x %y %b"
# 	bind $w <Shift-ButtonRelease-2> "$this-c mouse release %x %y %b"
# 	bind $w <Shift-ButtonRelease-3> "$this-c mouse release %x %y %b"

 	# controls for pan and zoom and reset
 	bind $w <ButtonPress-1> "$this-c mouse translate start %x %y"
 	bind $w <Button1-Motion> "$this-c mouse translate move %x %y"
 	bind $w <ButtonRelease-1> "$this-c mouse translate end %x %y"
 	bind $w <ButtonPress-2> "$this-c mouse reset %x %y 1"
 	bind $w <Button2-Motion> "$this-c mouse reset %x %y 1"
 	bind $w <ButtonRelease-2> "$this-c mouse reset %x %y 1"
 	bind $w <ButtonPress-3> "$this-c mouse scale start %x %y"
 	bind $w <Button3-Motion> "$this-c mouse scale move %x %y"
 	bind $w <ButtonRelease-3> "$this-c mouse scale end %x %y"

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

    method prefs_accept {win} {
        destroy $win
        $this-c "init"
    }

	method centerWindow {w1 w2} {
		update
		#wm overrideredirect $w1 1
		wm geometry $w1 ""
		update idletasks
		set w [winfo width $w2]
		set h [winfo height $w2]
		
		if {$w < 2} { set w [winfo screenwidth .] }
		if {$h < 2} { set h [winfo screenheight .] }

		set x [expr [winfo x $w2]+($w - [winfo width $w1])/2]
		set y [expr [winfo y $w2]+($h - [winfo height $w1])/2]
		wm geometry $w1 +${x}+${y}
		
		if { [winfo ismapped $w1] } {
			#raise $w1
			SciRaise $w1
		} else {
			wm deiconify $w1
		}
		
		grab $w1
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

    method set_prefs {} {
        set w .ui[modname]
        if {[winfo exists $w.prefs]} {
            SciRaise $w.prefs
            return
        } else {
            toplevel $w.prefs
            wm title $w.prefs "Preferences"
            wm withdraw $w.prefs
            wm resizable $w.prefs 0 0
                                                                                
            iwidgets::labeledframe $w.prefs.gen -labeltext "General" \
                -labelpos nw
            set gen [$w.prefs.gen childsite]
                                                                                
            checkbutton $gen.showname -text "Show Name" -padx 6 \
                -justify left -relief flat -variable $this-show_name \
                -onvalue 1 -offvalue 0 -anchor w
            pack $gen.showname -side top -fill x -pady 2
                                                                                
            checkbutton $gen.showalarm -text "Show Alarm" -padx 6 \
                -justify left -relief flat -variable $this-show_alarm \
                -onvalue 1 -offvalue 0 -anchor w
            pack $gen.showalarm -side top -fill x -pady 2
                                                                                
            frame $gen.fs -borderwidth 4
            pack $gen.fs -side top -fill x -pady 2
            scale $gen.fs.font -variable $this-font_scale \
                 -from 0.0 -to 3.0 -label "Font Scale:" \
                 -showvalue true -resolution 0.1 \
                 -orient horizontal -command "$this-c init"
            pack $gen.fs.font -fill x
                                                                                
            pack $w.prefs.gen -fill x -expand yes -side top

            iwidgets::labeledframe $w.prefs.time -labeltext "Timeline" \
                -labelpos nw
            set time [$w.prefs.time childsite]

            frame $time.f -borderwidth 2
            pack $time.f -side top -fill both

            frame $time.f.offset
            pack $time.f.offset -side top -fill x -pady 2
            label $time.f.offset.l -text "Index Offset:"
            entry $time.f.offset.val -textvariable $this-injury_offset \
                -width 6
            label $time.f.offset.u -text "s"
            #pack $time.f.offset.l -side left
            #pack $time.f.offset.val -side left
            #pack $time.f.offset.u -side left
            pack $time.f.offset.u -side right
            pack $time.f.offset.val -side right
            pack $time.f.offset.l -side right
                                                                                
            pack $w.prefs.time -fill x -expand yes -side top

            iwidgets::labeledframe $w.prefs.plot -labeltext "Thresholds" \
                -labelpos nw
            set plot [$w.prefs.plot childsite]

            frame $plot.f -borderwidth 2
            pack $plot.f -side top -fill both

            frame $plot.f.squ
            pack $plot.f.squ -side top -fill x -pady 2
            label $plot.f.squ.l -text "Survival Probability Threshold:"
            entry $plot.f.squ.val -textvariable $this-threshold_squ \
                -width 4
            label $plot.f.squ.u -text "%"
            pack $plot.f.squ.u -side right
            pack $plot.f.squ.val -side right
            pack $plot.f.squ.l -side right
                                                                                
            frame $plot.f.sod
            pack $plot.f.sod -side top -fill x -pady 2
            label $plot.f.sod.l -text "Survival/Death Threshold:"
            entry $plot.f.sod.val -textvariable $this-threshold_sod \
                -width 4
            label $plot.f.sod.u -text "%"
            pack $plot.f.sod.u -side right
            pack $plot.f.sod.val -side right
            pack $plot.f.sod.l -side right
                                                                                
            frame $plot.f.dqu
            pack $plot.f.dqu -side top -fill x -pady 2
            label $plot.f.dqu.l -text "Death Probability Threshold:"
            entry $plot.f.dqu.val -textvariable $this-threshold_dqu \
                -width 4
            label $plot.f.dqu.u -text "%"
            pack $plot.f.dqu.u -side right
            pack $plot.f.dqu.val -side right
            pack $plot.f.dqu.l -side right

				frame $plot.f.spacer
            pack $plot.f.spacer -side top -fill x -pady 4

            frame $plot.f.ttd
            pack $plot.f.ttd -side top -fill x -pady 2
            label $plot.f.ttd.l -text "Time To Death Threshold:"
            entry $plot.f.ttd.val -textvariable $this-threshold_ttd \
                -width 4
            label $plot.f.ttd.u -text "m"
            pack $plot.f.ttd.u -side right
            pack $plot.f.ttd.val -side right
            pack $plot.f.ttd.l -side right

            pack $w.prefs.plot -fill x -expand yes -side top
                                                                                
            frame $w.prefs.donef
            button $w.prefs.donef.done -text "Done" \
                -command "$this prefs_accept $w.prefs"
            pack $w.prefs.donef.done -side top -fill x
            pack $w.prefs.donef -side top -fill x -expand yes -pady 4 -padx 8
                                                                                
            centerWindow $w.prefs $w
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
	    checkbutton  $w.add.f.lines -text "Draw Min/Max Lines" -padx 6 \
		-justify left -relief flat -variable $this-lines-$v \
		-onvalue 1 -offvalue 0 -anchor n
	    label $w.add.f.minl -text "Min Value:" -relief groove
	    entry $w.add.f.min -textvariable $this-min-$v \
		-width 8
	    label $w.add.f.maxl -text "Max Value:" -relief groove
	    entry $w.add.f.max -textvariable $this-max-$v \
		-width 8
	    label $w.add.f.idxl -text "Data Index:" -relief groove
	    entry $w.add.f.idx -textvariable $this-idx-$v \
		-width 8
	    label $w.add.f.adl -text "Derived Data Label:" -relief groove
	    entry $w.add.f.ad -textvariable $this-aux_data_label-$v \
		-width 8
	    label $w.add.f.auxidxl -text "Derived Data Index:" -relief groove
	    entry $w.add.f.auxidx -textvariable $this-auxidx-$v \
		-width 8
	    checkbutton  $w.add.f.trace -text "Compare Traces" -padx 6 \
		-justify left -relief flat -variable $this-snd-$v \
		-onvalue 1 -offvalue 0 -anchor n
	    checkbutton  $w.add.f.draw_aux_data -text "Draw Derived Data" \
		-padx 6 -justify left -relief flat -variable \
		$this-draw_aux_data-$v -onvalue 1 -offvalue 0 -anchor n

	    frame $w.add.f.col -borderwidth 2
	    addColorSelection $w.add.f.col "Plot Color" $this-plot_color-$v  \
		"redraw"	   
 
	    pack  $w.add.f.pl $w.add.f.p $w.add.f.nwl $w.add.f.nw \
		$w.add.f.swl $w.add.f.sw $w.add.f.minrl $w.add.f.minr \
		$w.add.f.maxrl $w.add.f.maxr $w.add.f.lines $w.add.f.minl \
		$w.add.f.min $w.add.f.maxl $w.add.f.max $w.add.f.idxl \
		$w.add.f.idx $w.add.f.trace $w.add.f.adl $w.add.f.ad \
		$w.add.f.auxidxl $w.add.f.auxidx $w.add.f.draw_aux_data \
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
            SciRaise $w
        } else {
            set n "$this-c needexecute"

 	    bind $w <KeyPress-Up> "$this-c increment"
 	    bind $w <KeyPress-Down> "$this-c decrement"

            frame $w.f.menu -relief raised -borderwidth 2
            pack $w.f.menu -fill x -padx 2 -pady 2
            menubutton $w.f.menu.edit -text "Edit" -underline 0 \
                -menu $w.f.menu.edit.menu -state normal
            menu $w.f.menu.edit.menu -tearoff 0
            $w.f.menu.edit.menu add command -label "Preferences..." \
                -command "$this set_prefs"
            pack $w.f.menu.edit -side left
            menubutton $w.f.menu.mkrs -text "Markers" -underline 0 \
		-menu $w.f.menu.mkrs.menu -state disabled
            menu $w.f.menu.mkrs.menu -tearoff 0
            pack $w.f.menu.mkrs -side left

            frame $w.f.gl -relief groove -borderwidth 2
            pack $w.f.gl -padx 2 -pady 2 -fill both -expand 1

            # create an OpenGL widget
            opengl $w.f.gl.gl -doublebuffer true \
		-direct true -rgba true -redsize 1 -greensize 1 \
		-bluesize 1 -depthsize 2
	    bind_events $w.f.gl.gl
            # place the widget on the screen
            pack $w.f.gl.gl -fill both -expand 1
	    bind $w.f.gl.gl <Configure> "$this-c configure"

	    # time slider
	    scale $w.f.gl.time -variable $this-time \
                 -from 0.0 -to 1.0 \
                 -showvalue false -resolution 0.001 \
                 -orient horizontal -command "$this-c time"

	    frame $w.f.gl.timew -borderwidth 0
	    label $w.f.gl.timew.timel -text "Time: 00:00:00"
	    pack $w.f.gl.timew.timel -side left -padx 2 
	    pack $w.f.gl.time $w.f.gl.timew -side top \
		-fill x -padx 2

	    frame $w.f.plots -relief groove -borderwidth 2
	    pack $w.f.plots -side top -padx 2 -pady 2 -fill x

	    button $w.f.plots.add -text "Add Plot" \
		-command "$this add_plot cur_size"
	    button $w.f.plots.edit -text "Edit Plot" \
		-command "$this edit_plot"
	    button $w.f.plots.del -text "Delete Last" \
		-command "$this del_plot"

	    checkbutton  $w.f.plots.play -text "Play" -padx 6 \
		-justify center -relief flat -variable $this-play_mode \
		-onvalue 1 -offvalue 0 -anchor n
	    checkbutton  $w.f.plots.secs -text "Ticks" -padx 6 \
		-justify center -relief flat -variable $this-time_markers_mode \
		-onvalue 1 -offvalue 0 -anchor n

	    #pack $w.f.plots.add $w.f.plots.edit $w.f.plots.del \
	    	$w.f.plots.play $w.f.plots.secs \
		-side left -padx 2 -pady 2
	    pack $w.f.plots.play \
		-side left -padx 2 -pady 2

	    frame $w.f.settings -relief groove -borderwidth 2
            pack $w.f.settings -padx 2 -pady 2 -fill x

	    label $w.f.settings.srl -text "Sample Rate:" 
	    entry $w.f.settings.sr  -textvariable $this-sample_rate -width 8
	    label $w.f.settings.ssl -text "Speed:"
	    entry $w.f.settings.ss  -textvariable $this-sweep_speed -width 8
	    label $w.f.settings.dpil -text "Monitor dpi:"
	    entry $w.f.settings.dpi -textvariable $this-dots_per_inch -width 8
	    label $w.f.settings.ghl -text "Plot Height:"
	    entry $w.f.settings.gh  -textvariable $this-plot_height -width 8

	    #pack $w.f.settings.srl $w.f.settings.sr $w.f.settings.ssl \
		$w.f.settings.ss $w.f.settings.dpil $w.f.settings.dpi \
		$w.f.settings.ghl $w.f.settings.gh \
		-side left -padx 4 -pady 2
	    pack $w.f.settings.srl $w.f.settings.sr $w.f.settings.ssl \
		$w.f.settings.ss \
		-side left -padx 4 -pady 2
        }
    }

    method setTimeLabel {value} {
        set w .ui[modname]

	$w.f.gl.timew.timel configure -text $value
    }

    method selectMarker {} {
        set w .ui[modname]

	set $this-selected_marker [$w.f.menu.mkrs.menu index active]
        $this-c marker
    }

    method clearMarkers {} {
        set w .ui[modname]

        $w.f.menu.mkrs.menu delete 0 end

        $w.f.menu.mkrs configure -state disabled
    }

    method setMarkers {value} {
        set w .ui[modname]

        $w.f.menu.mkrs.menu add command -label $value \
		-command "$this selectMarker"

        $w.f.menu.mkrs configure -state normal
    }

    method setWindowTitle {value} {
        set w .ui[modname]

	wm title $w $value
    }

    method initialize_ui { {my_display "local"} } {
        $this ui
	
	if {[winfo exists .ui[modname]]!= 0} {
	    set w .ui[modname]

	    SciRaise $w

	    #wm title $w "Executive State"
	    wm title $w "Statistical Forecast"
	    #wm minsize $w 640 128
	    #wm geometry $w "640x640"
	    wm geometry $w [set $this-geom]
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
	pack $w.f -padx 2 -pady 2 -fill both -expand 1
        create_gl
    }
}
    
