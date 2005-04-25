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

proc incrementFrame { this } {
    upvar \#0 $this-frame frame
    upvar \#0 $this-inc inc    
    incr frame $inc
    setGlobal $this-frameReady 1
    setGlobal $this-frameCancelID ""
    setGlobal $this-frameReadyTime [clock clicks -milliseconds]
    
    puts "incrementFrame $frame $inc [uplevel \#0 set $this-frameReadyTime]"
    $this-c needexecute
}

proc cancelFrame { this } {
    upvar \#0 $this-frameCancelID frameCancelID
    if { $frameCancelID != "" } {
	after cancel $frameCancelID
	set frameCancelID ""
    }
}



itcl_class SCIRun_Render_Camera {
    inherit Module 

    constructor {config} {
        set name Camera
        set_defaults
    }

    method set_defaults {} {    
	setGlobal $this-playmode		once
	setGlobal $this-execmode		pause
	setGlobal $this-frame			1
	setGlobal $this-inc			1
	setGlobal $this-num_frames		100
	setGlobal $this-delay			500
	setGlobal $this-time			0
	setGlobal $this-execmode		init
	setGlobal $this-track			1
	setGlobal $this-B			1.0
	setGlobal $this-C			0.0
	setGlobal $this-frameReady		1
	setGlobal $this-frameCancelID		""
    }

    method post_next_frame {} {
	upvar \#0 $this-frame frame
	upvar \#0 $this-execmode execmode 
	if { $execmode == "pause" } {
	    return
	}

	upvar \#0 $this-playmode playmode 
	upvar \#0 $this-num_frames num_frames
	upvar \#0 $this-inc inc

	puts "post_next_frame $frame $execmode $playmode $num_frames $inc"

	set lastframe [expr $num_frames - 1]

	if { $frame < 1 } {
	    set frame 1
	}

	if { $frame > $lastframe } {
	    set frame $lastframe
	}

	if { $playmode == "once" && $frame == $lastframe } return

	if { $playmode == "loop" } {
	    if { $frame == $lastframe } {
		set frame 1
	    }
	} elseif { $playmode == "bounce" } {
	    if { $frame == 0 } {
		set inc 1
	    }
	    if { $frame == $lastframe } {
		set inc -1
	    }
	}

	upvar \#0 $this-frameReady frameReady
	upvar \#0 $this-frameCancelID frameCancelID
	upvar \#0 $this-frameReadyTime frameReadyTime
	upvar \#0 $this-delay delay

	set frameReady 0
	set clicks [clock clicks -milliseconds]
	if { ![info exists frameReadyTime] } {
	    set frameReadyTime $clicks
	}
	set next_frame_delay [expr $frameReadyTime+$delay-$clicks]
	puts "--- $frameReady $frameCancelID $frameReadyTime + $delay - $clicks = $next_frame_delay"
	if { $next_frame_delay > 0 } {
	    puts "after $next_frame_delay incrementFrame $this"
	    set frameCancelID [after $next_frame_delay "incrementFrame $this"]
	} else {
	    puts -nonewline "pushing out frame now:  "
	    incrementFrame $this
	}
    }

    method rewind {} {
	cancelFrame $this
	setGlobal $this-execmode pause
	setGlobal $this-frame 1
	setGlobal $this-frameReady 1
	$this-c needexecute
    }

    method stepb {} {
	cancelFrame $this
	setGlobal $this-execmode pause
	upvar \#0 $this-frame frame
	incr frame -1
	setGlobal $this-frameReady 1
	$this-c needexecute
    }

    method pause {} {
	cancelFrame $this
	setGlobal $this-execmode pause
	setGlobal $this-frameReady 1
	$this-c needexecute
    }

    method play {} {
	setGlobal $this-frameReady 1
	setGlobal $this-frameReadyTime [clock clicks -milliseconds]
	setGlobal $this-execmode play
	$this-c needexecute
    }


    method stepf {} {
	cancelFrame $this
	setGlobal $this-execmode pause
	upvar \#0 $this-frame frame
	incr frame 1
	$this-c needexecute
    }

    method fastforward {} {
	cancelFrame $this
	upvar \#0 $this-num_frames num_frames
	setGlobal $this-execmode pause
	setGlobal $this-frame $num_frames
	setGlobal $this-frameReady 1
	$this-c needexecute
    }

    method create_keyframe_views_ui_frame { parent_frame } {
	set w $parent_frame
	set f $w.time
	frame $f
	label $f.timelabel -text "Keyframe \#:"
	entry $f.time -textvariable $this-time -width 4
	pack $f.timelabel $f.time -side left -expand yes -fill both

	set f $w.fps
	frame $f
	label $f.label -text "Total Frames:"
	entry $f.entry -textvariable $this-num_frames -width 4 
#	trace variable $this-num_frames w "[$W.tabs childsite 0].posf.cur configure -to \[set $this-num_frames\];\#"
	pack $f.label $f.entry -side left -expand yes -fill x -anchor w

	set f $w.bee
	frame $f
	label $f.label -text B:
	entry $f.entry -textvariable $this-B -width 4
	pack $f.label $f.entry -side left -expand yes -fill both

	set f $w.cee
	frame $f
	label $f.label -text C:
	entry $f.entry -textvariable $this-C -width 4
	pack $f.label $f.entry -side left -expand yes -fill both

	set f $w.trackf
	frame $f
	label $f.label -text "Track Type"
	radiobutton $f.at -text "Track At" \
	    -variable $this-track -value 1
	radiobutton $f.from -text "Track From" \
	    -variable $this-track -value 2
	radiobutton $f.both -text "Track Both" \
	    -variable $this-track -value 3
	pack $f.at $f.from $f.both -side top -expand yes -fill both -anchor w


	set f $w
	button $f.add -text "Add Keyframe" -command "$this-c add_frame; incr $this-time"
	button $f.build -text "Build Camera Path" -command "$this-c create_frames"

	pack $f.add $w.time $w.fps $w.trackf $w.bee $w.cee $w.build -side top -expand 0 -fill x
	
    }


    method create_vcr_ui_frame { parent_frame } {
	set w $parent_frame

	frame $w.playmode -relief groove -borderwidth 2
	frame $w.vcr -relief groove -borderwidth 2
        set playmode $w.playmode
	set vcr $w.vcr

	# load the VCR button bitmaps
	set image_dir /gozer/SCIRun/src/pixmaps
	set rewind [image create photo -file ${image_dir}/rewind-icon.ppm]
	set stepb [image create photo -file ${image_dir}/step-back-icon.ppm]
	set pause [image create photo -file ${image_dir}/pause-icon.ppm]
	set play [image create photo -file ${image_dir}/play-icon.ppm]
	set stepf [image create photo -file ${image_dir}/step-forward-icon.ppm]
	set fforward [image create photo -file ${image_dir}/fast-forward-icon.ppm]

	# Create and pack the VCR buttons frame
	button $vcr.rewind -image $rewind -command "$this rewind"
	button $vcr.stepb -image $stepb -command "$this stepb"
	button $vcr.pause -image $pause -command "$this pause"
	button $vcr.play -image $play -command "$this play"
	button $vcr.stepf -image $stepf -command "$this stepf"
	button $vcr.fforward -image $fforward -command "$this fastforward"

	pack $vcr.rewind $vcr.stepb $vcr.pause \
	    $vcr.play $vcr.stepf $vcr.fforward -side left -fill both -expand 1
	global ToolTipText
	Tooltip $vcr.rewind $ToolTipText(VCRrewind)
	Tooltip $vcr.stepb $ToolTipText(VCRstepback)
	Tooltip $vcr.pause $ToolTipText(VCRpause)
	Tooltip $vcr.play $ToolTipText(VCRplay)
	Tooltip $vcr.stepf $ToolTipText(VCRstepforward)
	Tooltip $vcr.fforward $ToolTipText(VCRfastforward)

	set f $w.posf
	frame $f
        scale $f.cur -variable $this-frame \
	    -showvalue true -orient horizontal -relief groove -length 200
	pack $f.cur -fill x
	
	
	# Create and pack the play mode frame
	label $playmode.label -text "Play Mode"
	radiobutton $playmode.once -text "Once" \
	    -variable $this-playmode -value once
	radiobutton $playmode.loop -text "Loop" \
	    -variable $this-playmode -value loop
	radiobutton $playmode.bounce1 -text "Bounce" \
	    -variable $this-playmode -value bounce

	# Save the delay since the iwidget resets it
	global $this-delay
	set delay [set $this-delay]
	iwidgets::spinint $playmode.delay -labeltext {Step Delay (ms)} \
	    -range {0 86400000} -justify right -width 5 -step 10 \
	    -textvariable $this-delay -repeatdelay 300 -repeatinterval 10
	
	$playmode.delay delete 0 end
	$playmode.delay insert 0 $delay
	trace variable $this-delay w "$this maybeRestart;\#"

	pack $playmode.label -side top -expand yes -fill both
	pack $playmode.once $playmode.loop \
		$playmode.bounce1  $playmode.delay \
	        -side top -anchor w

        pack $w.vcr $w.posf $w.playmode  -padx 5 -pady 5 -fill x -expand 0
    }



    method ui_ {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w

	# create tabs
	iwidgets::tabnotebook $w.tabs -equaltabs 1 -raiseselect 1 \
	    -tabpos n -auto 1 -height 400
	$w.tabs add -label "Play Mode"
	$w.tabs add -label "Build Mode"
	$w.tabs select 0
	create_vcr_ui_frame [$w.tabs childsite 0]
	create_keyframe_views_ui_frame [$w.tabs childsite 1]
	pack $w.tabs -expand 1 -fill both

	# Create the sci button panel
	makeSciButtonPanel $w $w $this "-no_execute"

	moveToCursor $w
    }

}


proc update_slider_max { slider var args } {
    upvar \#0 $var val
    $slider configure -to $val
}
    