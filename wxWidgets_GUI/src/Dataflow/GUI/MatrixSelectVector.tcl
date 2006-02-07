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


itcl_class SCIRun_Math_MatrixSelectVector {
    inherit Module 

    constructor {config} {
        set name MatrixSelectVector
        set_defaults
    }

    method set_defaults {} {    
        setGlobal $this-row_or_col		row
        setGlobal $this-selectable_min		0
        setGlobal $this-selectable_max		100
        setGlobal $this-selectable_inc		1
	setGlobal $this-selectable_units	""
        setGlobal $this-range_min		0
        setGlobal $this-range_max		100
	setGlobal $this-playmode		once
	setGlobal $this-current			0
	setGlobal $this-execmode		init
	setGlobal $this-delay			0
	setGlobal $this-inc-amount		1
	setGlobal $this-send-amount		1
	setGlobal $this-data_series_done        0

	trace variable $this-data_series_done w "$this notify_series_done"
	trace variable $this-current w "update idletasks;\#"
    }

    method maybeRestart { args } {
	upvar \#0 $this-execmode execmode
	if ![string equal $execmode play] return
	$this-c restart
	$this-c needexecute
    }

    method notify_series_done { a1 a2 a3 } {
	upvar \#0 $this-data_series_done dsdone
	puts $dsdone
	if {$dsdone == 0} return
	puts " !!!!!!!!!!!!!!!!!!! in notify_series_done"
	foreach w [winfo children .] { 
	    if { [string first eader $w] != -1 } { 
		bfb_do_single_step $w
	    }
	}
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }

        toplevel $w

	frame $w.roc -borderwidth 0
	frame $w.playmode -relief groove -borderwidth 2
	frame $w.dependence -relief groove -borderwidth 2
	frame $w.vcr -relief groove -borderwidth 2
        set playmode $w.playmode
	set vcr $w.vcr

	# Create and pack the Row of Column frame
        radiobutton $w.roc.row -text "Row" -variable $this-row_or_col \
		-value row -command "set $this-execmode update
                                     $this-c needexecute"
        radiobutton $w.roc.col -text "Column" -variable $this-row_or_col \
		-value col -command "set $this-execmode update
                                     $this-c needexecute"
	pack $w.roc.row $w.roc.col -side left -expand yes -fill both

	# load the VCR button bitmaps
	set image_dir [netedit getenv SCIRUN_SRCDIR]/pixmaps
	set rewind   [image create photo -file ${image_dir}/rewind-icon.ppm]
	set stepb    [image create photo -file ${image_dir}/step-back-icon.ppm]
	set pause    [image create photo -file ${image_dir}/pause-icon.ppm]
	set play     [image create photo -file ${image_dir}/play-icon.ppm]
	set stepf    [image create photo -file ${image_dir}/step-forward-icon.ppm]
	set fforward [image create photo -file ${image_dir}/fast-forward-icon.ppm]

	# Create and pack the VCR buttons frame
	button $vcr.rewind -image $rewind \
	    -command "set $this-execmode rewind;   $this-c needexecute"
	button $vcr.stepb -image $stepb \
	    -command "set $this-execmode stepb;    $this-c needexecute"
	button $vcr.pause -image $pause \
	    -command "set $this-execmode stop;     $this-c needexecute"
	button $vcr.play  -image $play  \
	    -command "set $this-execmode play;     $this-c needexecute"
	button $vcr.stepf -image $stepf \
	    -command "set $this-execmode step;     $this-c needexecute"
	button $vcr.fforward -image $fforward \
	    -command "set $this-execmode fforward; $this-c needexecute"

	pack $vcr.rewind $vcr.stepb $vcr.pause \
	    $vcr.play $vcr.stepf $vcr.fforward -side left -fill both -expand 1
	global ToolTipText
	Tooltip $vcr.rewind $ToolTipText(VCRrewind)
	Tooltip $vcr.stepb $ToolTipText(VCRstepback)
	Tooltip $vcr.pause $ToolTipText(VCRpause)
	Tooltip $vcr.play $ToolTipText(VCRplay)
	Tooltip $vcr.stepf $ToolTipText(VCRstepforward)
	Tooltip $vcr.fforward $ToolTipText(VCRfastforward)


	# Save range, creating the scale resets it to defaults.
	set rmin [set $this-range_min]
	set rmax [set $this-range_max]

	# Create the various range sliders
        scale $w.min -variable $this-range_min \
	    -showvalue true -orient horizontal -relief groove -length 200 \
	    -command "$this maybeRestart"
        scale $w.cur -variable $this-current \
	    -showvalue true -orient horizontal -relief groove -length 200
        scale $w.max -variable $this-range_max \
	    -showvalue true -orient horizontal -relief groove -length 200 \
	    -command "$this maybeRestart"
        scale $w.inc -variable $this-inc-amount \
	    -showvalue true -orient horizontal -relief groove -length 200 \
	    -command "$this maybeRestart"
        scale $w.amount -variable $this-send-amount \
	    -showvalue true -orient horizontal -relief groove -length 200 \
	    -command "$this maybeRestart"

	bind $w.cur <ButtonRelease> "set $this-execmode init; $this-c needexecute"
	update_range

	# Restore range to pre-loaded value
	set $this-range_min $rmin
	set $this-range_max $rmax


	# Create and pack the play mode frame
	label $playmode.label -text "Play Mode"
	radiobutton $playmode.once -text "Once" \
	    -variable $this-playmode -value once \
	    -command "$this maybeRestart"
	radiobutton $playmode.loop -text "Loop" \
	    -variable $this-playmode -value loop \
	    -command "$this maybeRestart"
	radiobutton $playmode.bounce1 -text "Bounce" \
	    -variable $this-playmode -value bounce1 \
	    -command "$this maybeRestart"
	radiobutton $playmode.bounce2 -text "Bounce with repeating endpoints" \
	    -variable $this-playmode -value bounce2 \
	    -command "$this maybeRestart"
	radiobutton $playmode.aplay -text "Auto Play" \
	    -variable $this-playmode -value autoplay \
	    -command "$this maybeRestart"

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
	    $playmode.bounce1 $playmode.bounce2 \
	    $playmode.aplay $playmode.delay \
	    -side top -anchor w

	# Create the button to show/hide extened options
	button $w.expanded
	# Create the sci button panel
	makeSciButtonPanel $w $w $this "-no_execute"

	# Show the no-frills interface
	show_small_interface

	update
    }


    method forget_packing {} {
	set w .ui[modname]
	pack forget $w.vcr $w.roc $w.cur $w.expanded \
	    $w.min $w.max $w.inc $w.amount $w.playmode $w.dependence $w.buttonPanel
    }

    method show_small_interface {} {
	forget_packing
	set w .ui[modname]
        pack $w.vcr $w.roc $w.cur $w.expanded $w.buttonPanel \
	    -padx 5 -pady 5 -fill x -expand 0
	$w.expanded configure -text "Show Extended Options" \
	    -command "$this show_expanded_interface"
	wm geometry $w {}
    }

    method show_expanded_interface {} {
	forget_packing
	set w .ui[modname]
        pack $w.vcr $w.roc $w.min $w.cur $w.max $w.inc $w.amount $w.playmode \
	    $w.dependence $w.expanded $w.buttonPanel \
	    -padx 5 -pady 5 -fill x -expand 0
	$w.expanded configure -text "Hide Extended Options" \
	    -command "$this show_small_interface"
	wm geometry $w {}
    }




    method update_range { args } {
        set w .ui[modname]
        if {[winfo exists $w]} {
	    upvar \#0 $this-selectable_min min $this-selectable_max max 
	    upvar \#0 $this-selectable_units units $this-row_or_col roc
	    set pre $roc
	    if { [string equal $roc col] } { set pre column }

            $w.min configure -label "Start $pre $units:" \
		-from $min -to $max

            $w.cur config -label "Current $pre $units:" \
		-from $min -to $max

            $w.max config -label "End $pre $units:" \
		-from $min -to $max

            $w.inc config -label "Increment current $pre by:" \
		-from 1 -to [expr $max-$min]

            $w.amount config -label "Number of ${pre}s to send:" \
		-from 1 -to [expr $max-$min]
        }
    }
}
