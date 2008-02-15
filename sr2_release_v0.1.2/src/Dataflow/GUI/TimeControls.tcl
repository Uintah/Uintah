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


itcl_class SCIRun_Time_TimeControls {
    inherit Module 

    constructor {config} {
        set name TimeControls
        set_defaults
    }

    method set_defaults {} {    
        setGlobal $this-scale_factor		1.0
	setGlobal $this-time                    0
	setGlobal $this-execmode		init
	global zero_clicks
	set zero_clicks                   0

	trace variable $this-scale_factor w "$this-c update_scale"
    }
    method do_stepf {} {
	upvar \#0 $this-scale_factor sf
	set sf [expr $sf + 0.1]
	$this-c update_scale
    }

    method do_stepb {} {
	upvar \#0 $this-scale_factor sf
	set sf [expr $sf - 0.1]
	$this-c update_scale
    }

    method do_rewind {} {
	set w .ui[modname]
	set cur [$w.tf get -clicks]
	incr cur -600
	$w.tf show $cur
	set $this-time [$w.tf get -clicks]
	$this-c "rewind"
    }

    method do_fforward {} {
	set w .ui[modname]
	set cur [$w.tf get -clicks]
	incr cur 600
	$w.tf show $cur
	set $this-time [$w.tf get -clicks]
	$this-c "fforward"
    }

    method do_rewind_sec {} {
	set w .ui[modname]
	set cur [$w.tf get -clicks]
	incr cur -1 
	$w.tf show $cur
	set $this-time [$w.tf get -clicks]
	$this-c "rewind_sec"
    }

    method do_forward_sec {} {
	set w .ui[modname]
	set cur [$w.tf get -clicks]
	incr cur 1 
	$w.tf show $cur
	set $this-time [$w.tf get -clicks]
	$this-c "forward_sec"
    }

    method do_rewind_min {} {
	set w .ui[modname]
	set cur [$w.tf get -clicks]
	incr cur -60
	$w.tf show $cur
	set $this-time [$w.tf get -clicks]
	$this-c "rewind_min"
    }

    method do_forward_min {} {
	set w .ui[modname]
	set cur [$w.tf get -clicks]
	incr cur 60
	$w.tf show $cur
	set $this-time [$w.tf get -clicks]
	$this-c "forward_min"
    }

    method do_big_rewind {} {
	set w .ui[modname]
	$w.tf show "00:00:00"
	set $this-time [$w.tf get -clicks]
	$this-c "big_rewind"
    }
    
    method do_update {t} {
	global zero_clicks
	set w .ui[modname]
	if {[winfo exists $w]} {

	    set val [expr $t + $zero_clicks]
	    $w.tf show $val
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

	# load the VCR button bitmaps
	set image_dir [netedit getenv SCIRUN_SRCDIR]/pixmaps
	set rewind   [image create photo -file ${image_dir}/rewind-icon.ppm]
	set stepb    [image create photo -file ${image_dir}/step-back-icon.ppm]
	set pause    [image create photo -file ${image_dir}/pause-icon.ppm]
	set play     [image create photo -file ${image_dir}/play-icon.ppm]
	set stepf    [image create photo -file ${image_dir}/step-forward-icon.ppm]
	set fforward [image create photo -file ${image_dir}/fast-forward-icon.ppm]

	# Create and pack the VCR buttons frame
	#button $vcr.rewind -image $rewind -command "$this do_big_rewind"
	button $vcr.rewind -image $rewind
	bind $vcr.rewind <ButtonPress-1> "$this do_rewind_min"
	bind $vcr.rewind <Shift-ButtonPress-1> "$this do_rewind"
	#button $vcr.stepb -image $stepb -command "$this do_stepb"
	button $vcr.stepb -image $stepb -command "$this do_rewind_sec"
	button $vcr.pause -image $pause \
	    -command "set $this-execmode pause; $this-c pause"
	button $vcr.play  -image $play  \
	    -command "set $this-execmode play; $this-c play"
	#button $vcr.stepf -image $stepf -command "$this do_stepf"
	button $vcr.stepf -image $stepf -command "$this do_forward_sec"
	#button $vcr.fforward -image $fforward -command "$this do_fforward"
	button $vcr.fforward -image $fforward
	bind $vcr.fforward <ButtonPress-1> "$this do_forward_min"
	bind $vcr.fforward <Shift-ButtonPress-1> "$this do_fforward"

	pack $vcr.rewind $vcr.stepb $vcr.pause \
	    $vcr.play $vcr.stepf $vcr.fforward -side left -fill both -expand 1
	global ToolTipText
	#Tooltip $vcr.rewind "Reset time to 00:00:00"
	#Tooltip $vcr.rewind "Rewind 1 min (Press Shift to rewind 10 mins)"
	Tooltip $vcr.rewind "Rewind 1 min/10 mins"
	#Tooltip $vcr.stepb "Decrement Play Rate"
	Tooltip $vcr.stepb "Rewind 1 second"
	Tooltip $vcr.pause "Pause"
	Tooltip $vcr.play $ToolTipText(VCRplay)
	#Tooltip $vcr.stepf "Increment Play Rate"
	Tooltip $vcr.stepf "Fast forward 1 second"
	#Tooltip $vcr.fforward "Jump ahead 10 minutes"
	#Tooltip $vcr.fforward "Fast forward 1 min (Press Shift to fast forward 10 mins)"
	Tooltip $vcr.fforward "Fast forward 1 min/10 mins"



	iwidgets::timefield $w.tf -labeltext "Time:" \
	    -format military

	$w.tf show "00:00:00" 
	set $this-time [$w.tf get -clicks]
	global zero_clicks
	set zero_clicks [set $this-time]

	pack $w.tf -fill x -expand yes -padx 10 -pady 10
        pack $w.vcr $w.roc -padx 5 -pady 5 -fill x -expand 0

	button $w.big_rewind -text "Reset Time" -command "$this do_big_rewind"
	Tooltip $w.big_rewind "Reset time to 00:00:00"

	iwidgets::entryfield $w.nef -labeltext "Play Rate:" -validate real \
	    -width 6 -textvariable $this-scale_factor -fixed 7
	pack $w.big_rewind $w.nef -fill x -expand yes -padx 10 -pady 5

	# Create the sci button panel
	makeSciButtonPanel $w $w $this "-no_execute"


    }


}
