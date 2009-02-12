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


##
 #  EditPath.tcl
 #  Written by:
 #   David Weinstein & Alexei Samsonov
 #   Department of Computer Science
 #   University of Utah
 #   February 1999, July 2000
 #  Copyright (C) 1999, 2000  SCI Group
 ##
itcl_class SCIRun_Render_EditPath {
    inherit Module
   
    constructor { config } {
        set name EditPath
        set_defaults
    }
    method set_defaults {} {

        set $this-numKeyFrames 0
        set $this-currentFrame 0
        set $this-uiInitialized 0
        set $this-updateOnExecute 0
        set $this-delay 0.1
        set $this-loop 0
        set $this-reverse 0
        set $this-showCircleWidget 0
        set $this-showPathWidgets 0
        set $this-circleNumPoints 20
        set $this-numSubKeyFrames 1.0

        set $this-pathFilename ""
    }

    method doNotSave { varName } {
        if { $varName == "numKeyFrames" ||      \
             $varName == "currentFrame" ||      \
             $varName == "showCircleWidget" ||  \
             $varName == "showPathWidgets"  ||  \
             $varName == "uiInitialized" } {
            return 1
        }
        return 0
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }

	# Withdraw immediately so it doesn't flicker when being
	# created/moved to the mouse location.
        toplevel $w; wm withdraw $w
	wm title $w "Edit Camera Path"
        wm minsize $w 500 205

        puts "here: set w $w"

        frame $w.status 
        frame $w.status.frameNum -bd 2 -relief ridge
        label $w.status.frameNum.label -text "Number of Frames:"
        label $w.status.frameNum.num   -textvar $this-numKeyFrames
        pack  $w.status.frameNum.label $w.status.frameNum.num -padx 5 -pady 2 -side left

        frame $w.status.curFrame -bd 2 -relief ridge
        label $w.status.curFrame.label -text "Current Frame:"
        label $w.status.curFrame.num   -textvar $this-currentFrame
        pack  $w.status.curFrame.label $w.status.curFrame.num -padx 5 -pady 2 -side left

        pack  $w.status.frameNum -side left -padx 5
        pack  $w.status.curFrame -side right -padx 5

        #### Edit Frame ####
        frame  $w.edit
        frame  $w.edit.buttons
        button $w.edit.buttons.addView    -text " Add View "    -command "$this-c add_vp;    $this updateButtons"
        button $w.edit.buttons.removeView -text " Delete View " -command "$this-c remove_vp; $this updateButtons"
        button $w.edit.buttons.next       -text " Next "        -command "$this-c next_view"
        button $w.edit.buttons.prev       -text " Prev "        -command "$this-c prev_view"
        button $w.edit.buttons.deletePath -text " Delete Path " -command "$this-c delete_path; $this updateButtons"
        # Vertical separators
        frame  $w.edit.buttons.sep1 -width 2 -relief sunken -borderwidth 2
        frame  $w.edit.buttons.sep2 -width 2 -relief sunken -borderwidth 2

        pack $w.edit.buttons.addView $w.edit.buttons.removeView -padx 5 -pady 2 -side left
        pack $w.edit.buttons.sep1 -fill y -padx 5 -pady 2 -side left
        pack $w.edit.buttons.prev $w.edit.buttons.next -padx 5 -pady 2 -side left
        pack $w.edit.buttons.sep2 -fill y -padx 5 -pady 2 -side left
        pack $w.edit.buttons.deletePath -padx 5 -pady 2 -side left
        pack $w.edit.buttons -pady 4

        frame  $w.edit.circle
        button $w.edit.circle.make  -text " Make Circle " -command "$this-c make_circle; $this updateButtons"
        checkbutton $w.edit.circle.showTarget -text " Show Widget " -variable "$this-showCircleWidget" \
            -command "$this-c toggleCircleWidget"
        checkbutton $w.edit.circle.showPath   -text " Show Path "   -variable "$this-showPathWidgets" \
            -command "$this-c togglePathPoints"
        label  $w.edit.circle.numPointsLab -text "Number of Points:"
        entry  $w.edit.circle.numPoints    -textvariable $this-circleNumPoints -width 5
        Tooltip $w.edit.circle.numPoints "Number of points to use in creating circular path."
        Tooltip $w.edit.circle.showTarget "Shows a widget that specifies the 'lookat' location when a circular path is created."

        pack $w.edit.circle.make $w.edit.circle.showTarget $w.edit.circle.showPath \
            $w.edit.circle.numPointsLab $w.edit.circle.numPoints \
            -padx 5 -pady 2 -side left
        pack $w.edit.circle -pady 4

        frame  $w.edit.speed
        label  $w.edit.speed.label     -text "Sub Key Frames:"
        button $w.edit.speed.updateAll -text " Update All " -command "$this-c update_all_num_key_frames"
        entry  $w.edit.speed.speed     -textvariable $this-numSubKeyFrames -width 5
        bind   $w.edit.speed.speed <Return> "$this-c update_num_key_frames"
        pack   $w.edit.speed.label $w.edit.speed.speed $w.edit.speed.updateAll -padx 5 -pady 2 -side left
        pack   $w.edit.speed -pady 4

        Tooltip          $w.edit.speed.updateAll "Update the speed for all keyframes."
        TooltipMultiline $w.edit.speed.speed "0 (the minimum) sub key frames will result in jumping from one\n" \
                                             "key frame to the next.  Press 'Return' to save this value."

        #### Run Frame ####
        frame  $w.run -bd 2 -relief groove
        frame  $w.run.buttons
        button $w.run.buttons.run  -text " Run "    -command "$this run"
        button $w.run.buttons.stop -text " Stop "   -command "$this-c stop"
        checkbutton $w.run.buttons.loop    -text " Loop Path " -variable $this-loop
        checkbutton $w.run.buttons.reverse -text " Reverse " -variable $this-reverse

        iwidgets::spinner $w.run.buttons.delay -labeltext "Delay: " \
		-width 5 -fixed 10 \
		-validate "$this spin_in %P $this-delay" \
		-decrement "$this spin_update -0.1 $w.run.buttons.delay $this-delay" \
		-increment "$this spin_update  0.1 $w.run.buttons.delay $this-delay" \
                -repeatinterval 50

	$w.run.buttons.delay insert 0 [set $this-delay]
        Tooltip $w.run.buttons.delay "Delay in seconds between each update when 'Run'ning."

        frame  $w.run.buttons2
        checkbutton $w.run.buttons2.updateOnExec -text " Sync With Execute " -variable $this-updateOnExecute \
                  -command "$this turnOnLoop"
        Tooltip $w.run.buttons2.updateOnExec "If selected, then when _ANY_ other modules execute,\nthis module will send down a new view."

        pack $w.run.buttons.run $w.run.buttons.stop $w.run.buttons.loop $w.run.buttons.reverse \
                -padx 5 -pady 2 -side left
        pack $w.run.buttons2.updateOnExec  -padx 5 -pady 2 -side left
        pack $w.run.buttons.delay -side left

        pack $w.run.buttons -pady 5
        pack $w.run.buttons2

        #### Set State on Buttons and do Final Packing ####
        updateButtons
        pack $w.status -anchor w -pady 5 -fill x
        pack $w.run    -anchor w -fill x
        pack $w.edit   -anchor w

	makeSciButtonPanel $w $w $this "\"Save\" \"$this saveToFile\"   \"Save path to file.\""    \
                                       "\"Load\" \"$this loadFromFile\" \"Load path from file.\""  \
                                       { "separator" "" "" }
	moveToCursor $w
        set $this-uiInitialized 1
    }

    # Used, during 'sync with execute', to show that this module is actually doing something.
    method highlight {} {
        fadeinIcon [modname]
    }

    method turnOnLoop {} {
        if { [set $this-updateOnExecute] } {
            set $this-loop 1
        }
    }

    method spin_in { newValue varToSet } {

	if {! [regexp "\\A\\d*\\.*\\d+\\Z" $newValue] } {
            # If it is not a number, it is not valid.
	    return 0
	} elseif { $newValue <= 0.0 || $newValue > 10.0 } {
	    return 0
	} 
	set $varToSet $newValue
	return 1
    }

    method spin_update { step widget var } {
	set newValue [expr [set $var] + $step]

        if { [spin_in $newValue $var] } {
            $widget delete 0 end
            $widget insert 0 [set newValue]
        }
    }

    method saveToFile {} {
       set w .pathSaveDialog

       if {[winfo exists $w]} {
           if { [winfo ismapped $w] == 1} {
               raise $w
           } else {
               wm deiconify $w
           }
           return
        }

	# file types to appers in filter box
	set types {
	    { {Path Files}  {.path} }
	    { {All Files}   {.*}    }
	}

       toplevel $w
       makeSaveFilebox \
               -parent $w \
               -filevar $this-pathFilename \
               -filetypes $types \
               -command "$this-c doSavePath; wm withdraw $w" \
               -commandname "Save" \
               -cancel "wm withdraw $w" \
               -title "Save Path" \
               -formats { "None" } \
               -defaultextension ".path"

       moveToCursor $w
       wm deiconify $w
    }

    method loadFromFile {} {
       set w .pathLoadDialog

       if {[winfo exists $w]} {
           if { [winfo ismapped $w] == 1} {
               raise $w
           } else {
               wm deiconify $w
           }
           return
        }

	# file types to appers in filter box
	set types {
	    { {Path Files}  {.path} }
	    { {All Files}   {.*}    }
	}

       toplevel $w
       makeOpenFilebox                                                        \
           -parent $w                                                         \
           -filevar $this-pathFilename                                        \
           -filetypes $types                                                  \
           -command "$this-c doLoadPath; wm withdraw $w; $this updateButtons" \
           -commandname "Load"                                                \
           -cancel "wm withdraw $w"                                           \
           -title "Load Path"                                                 \
           -defaultextension ".path"

       $this-c stop

       moveToCursor $w
       wm deiconify $w
    }

    method stop {} {
        if { [set $this-numKeyFrames] >= 1 } {
            set w .ui[modname]
            $w.run.buttons.run          configure -state normal
            $w.run.buttons.stop         configure -state disabled
            $w.run.buttons2.updateOnExec configure -state normal

            $w.edit.circle.make          configure -state normal
            $w.edit.speed.label          configure -state normal
            $w.edit.speed.updateAll      configure -state normal
            $w.edit.speed.speed          configure -state normal
        }
    }

    method run {} {
        if { [set $this-numKeyFrames] >= 1 } {
            set w .ui[modname]

            $w.run.buttons.run           configure -state disabled
            $w.run.buttons.stop          configure -state normal
            $w.run.buttons2.updateOnExec configure -state disabled

            $w.edit.circle.make          configure -state disabled
            $w.edit.speed.label          configure -state disabled
            $w.edit.speed.updateAll      configure -state disabled
            $w.edit.speed.speed          configure -state disabled

            set $this-updateOnExecute 0

            $this-c run
        }
    }

    method updateButtons {} {

        set w .ui[modname]
        if { [set $this-numKeyFrames] >= 2 } {
            $w.run.buttons.run          configure -state normal
            $w.run.buttons.stop         configure -state disabled
            $w.run.buttons2.updateOnExec configure -state normal

            $w.edit.buttons.next        configure -state normal
            $w.edit.buttons.prev        configure -state normal
            $w.edit.buttons.removeView  configure -state normal
            $w.edit.buttons.deletePath  configure -state normal

            $w.status.curFrame.label configure -state normal
            $w.status.curFrame.num   configure -state normal

        } elseif { [set $this-numKeyFrames] == 1 } {

            $w.run.buttons.run          configure -state disabled
            $w.run.buttons.stop         configure -state disabled
            $w.run.buttons2.updateOnExec configure -state normal

            $w.edit.buttons.next        configure -state disabled
            $w.edit.buttons.prev        configure -state disabled
            $w.edit.buttons.removeView  configure -state normal
            $w.edit.buttons.deletePath  configure -state normal

            $w.status.curFrame.label configure -state normal
            $w.status.curFrame.num   configure -state normal

        } else {

            $w.run.buttons.run          configure -state disabled
            $w.run.buttons.stop         configure -state disabled
            $w.run.buttons2.updateOnExec configure -state disabled

            $w.edit.buttons.next        configure -state disabled
            $w.edit.buttons.prev        configure -state disabled
            $w.edit.buttons.removeView  configure -state disabled
            $w.edit.buttons.deletePath  configure -state disabled

            $w.status.curFrame.label configure -state disabled
            $w.status.curFrame.num   configure -state disabled
        }
    }
}


