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


#   BioPSEFilebox.tcl
#   Created by:
#      Samsonov Alexei
#      October 2000
#      based on standard TCL tkfbox.tcl source
#
#       Modified by:
#          Elisha Hughes
#          December 2004

#
# NOTE: If you use makeOpenFilebox or makeSaveFilebox and these
# dialogs ARE NOT the main module GUI (ie: created by the ui method
# _as_ the UI), then you will need to explicitly deiconify the window 
# (and move it to the location of the mouse...)
#
# ie: something like this:
#
#       set w .my_window
#
#       if {[winfo exists $w]} {
#           if { [winfo ismapped $w] == 1} {
#               raise $w
#           } else {
#               wm deiconify $w
#           }
#           return
#        }
#
#       toplevel $w
#       makeSaveFilebox \
#               -parent $w \
#               -filevar $this-saveFile \
#               -setcmd "wm withdraw $w" \ 
#               -command "$this doSaveImage; wm withdraw $w" \
#               -commandname "execute" \
#               -cancel "wm withdraw $w" \
#               -title $title \
#               -filetypes $types \
#               -initialfile $defname \
#               -initialdir $initdir \
#               -defaultextension $defext \
#               -formatvar $this-saveType \
#               -formats {ppm raw "by_extension"} \
#               -imgwidth $this-resx \
#               -imgheight $this-resy
#       moveToCursor $w
#       wm deiconify $w
#
#  However, if the save dialog is the main UI window, then the module UI
#  function takes care of raising it for you.
#
#  Other useful information:
#         -allowMultipleFiles <module (ie: $this)>   <- Turns on the multiple file reader tab
#
#  NOTE: For a module to allowMultipleFiles, it needs to implement two functions:
#
#               method handleMultipleFiles { { filesList "" } { delay -1 } }
#
#               method setMultipleFilePlayMode { mode }
#
#        Valid 'mode's are 'stop', 'fforward', 'rewind', 'stepb', 'step', and 'play'.  
#        See FieldReader.tcl for an example.


proc makeOpenFilebox {args} {
    biopseFDialog $args
}

proc makeSaveFilebox {args} {
    biopseFDialog $args
}

#--------------------------------------------------------
# procedures to call from readers/writers

# tkfbox.tcl --
#       Implements the "TK" standard file selection dialog box. This
#       dialog box is used on the Unix platforms whenever the tk_strictMotif
#       flag is not set.
#       The "TK" standard file selection dialog box is similar to the
#       file selection dialog box on Win95(TM). The user can navigate
#       the directories by clicking on the folder icons or by
#       selectinf the "Directory" option menu. The user can select
#       files by clicking on the file icons or by entering a filename
#       in the "Filename:" entry.
# Copyright (c) 1994-1996 Sun Microsystems, Inc.
# See the file "license.terms" for information on usage and redistribution
# of this file, and for a DISCLAIMER OF ALL WARRANTIES.

#----------------------------------------------------------------------
#                     I C O N   L I S T
# This is a pseudo-widget that implements the icon list inside the 
# biopseFDialog dialog box.
#----------------------------------------------------------------------

# biopseIconList --
#       Creates an IconList widget.
proc biopseIconList {w args} {
    upvar #0 $w data

    biopseIconList_Config $w $args
    biopseIconList_Create $w
}

# biopseIconList_Config --
#       Configure the widget variables of IconList, according to the command
#       line arguments.
proc biopseIconList_Config {w argList} {
    upvar #0 $w data

    # 1: the configuration specs
    set specs {
        {-browsecmd "" "" ""}
        {-command "" "" ""}
    }

    # 2: parse the arguments
    tclParseConfigSpec $w $specs "" $argList
}

# biopseIconList_Create --
#       Creates an IconList widget by assembling a canvas widget and a
#       scrollbar widget. Sets all the bindings necessary for the IconList's
#       operations.
proc biopseIconList_Create {w} {
    upvar #0 $w data

    frame $w
    set data(sbar)   [scrollbar $w.sbar -orient horizontal \
        -highlightthickness 0 -takefocus 0]
    set data(canvas) [canvas $w.canvas -bd 2 -relief sunken \
        -width 400 -height 120 -takefocus 1]
    pack $data(sbar) -side bottom -fill x -padx 2
    pack $data(canvas) -expand yes -fill both

    $data(sbar) config -command "$data(canvas) xview"
    $data(canvas) config -xscrollcommand "$data(sbar) set"

    # Initializes the max icon/text width and height and other variables
    set data(maxIW) 1
    set data(maxIH) 1
    set data(maxTW) 1
    set data(maxTH) 1
    set data(numItems) 0
    set data(curItem)  {}
    set data(noScroll) 1

    # Multiple File Playing Timer Event ID
    set data(mf_event_id) -1
    set data(mf_play_mode) "stop"
    # 1/2 second delay
    set data(mf_delay) 500
    set data(mf_file_list) ""
    # Index of the file (in mf_file_list) to send down
    set data(mf_file_number) 0
    # Creates the event bindings.
    bind $data(canvas) <Configure> "biopseIconList_Arrange $w"

    bind $data(canvas) <1>         "biopseIconList_Btn1 $w %x %y"
    bind $data(canvas) <B1-Motion> "biopseIconList_Motion1 $w %x %y"
    bind $data(canvas) <Double-1>  "biopseIconList_Double1 $w %x %y"
    bind $data(canvas) <ButtonRelease-1> "tkCancelRepeat"
    bind $data(canvas) <B1-Leave>  "biopseIconList_Leave1 $w %x %y"
    bind $data(canvas) <B1-Enter>  "tkCancelRepeat"

    bind $data(canvas) <Up>        "biopseIconList_UpDown $w -1"
    bind $data(canvas) <Down>      "biopseIconList_UpDown $w  1"
    bind $data(canvas) <Left>      "biopseIconList_LeftRight $w -1"
    bind $data(canvas) <Right>     "biopseIconList_LeftRight $w  1"
    bind $data(canvas) <Return>    "biopseIconList_ReturnKey $w"
    bind $data(canvas) <KeyPress>  "biopseIconList_KeyPress $w %A"
    bind $data(canvas) <Control-KeyPress> ";"
    bind $data(canvas) <Alt-KeyPress>  ";"

    bind $data(canvas) <FocusIn>   "biopseIconList_FocusIn $w"

    return $w
}

# biopseIconList_AutoScan --
# This procedure is invoked when the mouse leaves an entry window
# with button 1 down.  It scrolls the window up, down, left, or
# right, depending on where the mouse left the window, and reschedules
# itself as an "after" command so that the window continues to scroll until
# the mouse moves back into the window or the mouse button is released.
# Arguments:
# w -           The IconList window.
proc biopseIconList_AutoScan {w} {
    upvar #0 $w data
    global biopsePriv

    if {![winfo exists $w]} return
    set x $biopsePriv(x)
    set y $biopsePriv(y)

    if {$data(noScroll)} {
        return
    }
    if {$x >= [winfo width $data(canvas)]} {
        $data(canvas) xview scroll 1 units
    } elseif {$x < 0} {
        $data(canvas) xview scroll -1 units
    } elseif {$y >= [winfo height $data(canvas)]} {
        # do nothing
    } elseif {$y < 0} {
        # do nothing
    } else {
        return
    }

    biopseIconList_Motion1 $w $x $y
    set biopsePriv(afterId) [after 50 biopseIconList_AutoScan $w]
}

# Deletes all the items inside the canvas subwidget and reset the IconList's
# state.
proc biopseIconList_DeleteAll {w} {
    upvar #0 $w data
    upvar #0 $w:itemList itemList

    $data(canvas) delete all
    catch {unset data(selected)}
    catch {unset data(rect)}
    catch {unset data(list)}
    catch {unset itemList}
    set data(maxIW) 1
    set data(maxIH) 1
    set data(maxTW) 1
    set data(maxTH) 1
    set data(numItems) 0
    set data(curItem)  {}
    set data(noScroll) 1

    $data(sbar) set 0.0 1.0
    $data(canvas) xview moveto 0
}

# Adds an icon into the IconList with the designated image and text
proc biopseIconList_Add {w image text} {
    upvar #0 $w data
    upvar #0 $w:itemList itemList
    upvar #0 $w:textList textList

    set iTag [$data(canvas) create image 0 0 -image $image -anchor nw]
    set tTag [$data(canvas) create text  0 0 -text  $text  -anchor nw \
        -font $data(font)]
    set rTag [$data(canvas) create rect  0 0 0 0 -fill "" -outline ""]
    
    set b [$data(canvas) bbox $iTag]
    set iW [expr {[lindex $b 2]-[lindex $b 0]}]
    set iH [expr {[lindex $b 3]-[lindex $b 1]}]
    if {$data(maxIW) < $iW} {
        set data(maxIW) $iW
    }
    if {$data(maxIH) < $iH} {
        set data(maxIH) $iH
    }
    
    set b [$data(canvas) bbox $tTag]
    set tW [expr {[lindex $b 2]-[lindex $b 0]}]
    set tH [expr {[lindex $b 3]-[lindex $b 1]}]
    if {$data(maxTW) < $tW} {
        set data(maxTW) $tW
    }
    if {$data(maxTH) < $tH} {
        set data(maxTH) $tH
    }
    
    lappend data(list) [list $iTag $tTag $rTag $iW $iH $tW $tH $data(numItems)]
    set itemList($rTag) [list $iTag $tTag $text $data(numItems)]
    set textList($data(numItems)) [string tolower $text]
    incr data(numItems)
}

# Places the icons in a column-major arrangement.
proc biopseIconList_Arrange {w} {
    upvar #0 $w data

    if {![info exists data(list)]} {
        if {[info exists data(canvas)] && [winfo exists $data(canvas)]} {
            set data(noScroll) 1
            $data(sbar) config -command ""
        }
        return
    }

    set W [winfo width  $data(canvas)]
    set H [winfo height $data(canvas)]
    set pad [expr {[$data(canvas) cget -highlightthickness] + \
            [$data(canvas) cget -bd]}]
    if {$pad < 2} {
        set pad 2
    }

    incr W -[expr {$pad*2}]
    incr H -[expr {$pad*2}]

    set dx [expr {$data(maxIW) + $data(maxTW) + 8}]
    if {$data(maxTH) > $data(maxIH)} {
        set dy $data(maxTH)
    } else {
        set dy $data(maxIH)
    }
    incr dy 2
    set shift [expr {$data(maxIW) + 4}]

    set x [expr {$pad * 2}]
    set y [expr {$pad * 1}] ; # Why * 1 ?
    set usedColumn 0
    foreach sublist $data(list) {
        set usedColumn 1
        set iTag [lindex $sublist 0]
        set tTag [lindex $sublist 1]
        set rTag [lindex $sublist 2]
        set iW   [lindex $sublist 3]
        set iH   [lindex $sublist 4]
        set tW   [lindex $sublist 5]
        set tH   [lindex $sublist 6]

        set i_dy [expr {($dy - $iH)/2}]
        set t_dy [expr {($dy - $tH)/2}]

        $data(canvas) coords $iTag $x                    [expr {$y + $i_dy}]
        $data(canvas) coords $tTag [expr {$x + $shift}]  [expr {$y + $t_dy}]
        $data(canvas) coords $tTag [expr {$x + $shift}]  [expr {$y + $t_dy}]
        $data(canvas) coords $rTag $x $y [expr {$x+$dx}] [expr {$y+$dy}]

        incr y $dy
        if {($y + $dy) > $H} {
            set y [expr {$pad * 1}] ; # *1 ?
            incr x $dx
            set usedColumn 0
        }
    }

    if {$usedColumn} {
        set sW [expr {$x + $dx}]
    } else {
        set sW $x
    }

    if {$sW < $W} {
        $data(canvas) config -scrollregion "$pad $pad $sW $H"
        $data(sbar) config -command ""
        $data(canvas) xview moveto 0
        set data(noScroll) 1
    } else {
        $data(canvas) config -scrollregion "$pad $pad $sW $H"
        $data(sbar) config -command "$data(canvas) xview"
        set data(noScroll) 0
    }

    set data(itemsPerColumn) [expr {($H-$pad)/$dy}]
    if {$data(itemsPerColumn) < 1} {
        set data(itemsPerColumn) 1
    }

    if {$data(curItem) != {}} {
        biopseIconList_Select $w [lindex [lindex $data(list) $data(curItem)] 2] 0
    }
}

# Gets called when the user invokes the IconList (usually by double-clicking
# or pressing the Return key).
proc biopseIconList_Invoke {w} {
    upvar #0 $w data

    if {[string compare $data(-command) ""] && [info exists data(selected)]} {
        eval $data(-command)
    }
}

# biopseIconList_See --
#       If the item is not (completely) visible, scroll the canvas so that
#       it becomes visible.
proc biopseIconList_See {w rTag} {
    upvar #0 $w data
    upvar #0 $w:itemList itemList

    if {$data(noScroll)} {
        return
    }
    set sRegion [$data(canvas) cget -scrollregion]
    if {![string compare $sRegion {}]} {
        return
    }

    if {![info exists itemList($rTag)]} {
        return
    }


    set bbox [$data(canvas) bbox $rTag]
    set pad [expr {[$data(canvas) cget -highlightthickness] + \
            [$data(canvas) cget -bd]}]

    set x1 [lindex $bbox 0]
    set x2 [lindex $bbox 2]
    incr x1 -[expr {$pad * 2}]
    incr x2 -[expr {$pad * 1}] ; # *1 ?

    set cW [expr {[winfo width $data(canvas)] - $pad*2}]

    set scrollW [expr {[lindex $sRegion 2]-[lindex $sRegion 0]+1}]
    set dispX [expr {int([lindex [$data(canvas) xview] 0]*$scrollW)}]
    set oldDispX $dispX

    # check if out of the right edge
    if {($x2 - $dispX) >= $cW} {
        set dispX [expr {$x2 - $cW}]
    }
    # check if out of the left edge
    if {($x1 - $dispX) < 0} {
        set dispX $x1
    }

    if {$oldDispX != $dispX} {
        set fraction [expr {double($dispX)/double($scrollW)}]
        $data(canvas) xview moveto $fraction
    }
}

proc biopseIconList_SelectAtXY {w x y} {
    upvar #0 $w data

    biopseIconList_Select $w [$data(canvas) find closest \
        [$data(canvas) canvasx $x] [$data(canvas) canvasy $y]]
}

proc biopseIconList_Select {w rTag {callBrowse 1}} {
    upvar #0 $w data
    upvar #0 $w:itemList itemList

    if {![info exists itemList($rTag)]} {
        return
    }
    set iTag   [lindex $itemList($rTag) 0]
    set tTag   [lindex $itemList($rTag) 1]
    set text   [lindex $itemList($rTag) 2]
    set serial [lindex $itemList($rTag) 3]

    if {![info exists data(rect)]} {
        set data(rect) [$data(canvas) create rect 0 0 0 0 \
            -fill #a0a0ff -outline #a0a0ff]
    }

    $data(canvas) lower $data(rect)
    set bbox [$data(canvas) bbox $tTag]
    eval $data(canvas) coords $data(rect) $bbox

    set data(curItem) $serial
    set data(selected) $text
    
    if {$callBrowse} {
        if {[string compare $data(-browsecmd) ""]} {
            eval $data(-browsecmd) [list $text]
        }
    }
}

proc biopseIconList_Unselect {w} {
    upvar #0 $w data

    if {[info exists data(rect)]} {
        $data(canvas) delete $data(rect)
        unset data(rect)
    }
    if {[info exists data(selected)]} {
        unset data(selected)
    }
    set data(curItem)  {}
}

# Returns the selected item
proc biopseIconList_Get {w} {
    upvar #0 $w data

    if {[info exists data(selected)]} {
        return $data(selected)
    } else {
        return ""
    }
}


proc biopseIconList_Btn1 {w x y} {
    upvar #0 $w data

    focus $data(canvas)
    biopseIconList_SelectAtXY $w $x $y
}

# Gets called on button-1 motions
proc biopseIconList_Motion1 {w x y} {
    global biopsePriv
    set biopsePriv(x) $x
    set biopsePriv(y) $y

    biopseIconList_SelectAtXY $w $x $y
}

proc biopseIconList_Double1 {w x y} {
    upvar #0 $w data

    if {$data(curItem) != {}} {
        biopseIconList_Invoke $w
    }
}

proc biopseIconList_ReturnKey {w} {
    biopseIconList_Invoke $w
}

proc biopseIconList_Leave1 {w x y} {
    global biopsePriv

    set biopsePriv(x) $x
    set biopsePriv(y) $y
    biopseIconList_AutoScan $w
}

proc biopseIconList_FocusIn {w} {
    upvar #0 $w data

    if {![info exists data(list)]} {
        return
    }

    if {$data(curItem) == {}} {
        set rTag [lindex [lindex $data(list) 0] 2]
        biopseIconList_Select $w $rTag
    }
}

# biopseIconList_UpDown --
# Moves the active element up or down by one element
# Arguments:
# w -           The IconList widget.
# amount -      +1 to move down one item, -1 to move back one item.
proc biopseIconList_UpDown {w amount} {
    upvar #0 $w data

    if {![info exists data(list)]} {
        return
    }

    if {$data(curItem) == {}} {
        set rTag [lindex [lindex $data(list) 0] 2]
    } else {
        set oldRTag [lindex [lindex $data(list) $data(curItem)] 2]
        set rTag [lindex [lindex $data(list) [expr {$data(curItem)+$amount}]] 2]
        if {![string compare $rTag ""]} {
            set rTag $oldRTag
        }
    }

    if {[string compare $rTag ""]} {
        biopseIconList_Select $w $rTag
        biopseIconList_See $w $rTag
    }
}

# biopseIconList_LeftRight --
# Moves the active element left or right by one column
# Arguments:
# w -           The IconList widget.
# amount -      +1 to move right one column, -1 to move left one column.
proc biopseIconList_LeftRight {w amount} {
    upvar #0 $w data

    if {![info exists data(list)]} {
        return
    }
    if {$data(curItem) == {}} {
        set rTag [lindex [lindex $data(list) 0] 2]
    } else {
        set oldRTag [lindex [lindex $data(list) $data(curItem)] 2]
        set newItem [expr {$data(curItem)+($amount*$data(itemsPerColumn))}]
        set rTag [lindex [lindex $data(list) $newItem] 2]
        if {![string compare $rTag ""]} {
            set rTag $oldRTag
        }
    }

    if {[string compare $rTag ""]} {
        biopseIconList_Select $w $rTag
        biopseIconList_See $w $rTag
    }
}

#----------------------------------------------------------------------
#               Accelerator key bindings
#----------------------------------------------------------------------

# biopseIconList_KeyPress --
#       Gets called when user enters an arbitrary key in the listbox.
proc biopseIconList_KeyPress {w key} {
    global biopsePriv

    append biopsePriv(ILAccel,$w) $key
    biopseIconList_Goto $w $biopsePriv(ILAccel,$w)
    catch {
        after cancel $biopsePriv(ILAccel,$w,afterId)
    }
    set biopsePriv(ILAccel,$w,afterId) [after 500 biopseIconList_Reset $w]
}

proc biopseIconList_Goto {w text} {
    upvar #0 $w data
    upvar #0 $w:textList textList
    global biopsePriv
    
    if {![info exists data(list)]} {
        return
    }

    if {[string length $text] == 0} {
        return
    }

    if {$data(curItem) == {} || $data(curItem) == 0} {
        set start  0
    } else {
        set start  $data(curItem)
    }

    set text [string tolower $text]
    set theIndex -1
    set less 0
    set len [string length $text]
    set len0 [expr {$len-1}]
    set i $start

    # Search forward until we find a filename whose prefix is an exact match
    # with $text
    while 1 {
        set sub [string range $textList($i) 0 $len0]
        if {[string compare $text $sub] == 0} {
            set theIndex $i
            break
        }
        incr i
        if {$i == $data(numItems)} {
            set i 0
        }
        if {$i == $start} {
            break
        }
    }

    if {$theIndex > -1} {
        set rTag [lindex [lindex $data(list) $theIndex] 2]
        biopseIconList_Select $w $rTag 0
        biopseIconList_See $w $rTag
    }
}

proc biopseIconList_Reset {w} {
    global biopsePriv
    
    catch {unset biopsePriv(ILAccel,$w)}
}

#----------------------------------------------------------------------
#                     F I L E   D I A L O G
#----------------------------------------------------------------------

# biopseFDialog --
#       Implements the BIOPSE file selection dialog. This dialog is used when
#       the tk_strictMotif flag is set to false. This procedure shouldn't
#       be called directly. Call tk_getOpenFile or tk_getSaveFile instead.
#proc biopseFDialog {filename command cancel args} {
proc biopseFDialog {argstring} {
    
    # Find parent id
    set par_loc [lsearch $argstring "-parent"]
    if { $par_loc < 0 } { 
       set w .
    } else {
        set w [lindex $argstring [expr $par_loc + 1]]
    }

    upvar #0 $w data

    if {![string compare [lindex [info level -1] 0] makeOpenFilebox]} {
        set type open
    } else {
        set type save
    }

    biopseFDialog_Config $w $type $argstring
    biopseFDialog_Create $w

    # 5. Initialize the file types menu
    if {$data(-filetypes) != {}} {
        set filterloc 0
        set counter 0
        $data(typeMenu) delete 0 end
        foreach type $data(-filetypes) {
            set title  [lindex $type 0]
            set filter [lindex $type 1]
            $data(typeMenu) add command -label $title \
                -command [list biopseFDialog_SetFilter $w $type]
            if {[info exists $data(-selectedfiletype)]} {
                if {![string compare $title [set $data(-selectedfiletype)]]} {
                    set filterloc $counter
                }
            }
            set counter [expr $counter + 1]
        }
        biopseFDialog_SetFilter $w [lindex $data(-filetypes) $filterloc]
        $data(typeMenuBtn) config -state normal
        $data(typeMenuLab) config -state normal
    } else {
        set data(filter) "*"
        $data(typeMenuBtn) config -state disabled -takefocus 0
        $data(typeMenuLab) config -state disabled
    }

    biopseFDialog_UpdateWhenIdle $w

    # Set the timestamp for the current directory
    global biopse_TimeStamps biopse_ID

    if {![file isdirectory $data(selectPath)]} {
        # It is possible if the initial file was specified as a file
        # that does not exist, that the selectPath does not exist, so use bogus time.
        set biopse_TimeStamps($w) -1
    } else {
        set biopse_TimeStamps($w) [file mtime $data(selectPath)]
    }

    # set the refreshing ID to be -1.  For some reason when a
    # UI is opened or closed the map/unmap methods get called
    # about 20 times so this let it check if it is the first time
    set biopse_ID($w) -1

    # bind mapping and unmapping to start and stop update directory command
    bind $w <Map> "biopseFDialog_Mapped $w"
    bind $w <Unmap> "biopseFDialog_Unmapped $w"

    # 6. Withdraw the window, then update all the geometry information
    # so we know how big it wants to be, then center the window in the
    # display and de-iconify it.

    wm withdraw $w
    update idletasks
    set x [expr {[winfo screenwidth $w]/2 - [winfo reqwidth $w]/2 \
            - [winfo vrootx [winfo parent $w]]}]
    set y [expr {[winfo screenheight $w]/2 - [winfo reqheight $w]/2 \
            - [winfo vrooty [winfo parent $w]]}]
    wm geom $w [winfo reqwidth $w]x[winfo reqheight $w]+$x+$y
    wm title $w $data(-title)

    # 7. Set a grab and claim the focus too.

    set oldFocus [focus]
    set oldGrab [grab current $w]
    if {$oldGrab != ""} {
        set grabStatus [grab status $oldGrab]
    }

    grab $w
    focus $data(ent)
    $data(ent) delete 0 end

    $data(ent) insert 0 $data(selectFile)
    $data(ent) select from 0
    $data(ent) select to   end
    $data(ent) icursor end

    catch {focus $oldFocus}
    grab release $w

    # For now, return the window that this thing created so other
    # parts of the code can deal with it... sigh.
    return $w
}

# biopseFDialog_Config --
#       Configures the BIOPSE filedialog according to the argument list
proc biopseFDialog_Config {w type argList} {
    upvar #0 $w data
    set data(type) $type

    # 1: the configuration specs
    if {![string compare $type "open"]} {
        # options for Open-boxes
        set specs {
            {-filetypes "" "" ""}
            {-initialdir "" "" ""}
            {-parent "" "" "."}
            {-title "" "" ""}
            {-command "" "" ""}
            {-filevar "" "" ""}
            {-cancel "" "" ""}
            {-setcmd "" "" ""}
            {-defaultextension "" "" ""}
            {-selectedfiletype "" "" ""}
            {-allowMultipleFiles "" "" ""}
	    {-commandname "" "" ""}
        }
        set data(-initialfile) ""
    } else {
        # options for Save-boxes
        set specs {
            {-defaultextension "" "" ""}
            {-filetypes "" "" ""}
            {-initialdir "" "" ""}
            {-initialfile "" "" ""}
            {-parent "" "" "."}
            {-title "" "" ""}
            {-command "" "" ""}
            {-filevar "" "" ""}
            {-cancel "" "" ""}
            {-setcmd "" "" ""}
            {-formatvar "" "" ""}
            {-formats "" "" ""}
            {-splitvar "" "" ""}
            {-imgwidth "" "" ""}
            {-imgheight "" "" ""}
            {-confirmvar "" "" ""}
                {-incrementvar "" "" ""}
                {-currentvar "" "" ""}
            {-selectedfiletype "" "" ""}
            {-allowMultipleFiles "" "" ""}
	    {-commandname "" "" ""}
        }
    }

    # 2: default values depending on the type of the dialog
    if {![info exists data(selectPath)]} {
        # first time the dialog has been popped up
        set data(selectPath) [pwd]
        set data(selectFile) ""
    }

    # 3: parse the arguments
    tclParseConfigSpec $w $specs "" $argList

    if {![string compare $data(-title) ""]} {
        if {![string compare $type "open"]} {
            # set data(-title) "Open"
        } else {
            # set data(-title) "Save As"
            # Do not allow for saving "multiple files"
            set $data(-allowMultipleFiles) ""
        }
    }
    
    # 4.a: setting initial file to the filevar contents, if it is specified
    if { [info exists $data(-filevar)]} {

        set tmp [set $data(-filevar)]
        if { $tmp != "" } {
            # If the filevar is set to anything, use it... (even if the file
            # does not exist.)
            set data(-initialdir) [file dirname $tmp]
            set data(-initialfile) [file tail $tmp]
        }
    }
    
    # 4.b: set the default directory and selection according to the -initial
    #    settings
    if { $data(-initialdir) != "" } {

        if {[file isdirectory $data(-initialdir)]} {
            set data(selectPath) [glob $data(-initialdir)]
            # Convert the initialdir to an absolute path name. 
            set old [pwd]
            cd $data(selectPath)
            set data(selectPath) [pwd]
            cd $old
        } else {
            set data(selectPath) $data(-initialdir)
        }
    }

    set data(selectFile) $data(-initialfile)

    # 5. Parse the -filetypes option
    set data(-filetypes) [tkFDGetFileTypes $data(-filetypes)]

    if {![winfo exists $data(-parent)]} {
        error "bad window path name \"$data(-parent)\""
    }
}


proc biopseFDialog_Create {w} {
    
    set dataName [lindex [split $w .] end]

    upvar #0 $dataName data
    upvar #0 $w data
    global tk_library command

    # toplevel is now created in the modules UI function (as it should be)
    #toplevel $w -class TkFDialog

    # f1: the frame with the directory option menu
    set f1 [frame $w.f1]
    label $f1.lab -text "Directory:" -under 0
    set data(dirMenuBtn) $f1.menu
    set data(dirMenu) [tk_optionMenu $f1.menu [format .%s(selectPath) $dataName] ""]
    
    set data(upBtn) [button $f1.up]
    if {![info exists biopsePriv(updirImage)]} {
        set biopsePriv(updirImage) [image create bitmap -data {
	    #define updir_width 28
	    #define updir_height 16
	    static char updir_bits[] = {
     0x00, 0x00, 0x00, 0x00, 0x80, 0x1f, 0x00, 0x00, 0x40, 0x20, 0x00, 0x00,
     0x20, 0x40, 0x00, 0x00, 0xf0, 0xff, 0xff, 0x01, 0x10, 0x00, 0x00, 0x01,
     0x10, 0x02, 0x00, 0x01, 0x10, 0x07, 0x00, 0x01, 0x90, 0x0f, 0x00, 0x01,
     0x10, 0x02, 0x00, 0x01, 0x10, 0x02, 0x00, 0x01, 0x10, 0x02, 0x00, 0x01,
     0x10, 0xfe, 0x07, 0x01, 0x10, 0x00, 0x00, 0x01, 0x10, 0x00, 0x00, 0x01,
     0xf0, 0xff, 0xff, 0x01};}]
    }

    $data(upBtn) config -image $biopsePriv(updirImage)
    
    $f1.menu config -takefocus 1 -highlightthickness 2
 
    pack $data(upBtn) -side right -padx 4 -fill both
    pack $f1.lab -side left -padx 4 -fill both
    pack $f1.menu -expand yes -fill both -padx 4

    # data(icons): the IconList that list the files and directories.
    set data(icons) [biopseIconList $w.icons \
        -browsecmd "biopseFDialog_ListBrowse $w" \
        -command   "biopseFDialog_SetCmd $w"]

    if { $data(-allowMultipleFiles) != "" } {
        iwidgets::tabnotebook $w.tabs -raiseselect true -tabpos n -backdrop gray
        # sd = Single Data, md = Multiple Data
        set sd_tab [$w.tabs add -label "Single File"]
        set md_tab [$w.tabs add -label "Time Series"]
        
        $w.tabs view 0

        # sd_tab: the tab with the "file name" lab/entry.
        label $sd_tab.lab -text "File name:" -anchor e -width 14 -under 5 -pady 0
        set data(ent) [entry $sd_tab.ent]

        # md_tab: the tab for inputing multiple files

        ## Frame for VCR Buttons
        frame $md_tab.vcr -relief groove -borderwidth 2
        ## Frame for file base, and delay
        frame $md_tab.f1
        ## Frame for current file.
        frame $md_tab.f2

        label $md_tab.f1.lab -text "File base:" -anchor e -width 10 -pady 0
        set data(md_ent) [entry $md_tab.f1.ent]
        label $md_tab.f1.delay_lab -text "Delay:" -anchor e -width 8 -pady 0
        set data(md_delay) [entry $md_tab.f1.delay_ent]

        if { [info exists $data(-filevar)]} {
            label $md_tab.f2.current_file_lab -text "Current File:" -anchor e -width 10 -pady 0
            entry $md_tab.f2.current_file_ent -textvariable  $data(-filevar)
        }

        TooltipMultiWidget "$md_tab.f1.ent $md_tab.f1.lab" "Select first file in series."
        TooltipMultiWidget "$md_tab.f1.delay_ent $md_tab.f1.delay_lab" "Milliseconds between each file being read in."
    
        set data(mf_event_id) -1
        set data(mf_play_mode) "stop"
        set data(mf_delay) 500
        set data(mf_file_list) ""
        set data(mf_file_number) 0
        
        # load the VCR button bitmaps
        set image_dir [netedit getenv SCIRUN_SRCDIR]/pixmaps
        set rewind   [image create photo -file ${image_dir}/rewind-icon.ppm]
        set stepb    [image create photo -file ${image_dir}/step-back-icon.ppm]
        set pause    [image create photo -file ${image_dir}/pause-icon.ppm]
        set play     [image create photo -file ${image_dir}/play-icon.ppm]
        set stepf    [image create photo -file ${image_dir}/step-forward-icon.ppm]
        set fforward [image create photo -file ${image_dir}/fast-forward-icon.ppm]

        # Create and pack the VCR buttons frame
        button $md_tab.vcr.rewind -image $rewind \
            -command "setMultipleFilePlayMode $w rewind;\
                      handleMultipleFiles $w"
        button $md_tab.vcr.stepb -image $stepb \
            -command "setMultipleFilePlayMode $w stepb;\
                      handleMultipleFiles $w"
        button $md_tab.vcr.pause -image $pause \
            -command "setMultipleFilePlayMode $w stop;\
                      handleMultipleFiles $w"
        button $md_tab.vcr.play  -image $play  \
            -command "setMultipleFilePlayMode $w play;\
                      handleMultipleFiles $w"
        button $md_tab.vcr.stepf -image $stepf \
            -command "setMultipleFilePlayMode $w step;\
                      handleMultipleFiles $w"
        button $md_tab.vcr.fforward -image $fforward \
            -command "setMultipleFilePlayMode $w fforward;\
                      handleMultipleFiles $w"

        global ToolTipText
        Tooltip $md_tab.vcr.rewind $ToolTipText(VCRrewind)
        Tooltip $md_tab.vcr.stepb $ToolTipText(VCRstepback)
        Tooltip $md_tab.vcr.pause $ToolTipText(VCRpause)
        Tooltip $md_tab.vcr.play $ToolTipText(VCRplay)
        Tooltip $md_tab.vcr.stepf $ToolTipText(VCRstepforward)
        Tooltip $md_tab.vcr.fforward $ToolTipText(VCRfastforward)

        pack $md_tab.vcr.rewind $md_tab.vcr.stepb $md_tab.vcr.pause \
            $md_tab.vcr.play $md_tab.vcr.stepf $md_tab.vcr.fforward -side left -fill both -expand 1

    } else {
       # sd_tab, in the case of single file reading only, is actually a frame.
       set sd_tab [frame $w.f2 -bd 0]
       label $sd_tab.lab -text "File name:" -anchor e -width 14 -under 5 -pady 0
       set data(ent) [entry $sd_tab.ent]
    }

    # The font to use for the icons. The default Canvas font on Unix
    # is just deviant.
    global $w.icons
    set $w.icons(font) [$data(ent) cget -font]

    # f3: the frame with the cancel button and the file types field
    set f3 [frame $w.f3 -bd 0]

    # f4: the frame for Save dialog boxes with 
    # split-checkbutton (if -splitvar specified) and format menu
    set f4 [frame $w.f4 -bd 0]

    # The "File of types:" label needs to be grayed-out when
    # -filetypes are not specified. The label widget does not support
    # grayed-out text on monochrome displays. Therefore, we have to
    # use a button widget to emulate a label widget (by setting its
    # bindtags)

    set data(typeMenuLab) [button $f3.lab -text "Files of type:" \
        -anchor e -width 14 -under 9 \
        -bd [$sd_tab.lab cget -bd] \
        -highlightthickness [$sd_tab.lab cget -highlightthickness] \
        -relief [$sd_tab.lab cget -relief] \
        -padx [$sd_tab.lab cget -padx] \
        -pady [$sd_tab.lab cget -pady]]
    bindtags $data(typeMenuLab) [list $data(typeMenuLab) Label \
            [winfo toplevel $data(typeMenuLab)] all]

    set data(typeMenuBtn) [menubutton $f3.menu -indicatoron 1 -menu $f3.menu.m]
    set data(typeMenu) [menu $data(typeMenuBtn).m -tearoff 0]
    $data(typeMenuBtn) config -takefocus 1 -highlightthickness 2 \
        -relief raised -bd 2 -anchor w

   # the setBtn is created after the typeMenu so that the keyboard traversal
    # is in the right order
    # Ok, Execute, Cancel, and Find (optional) at bottom
    set f7 [frame $w.f7]
    if { [string length $data(-setcmd)] } {
        set data(setBtn)     [button $f7.ok     -text Set     -under 0 -width 10 \
                                  -default active -pady 3]
        Tooltip $f7.ok "Set the filename and dimiss\nthe UI without executing"
    }

    set bName Execute 
    if { [string length $data(-commandname)] } {
	set bName $data(-commandname)
    }
    set data(executeBtn) [button $f7.execute -text $bName -under 0 -width 10\
                              -default normal -pady 3]
    # For the viewer Save Image window, have it say Save instead of Execute
    if {[string first "ViewWindow" $w] != -1} {
        $data(executeBtn) config -text "Save"
    } 

    if { $bName == "Save"} {
	Tooltip $f7.execute "Write the file to disk"
    } elseif {$bName == "Load"} {
 	Tooltip $f7.execute "Load the file from disk"
    } else {
 	Tooltip $f7.execute "Set the filename, execute\nthe module, and dismiss\nthe UI"
    }

    set data(cancelBtn) [button $f7.cancel -text Cancel -under 0 -width 10\
			     -default normal -pady 3]
    Tooltip $f7.cancel "Dismiss the UI without\nsetting the filename"
    
    # only include a find button if this dialog has the form
    # .ui[modname].  otherwise it must be a child of a ui.
    set mod [string range $w 3 end]
    global Subnet
    set have_find [array get Subnet $mod]
    if {[llength $have_find] > 0} {
        set data(findBtn) [button $f7.find   -text Find   -under 0  -width 10 \
                               -default normal -pady 3] 
        Tooltip $f7.find "Highlights (on the Network Editor) the\nmodule that corresponds to this GUI"    
    }

    # pack the widgets in f7
    if {[llength $have_find] > 0} {
        pack $data(findBtn) -side right -padx 4 -anchor e -fill x -expand 0
    }
    pack $data(cancelBtn) -side right -padx 4 -anchor e -fill x -expand 0
    pack $data(executeBtn) -side right -padx 4 -anchor e -fill x -expand 0
    if { [string length $data(-setcmd)] } {
        pack $data(setBtn) -side right -padx 4 -anchor e -fill x -expand 0
    }

    pack $f7 -side bottom -anchor s 

    # creating additional widgets for Save-dialog box
    if {![string compare $data(type) save]} {
        set data(formatMenuLab) [button $f4.lab -text "Format:" \
                -anchor e -width 14 -under 9 \
                -bd [$sd_tab.lab cget -bd] \
                -highlightthickness [$sd_tab.lab cget -highlightthickness] \
                -relief [$sd_tab.lab cget -relief] \
                -padx [$sd_tab.lab cget -padx] \
                -pady [$sd_tab.lab cget -pady]]

        set data(formatMenuBtn) [menubutton $f4.menu -indicatoron 1 -menu $f4.menu.m]
        set data(formatMenu) [menu $data(formatMenuBtn).m -tearoff 0]

        set formats $data(-formats)
        if {[llength $formats] == 0} {
            set formats {ASCII Binary}
        }
        if {[lindex $formats 0] == "None"} {
            set formats {ASCII Binary}
            $data(formatMenuLab) configure -state disabled
            $data(formatMenuBtn) configure -state disabled
        }
        foreach f $formats {
            $data(formatMenu) add command -label $f \
                -command "biopseFDialog_SetFormat $w $f"

            if { [string compare [set $data(-formatvar)] $f] == 0} {
                biopseFDialog_SetFormat $w $f
            }
        }
                
        # creating 'Increment' and 'Current index' widgets
        if {$data(-incrementvar) != ""} {
                set f8 [frame $w.f8 -bd 0]
                label $f8.lab -text "" -width 15 -anchor e
        #add a current index label and entry
                label $f8.current_label -text "Current index: " \
                        -foreground grey64
                entry $f8.entry -text $data(-currentvar) \
                        -foreground grey64
        # add an increment checkbutton
                checkbutton $f8.button  -text "Increment " \
                        -variable $data(-incrementvar) \
                        -command "toggle_Current $w"
        #pack these widgets
            pack $f8.lab -side left -padx 2
            pack $f8.button -side left -padx 2
                        pack $f8.current_label -side left -padx 2
                        pack $f8.entry -side left -padx 2 -expand true
            pack $f8 -side bottom -fill x -pady 4
                        
                        toggle_Current $w
                }

        # setting flag if the file to be split
        if  { ![string compare [set data(-splitvar)] ""] } {
            set data(is_split) 0
        } else {
            set data(is_split) 1
        }

        if {$data(-confirmvar) != ""} {
            set f6 [frame $w.f6 -bd 0]
            label $f6.lab -text "" -width 15 -anchor e
            checkbutton $f6.button  -text "Confirm Before Overwriting File " \
              -variable $data(-confirmvar) 
            pack $f6.lab -side left -padx 2
            pack $f6.button -side left -padx 2
            pack $f6 -side bottom -fill x -pady 4
        }
            

        if {$data(-imgwidth) != ""} {
            set f5 [frame $w.f5 -bd 0]
            label $f5.resxl  -text "Width:"  -width 15 -anchor e
            entry $f5.resx -width 5 -text $data(-imgwidth)
        
            label $f5.resyl  -text "Height:" -width 15 -anchor e
            entry $f5.resy -width 5 -text $data(-imgheight)
        
            pack $f5.resxl $f5.resx $f5.resyl $f5.resy -side left -padx 2
            pack $f5 -side bottom -fill x -pady 4
        }

        $data(formatMenuBtn) config -takefocus 1 -highlightthickness 2 \
        -relief raised -bd 2 -anchor w
        set data(splitBtn) [checkbutton $f4.split -text Split -disabledforeground "" \
                -onvalue 1 -offvalue 0 -width 5 -pady 2]
        pack $data(splitBtn) -side right -padx 4 -anchor w

        if { [set data(is_split)] } {       
            $data(splitBtn) configure -state normal
            $data(splitBtn) configure -variable $data(-splitvar)
            if { [set $data(-splitvar)]!=0 } {
                $data(splitBtn) select
            } else {
                $data(splitBtn) deselect
            }
        } else {
            $data(splitBtn) configure -state disabled
        }

        pack $f4.lab -side left -padx 4
        pack $data(formatMenuBtn) -expand yes -fill x -side right
    }

    # pack the widgets in sd_tab and md_tab
    pack $sd_tab.lab -side left -padx 4
    pack $sd_tab.ent -expand yes -fill x -padx 2 -pady 0
    
    if { $data(-allowMultipleFiles) != "" } {
        pack $md_tab.vcr -fill x -padx 8 -pady 4
        pack $md_tab.f1 -fill x -expand y
        pack $md_tab.f2 -fill x -expand y
        pack $md_tab.f1.lab $md_tab.f1.ent -side left -fill x -padx 4 -pady 0
        pack $md_tab.f1.delay_lab $md_tab.f1.delay_ent -side left -fill x -padx 4 -pady 0
        if { [info exists $data(-filevar)]} {
            # Back both the label and entry on the same line...
            pack $md_tab.f2.current_file_lab $md_tab.f2.current_file_ent -side left -fill x -padx 4 -pady 0
            # ... but allow the entry to expand
            pack $md_tab.f2.current_file_ent -expand y
        }
    }

    pack $data(typeMenuLab) -side left -padx 4
    pack $data(typeMenuBtn) -expand yes -fill x -side right

    
    # Pack all the frames together. We are done with widget construction.
    pack $f1 -side top -fill x -pady 4
    pack $f4 -side bottom -fill x -pady 2
    pack $f3 -side bottom -fill x
    if { $data(-allowMultipleFiles) != "" } {
       pack $w.tabs -side bottom -fill x
    } else {
       pack $sd_tab -side bottom -fill x
    }
    pack $data(icons) -expand yes -fill both -padx 4 -pady 1

    # Set up the event handlers
    bind $data(ent) <Return>  "biopseFDialog_ActivateEnt $w"
    
    $data(upBtn)     config -command "biopseFDialog_UpDirCmd $w"
    if { [string length $data(-setcmd)] } {
        $data(setBtn)     config -command "biopseFDialog_SetCmd $w set"
    }
    $data(executeBtn) config -command "biopseFDialog_SetCmd $w execute"
    $data(cancelBtn) config -command "biopseFDialog_CancelCmd $w"
    # $data(refreshBtn) config -command "biopseFDialog_RefreshCmd $w"
    if {[llength $have_find] > 0} {
        $data(findBtn) config -command "biopseFDialog_FindCmd $w"
    }

    trace variable data(selectPath) w "biopseFDialog_SetPath $w"

    bind $w <Alt-d> "focus $data(dirMenuBtn)"
    bind $w <Alt-t> [format {
        if {"[%s cget -state]" == "normal"} {
            focus %s
        }
    } $data(typeMenuBtn) $data(typeMenuBtn)]
    bind $w <Alt-n> "focus $data(ent)"
    bind $w <KeyPress-Escape> "tkButtonInvoke $data(cancelBtn)"
    bind $w <Alt-c> "tkButtonInvoke $data(cancelBtn)"
    bind $w <Alt-o> "biopseFDialog_InvokeBtn $w Open"
    bind $w <Alt-s> "biopseFDialog_InvokeBtn $w Save"

    wm protocol $w WM_DELETE_WINDOW "biopseFDialog_CancelCmd $w"

    # Build the focus group for all the entries
    tkFocusGroup_Create $w
    tkFocusGroup_BindIn $w  $data(ent) "biopseFDialog_EntFocusIn $w"
    tkFocusGroup_BindOut $w $data(ent) "biopseFDialog_EntFocusOut $w"
}


# toggle_Current --
#       Greys out the 'Current index:' label and entry whenever the 'Increment'
#       button is unchecked.
proc toggle_Current {w} {
        upvar #0 $w data
        global $w.f8
        if {[set $data(-incrementvar)]} {
                $w.f8.current_label configure -foreground black
                $w.f8.entry configure -foreground black
        } else {
                $w.f8.current_label configure -foreground grey64
                $w.f8.entry configure -foreground grey64
        }
}

# biopseFDialog_UpdateWhenIdle --
#       Creates an idle event handler which updates the dialog in idle
#       time. This is important because loading the directory may take a long
#       time and we don't want to load the same directory for multiple times
#       due to multiple concurrent events.
proc biopseFDialog_UpdateWhenIdle {w} {
    upvar #0 $w data

    if {[info exists data(updateId)]} {
        return
    } else {
        set data(updateId) [after idle biopseFDialog_Update $w]
    }
}

# biopseFDialog_Mapped --
#       This is called when the Reader/Writer is mapped
#       meaning the ui is opened.  At this point we want
#       to start the refreshing script to update the directory
proc biopseFDialog_Mapped {w} {
    global biopse_ID

    # start calls to peridically refresh ui
    # only call if it isn't all ready running though which
    # is indicated by biopse_ID($w) not being equal to -1
    if {[info exists biopse_ID] && [info exists biopse_ID($w)] && \
            $biopse_ID($w) == -1} {
        biopseFDialog_UpdateDirectory $w
    } 
}

# biopseFDialog_Unmapped --
#       This is called when the Reader/Writer is unmapped
#       meaning the ui is closed. At this point we want to
#       kill the refreshing script.
proc biopseFDialog_Unmapped {w} {
    global biopse_ID
    # only end script if the biopse_ID($w) has a valid value
    # which indicates it is running
    if {$biopse_ID($w) != -1} {
        after cancel $biopse_ID($w)
        set biopse_ID($w) -1
    } 
}

# biopseFDialog_UpdateDirectory --
#       Checks the current modification time of the selectPath directory
#       and compares it to the old modification time stored in
#       the biopse_TimeStamps array.  If different, it updates 
#       the biopseFDialog.
proc biopseFDialog_UpdateDirectory {w} {
    upvar #0 $w data
    global biopse_TimeStamps biopse_ID

    if {[file isdirectory $data(selectPath)]} {
        set curr_time [file mtime $data(selectPath)]
        set old_time $biopse_TimeStamps($w)
    
        if {$curr_time != $old_time} {
            set biopse_TimeStamps($w) $curr_time
            biopseFDialog_Update $w
        }
    }
    set biopse_ID($w) [after 5000 "biopseFDialog_UpdateDirectory $w"]
}

# biopseFDialog_Update --
#       Loads the files and directories into the IconList widget. Also
#       sets up the directory option menu for quick access to parent
#       directories.
proc biopseFDialog_Update {w} {
    # This proc may be called within an idle handler. Make sure that the
    # window has not been destroyed before this proc is called
    if {![winfo exists $w] || [string compare [winfo class $w] TkFDialog]} {
        return
    }

    upvar #0 $w data

    if {![file isdirectory $data(selectPath)]} {
        # It is possible if the initial file was specified as a file
        # that does not exist, that the selectPath does not exist.  If so, just return.
        return
    }

    global tk_library biopsePriv
    catch {unset data(updateId)}

    if {![info exists biopsePriv(folderImage)]} {
        set biopsePriv(folderImage) [image create photo -data {
R0lGODlhEAAMAKEAAAD//wAAAPD/gAAAACH5BAEAAAAALAAAAAAQAAwAAAIghINhyycvVFsB
QtmS3rjaH1Hg141WaT5ouprt2HHcUgAAOw==}]
        set biopsePriv(fileImage)   [image create photo -data {
R0lGODlhDAAMAKEAALLA3AAAAP//8wAAACH5BAEAAAAALAAAAAAMAAwAAAIgRI4Ha+IfWHsO
rSASvJTGhnhcV3EJlo3kh53ltF5nAhQAOw==}]
    }
    set folder $biopsePriv(folderImage)
    set file   $biopsePriv(fileImage)

    set appPWD [pwd]
    if {[catch {
        cd $data(selectPath)
    }]} {
        # We cannot change directory to $data(selectPath). $data(selectPath)
        # should have been checked before biopseFDialog_Update is called, so
        # we normally won't come to here. Anyways, give an error and abort
        # action.
        tk_messageBox -type ok -parent $data(-parent) -message \
            "Cannot change to the directory \"$data(selectPath)\".\nPermission denied."\
            -icon warning
        cd $appPWD
        return
    }

    # Turn on the busy cursor. BUG?? We haven't disabled X events, though,
    # so the user may still click and cause havoc ...
    set entCursor [$data(ent) cget -cursor]
    set dlgCursor [$w         cget -cursor]
    $data(ent) config -cursor watch
    $w         config -cursor watch
    update idletasks
    
    biopseIconList_DeleteAll $data(icons)

    # Make the dir list
    foreach f [lsort -dictionary [glob -nocomplain .* *]] {
        if {![string compare $f .]} {
            continue
        }
        if {![string compare $f ..]} {
            continue
        }
        if {[file isdir ./$f]} {
            if {![info exists hasDoneDir($f)]} {
                biopseIconList_Add $data(icons) $folder $f
                set hasDoneDir($f) 1
            }
        }
    }
    # Make the file list
    if {![string compare $data(filter) *]} {
        set files [lsort -dictionary \
            [glob -nocomplain .* *]]
    } else {
        set files [lsort -dictionary \
            [eval glob -nocomplain $data(filter)]]
    }

    set top 0
    foreach f $files {
        if {![file isdir ./$f]} {
            if {![info exists hasDoneFile($f)]} {
                biopseIconList_Add $data(icons) $file $f
                set hasDoneFile($f) 1
            }
        }
    }

    biopseIconList_Arrange $data(icons)

    # Update the Directory: option menu
    set list ""
    set dir ""
    foreach subdir [file split $data(selectPath)] {
        set dir [file join $dir $subdir]
        lappend list $dir
    }

    $data(dirMenu) delete 0 end
    set var [format %s(selectPath) $w]
    foreach path $list {
        $data(dirMenu) add command -label $path -command [list set $var $path]
    }

    # Add any of the additional default directories
    # i.e. MY_SCIRUN_DATA, SCIRUN_DATA or pwd.
    # Do not add if all ready in 
    set defaultdirs ""

    $data(dirMenu) add separator 

    if { [string length [netedit getenv SCIRUN_DATA]] } {
        lappend defaultdirs [netedit getenv SCIRUN_DATA]
    }

    # MY_SCIRUN_DATA (might change)
    if { [string length [netedit getenv SCIRUN_MYDATA_DIR]] } {
        lappend defaultdirs [netedit getenv SCIRUN_MYDATA_DIR]
    }
    
    # PWD
    lappend defaultdirs $appPWD
    
    foreach path $defaultdirs {
        # some environment variables have multiple
        # paths separated by ':'
        foreach p [split $path :] {
            # If there is a slash at the end, remove it
            set p [file nativename $p]
            if {[lsearch -exact $list $p] == -1} {
                $data(dirMenu) add command -label $p -command [list set $var $p]
            }
        }
     }

    # Restore the PWD to the application's PWD
    cd $appPWD

    # turn off the busy cursor.
    $data(ent) config -cursor $entCursor
    $w         config -cursor $dlgCursor
}

# biopseFDialog_SetPathSilently --
#       Sets data(selectPath) without invoking the trace procedure
proc biopseFDialog_SetPathSilently {w path} {
    upvar #0 $w data
    
    trace vdelete  data(selectPath) w "biopseFDialog_SetPath $w"
    set data(selectPath) $path
    trace variable data(selectPath) w "biopseFDialog_SetPath $w"
}


# This proc gets called whenever data(selectPath) is set
proc biopseFDialog_SetPath {w name1 name2 op} {
    if {[winfo exists $w]} {
        upvar #0 $w data
        biopseFDialog_UpdateWhenIdle $w
    }
}

# This proc gets called whenever data(filter) is set
proc biopseFDialog_SetFilter {w type} {
    
    upvar #0 $w data
    upvar \#0 $data(icons) icons
  
    set data(filter) [lindex $type 1]
    set $data(-selectedfiletype) [lindex $type 0]
    
    $data(typeMenuBtn) config -text [lindex $type 0] -indicatoron 1

    $icons(sbar) set 0.0 0.0
    
    biopseFDialog_UpdateWhenIdle $w
}

# sets output format in response to the formatMenu entries
proc biopseFDialog_SetFormat { w format } {
    upvar #0 $w data
    
    set $data(-formatvar) $format
    $data(formatMenuBtn) configure -text [set $data(-formatvar)]
}



# biopseFDialogResolveFile --
#       Interpret the user's text input in a file selection dialog.
#       Performs:
#       (1) ~ substitution
#       (2) resolve all instances of . and ..
#       (3) check for non-existent files/directories
#       (4) check for chdir permissions
# Arguments:
#       context:  the current directory you are in
#       text:     the text entered by the user
#       defaultext: the default extension to add to files with no extension
# Return vaue:
#       [list $flag $directory $file]
#        flag = OK      : valid input
#             = PATTERN : valid directory/pattern
#             = PATH    : the directory does not exist
#             = FILE    : the directory exists but the file doesn't
#                         exist
#             = CHDIR   : Cannot change to the directory
#             = ERROR   : Invalid entry
#        directory      : valid only if flag = OK or PATTERN or FILE
#        file           : valid only if flag = OK or PATTERN
#       directory may not be the same as context, because text may contain
#       a subdirectory name
proc biopseFDialogResolveFile {context text defaultext} {

    set appPWD [pwd]

    set path [biopseFDialog_JoinFile $context $text]

    # DMW: directories have their final / lopped off by "file join"
    if {[file isdirectory $path] && [string index $path end] != "/" && \
            [string equal [file join $text] [file join $path]]} {
        set path $path/
    }

    # Only consider adding default extension if the file is not a directory
    if {![file isdirectory $path]} {
      # DMW: added second comparison so we can specify a directory
      if {[file ext $path] == "" && [string index $path end] != "/"} {
        set path "$path$defaultext"
      }
    }

    if {[catch {file exists $path}]} {
        # This "if" block can be safely removed if the following code
        # stop generating errors.
        #       file exists ~nonsuchuser
        return [list ERROR $path ""]
    }

    if {[file exists $path]} {
        if {[file isdirectory $path]} {
            if {[catch {
                cd $path
            }]} {
                return [list CHDIR $path ""]
            }
            set directory [pwd]
            set file ""
            set flag OK
            cd $appPWD
        } else {
            if {[catch {
                cd [file dirname $path]
            }]} {
                return [list CHDIR [file dirname $path] ""]
            }
            set directory [pwd]
            set file [file tail $path]
            set flag OK
            cd $appPWD
        }
    } else {
        set dirname [file dirname $path]
        if {[file exists $dirname]} {
            if {[catch {
                cd $dirname
            }]} {
                return [list CHDIR $dirname ""]
            }
            set directory [pwd]
            set file [file tail $path]
            if {[regexp {[*]|[?]} $file]} {
                set flag PATTERN
            } else {
                set flag FILE
            }
            cd $appPWD
        } else {
            set directory $dirname
            set file [file tail $path]
            set flag PATH
        }
    }

    return [list $flag $directory $file]
}


# Gets called when the entry box gets keyboard focus. We clear the selection
# from the icon list . This way the user can be certain that the input in the 
# entry box is the selection.
proc biopseFDialog_EntFocusIn {w} {
    upvar #0 $w data

    if {[string compare [$data(ent) get] ""]} {
        $data(ent) selection from 0
        $data(ent) selection to   end
        $data(ent) icursor end
    } else {
        $data(ent) selection clear
    }

    biopseIconList_Unselect $data(icons)

    if {![string compare $data(type) open]} {
        # $data(setBtn) config -text "Open"
    } else {
        # $data(setBtn) config -text "Save"
    }
}

proc biopseFDialog_EntFocusOut {w} {
    upvar #0 $w data

    $data(ent) selection clear
}


# Gets called when user presses Return in the "File name" entry.
proc biopseFDialog_ActivateEnt {w {whichBtn execute}} {
    upvar #0 $w data

    set text [string trim [$data(ent) get]]
    set list [biopseFDialogResolveFile $data(selectPath) $text \
                  $data(-defaultextension)]
    set flag [lindex $list 0]
    set path [lindex $list 1]
    set file [lindex $list 2]

    switch -- $flag {
        OK {
            if {![string compare $file ""]} {
                # user has entered an existing (sub)directory
                set data(selectPath) $path
                $data(ent) delete 0 end
            } else {
                biopseFDialog_SetPathSilently $w $path
                set data(selectFile) $file

                biopseFDialog_Done $w "" $whichBtn
            }
        }
        PATTERN {
            set data(selectPath) $path
            set data(filter) $file
        }
        FILE {
            if {![string compare $data(type) open]} {
                tk_messageBox -icon warning -type ok -parent $data(-parent) \
                    -message "File \"[file join $path $file]\" does not exist."
                $data(ent) select from 0
                $data(ent) select to   end
                $data(ent) icursor end
            } else {
                biopseFDialog_SetPathSilently $w $path
                set data(selectFile) $file
                biopseFDialog_Done $w "" $whichBtn
            }
        }
        PATH {
            tk_messageBox -icon warning -type ok -parent $data(-parent) \
                -message "Directory \"$path\" does not exist."
            $data(ent) select from 0
            $data(ent) select to   end
            $data(ent) icursor end
        }
        CHDIR {
            tk_messageBox -type ok -parent $data(-parent) -message \
               "Cannot change to the directory \"$path\".\nPermission denied."\
                -icon warning
            $data(ent) select from 0
            $data(ent) select to   end
            $data(ent) icursor end
        }
        ERROR {
            tk_messageBox -type ok -parent $data(-parent) -message \
               "Invalid file name \"$path\"."\
                -icon warning
            $data(ent) select from 0
            $data(ent) select to   end
            $data(ent) icursor end
        }
    }
}

# Gets called when user presses the Alt-s or Alt-o keys.
proc biopseFDialog_InvokeBtn {w key} {
    upvar #0 $w data

    if {![string compare [$data(setBtn) cget -text] $key]} {
        tkButtonInvoke $data(setBtn)
    }
}

# Gets called when user presses the "parent directory" button
proc biopseFDialog_UpDirCmd {w} {
    upvar #0 $w data
    global biopse_ID

    if { $data(selectPath) != "/" } {
        set data(selectPath) [file dirname $data(selectPath)]
        # Cancel the scheduled update and update right now.
        after cancel $biopse_ID($w)
        biopseFDialog_UpdateDirectory $w
    }
}

# Join a file name to a path name. The "file join" command will break
# if the filename begins with ~
proc biopseFDialog_JoinFile {path file} {
    if {[string match {~*} $file] && [file exists $path/$file]} {
        return [file join $path ./$file]
    } else {
        return [file join $path $file]
    }
}



# Gets called when user presses the "Set" button
proc biopseFDialog_SetCmd {w {whichBtn execute}} {
    upvar #0 $w data
        
    set text [biopseIconList_Get $data(icons)]

    if {[string compare $text ""]} {
        set file [biopseFDialog_JoinFile $data(selectPath) $text]
        if {[file isdirectory $file]} {
            biopseFDialog_ListInvoke $w $text
	    return
        }
    }
    biopseFDialog_ActivateEnt $w $whichBtn
}

# Gets called when user presses the "Cancel" button
proc biopseFDialog_CancelCmd {w} {
    upvar #0 $w data
    global biopsePriv

    # AS: setting file variable to "" and executing cancel command
    # set data(-filevar) ""
    eval $data(-cancel)
    #set biopsePriv(selectFilePath) ""
}

# biopseFDialog_Refresh --
#       Refresh the files and directories in the IconList.  If a 
#       current filename is present, leave focus on that.
proc biopseFDialog_RefreshCmd {w} {
     upvar #0 $w data
     biopseFDialog_Update $w
     focus $data(ent) 
}

# Gets called when user presses the "Find" button
proc biopseFDialog_FindCmd {w} {
    # Fade in Icon
    # Indicate which module by removing the .ui from $w
    set mod [string range $w 3 end]
    global Subnet
    set list [array get Subnet $mod]
    if {[llength $list] > 0} {
        fadeinIcon $mod 1 1
    }
    
}

# Gets called when user browses the IconList widget (dragging mouse, arrow
# keys, etc)
proc biopseFDialog_ListBrowse {w text} {
    upvar #0 $w data

    if {$text == ""} {
        return
    }

    set file [biopseFDialog_JoinFile $data(selectPath) $text]
    if {![file isdirectory $file]} {
        $data(ent) delete 0 end
        $data(ent) insert 0 $text

        if { $data(-allowMultipleFiles) != "" } {
           $data(md_ent) delete 0 end
           $data(md_ent) insert 0 $text
        }

        if {![string compare $data(type) open]} {
            # $data(setBtn) config -text "Open"
        } else {
            # $data(setBtn) config -text "Save"
        }
    } else {
        # $data(setBtn) config -text "Open"
    }
}

# Gets called when user invokes the IconList widget (double-click, 
# Return key, etc)
proc biopseFDialog_ListInvoke {w text} {
    upvar #0 $w data

    if {$text == ""} {
        return
    }

    set file [biopseFDialog_JoinFile $data(selectPath) $text]

    if {[file isdirectory $file]} {
        set appPWD [pwd]
        if {[catch {cd $file}]} {
            tk_messageBox -type ok -parent $data(-parent) -message \
               "Cannot change to the directory \"$file\".\nPermission denied."\
                -icon warning
        } else {
            cd $appPWD
            set data(selectPath) $file
        }
    } else {
        set data(selectFile) $file
        biopseFDialog_Done $w
    }
}

# biopseFDialog_Done --
#       Gets called when user has input a valid filename.  Pops up a
#       dialog box to confirm selection when necessary. Sets the
#       biopsePriv(selectFilePath) variable, which will break the "tkwait"
#       loop in biopseFDialog and return the selected filename to the
#       script that calls biopse_getOpenFile or biopse_getSaveFile
proc biopseFDialog_Done {w {selectFilePath ""} {whichBtn execute}} {
    upvar #0 $w data
    global biopsePriv

    if {![string compare $selectFilePath ""]} {
        set selectFilePath [biopseFDialog_JoinFile $data(selectPath) \
				$data(selectFile)]
        set biopsePriv(selectFile)     $data(selectFile)
        set biopsePriv(selectPath)     $data(selectPath)

        set confirm 1
        if { [info exists data(-confirmvar)] &&
             [string length $data(-confirmvar)] } {
            set confirm 0
        }
        
        if {$confirm && 
            [file exists $selectFilePath] && 
            ![string compare $data(type) save]} {
	    set reply [tk_messageBox -icon warning -type yesno\
			   -parent $data(-parent) -message "File\
                        \"$selectFilePath\" already exists.\nDo\
                        you want to overwrite it?"]
            if {![string compare $reply "no"]} {
                return
            }
        }
    }
    
    # AS: final steps before returning: setting filename variable and executing command
    set $data(-filevar) $selectFilePath

    if {$whichBtn == "set"} {
        eval $data(-setcmd)
    } else {
        
        if { $data(-allowMultipleFiles) != "" && [$w.tabs view] == 1 } {
	    
            # If allowing multiple files and in multi file mode...
            #puts "multi file handling... skipping given command"
	    
	    set words [split $selectFilePath /]
	    set len [llength $words]
            set idx [expr $len - 1]
	    set justname [lindex $words $idx]
	    #puts $justname
            set parts [split $justname 0123456789]
	    
	    set tmp_idx [string first [lindex $parts 0] $selectFilePath]
	    #puts $tmp_idx
	    set base [string range $selectFilePath 0 [expr $tmp_idx - 1]]
	    append base [lindex $parts 0]
            set ext [lindex $parts end]

            #puts "working with $base ... $ext"

            set fileList [lsort [glob -nocomplain $base*$ext]]
            #puts "looking at: $fileList"

            #puts "this is $data(-allowMultipleFiles)"

            set position [lsearch $fileList $selectFilePath]
            if { $position != 0 } {
                # If the selected file was not the first file in the list, then
                # trim off all files up to and including the selected file.
                set fileList [lrange $fileList $position end]
                #puts "USING this list: $fileList"
            }

            set delay [$data(md_delay) get]
            #puts "delay is $delay"

            # Should get delay from this var: $md_tab.delay_ent
	    setMultipleFilePlayMode $w "play"
            handleMultipleFiles $w "$fileList" $delay

        } else {
            # In single file mode...
            eval $data(-command)
        }
    }
}

proc bfb_do_single_step { w } {
    #puts "proc bfb_do_single_step"
    setMultipleFilePlayMode $w step
    handleMultipleFiles $w
}

# Sets the first file in "filesList" to the active file, calls
# execute, and then delays for "delay".  Repeat until "filesList"
# is empty.
proc handleMultipleFiles { w { filesList "" } { delay -1 } } {
    upvar #0 $w data
    #parray data
    #puts "------------------------ ----------------- ------------------"
    #puts $filesList

    # handleMultipleFiles can be called two ways, from
    # handleMultipleFiles itself, and from the Reader dialog.  If
    # we are in the middle of a sequence and handleMultipleFiles
    # is calling itself, but the user clicks a button, then we
    # want to cancel the current event.  
    if { $data(mf_event_id) != -1 } {
	after cancel $data(mf_event_id)
	set data(mf_event_id) -1
    }
    
    if { $delay > 0 } {
	if { $delay < 1 } {
	    puts "WARNING: casting decimal input from seconds to milliseconds!"
	    # User probably put in .X seconds... translating
	    set data(mf_delay) [expr int ($delay * 1000)]
	} else {
	    # Delays can only be integers...
	    set data(mf_delay) [expr int ($delay)]
	}
	#puts "delaying for $data(mf_delay)"
    }
    
    if { $filesList != "" } {
	set data(mf_file_list) $filesList
	set data(mf_file_number) 0
	#puts "setting file list to $data(mf_file_list)"
    }
    
    set num_files [llength $data(mf_file_list)]
    #puts "num files: $num_files"
    
    if { $num_files == 0 } {
	puts "error, no files specified..."
	return
    }

    

    #puts "mode: $data(mf_play_mode)"
    
    if { $data(mf_play_mode) == "stop" } {
	puts "play mode was stop, returning"
	return
    }
    
    if { $data(mf_play_mode) == "fforward" } {
	set data(mf_file_number) [expr $num_files - 1]
    } elseif { $data(mf_play_mode) == "rewind" } {
	set data(mf_file_number) 0
    } elseif { $data(mf_play_mode) == "stepb" } {
	if { $data(mf_file_number) == 0 || $data(mf_file_number) == 1 } {
	    set data(mf_file_number) 0
	} else {
	    incr data(mf_file_number) -2
	}
    } elseif { $data(mf_play_mode) == "step" } {
	if { $data(mf_file_number) == $num_files } {
	    set data(mf_file_number) [expr $num_files - 1]
	}
    }
    
    # Send the current file through...
    set currentFile [lindex $data(mf_file_list) $data(mf_file_number)]
    #puts "working on ([expr $data(mf_file_number)]) '$currentFile'"

    incr data(mf_file_number)
    if { $data(mf_file_number) == $num_files } {
	#loop
	set data(mf_file_number) 0
    }    

    
    set mfthis $data(-allowMultipleFiles)
    set $mfthis-filename $currentFile
    $mfthis-c needexecute
    #set remainder [lrange $filesList 1 end]
    

    if { $data(mf_play_mode) == "play"} {
	if { $data(mf_file_number) == $num_files } {
	    #loop
	    set $data(mf_file_number) 0
	}
	# If in play mode, then keep the sequence going...
	set data(mf_event_id) [after $data(mf_delay) "handleMultipleFiles $w"]
	#puts "event_id: $data(mf_event_id)"
    }
}
    
proc setMultipleFilePlayMode { w mode } {
    upvar #0 $w data
    set data(mf_play_mode) $mode
}