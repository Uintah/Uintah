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
#  Tooltips.tcl
#
#  Written by:
#   McKay Davis
#   Department of Computer Science
#   University of Utah
#   September 2003
#
#  Copyright (C) 2003 SCI Group
#

global Tooltip Font time_font
set Tooltip(ID) 0
set Tooltip(Color) white
set Font(Tooltip) $time_font

# MS == Miliseconds
set tooltipDelayMS 1000

proc showTooltip { id } {
    if [winfo exists .tooltip] { destroy .tooltip }
    global Tooltip Font
    toplevel .tooltip -bg black
    label .tooltip.text -text $Tooltip($id) \
	-bg $Tooltip(Color) -fg black -justify left -font $Font(Tooltip)
    pack .tooltip.text -padx 1 -pady 1
    wm overrideredirect .tooltip yes
    update idletasks
    wm geometry .tooltip +$Tooltip(X)+[expr $Tooltip(Y)-2-[winfo height .tooltip]]
}

proc enterTooltip { x y id } {
    global Tooltip tooltipDelayMS

    set Tooltip(X) $x
    set Tooltip(Y) $y
    set Tooltip(ID) [after $tooltipDelayMS "showTooltip $id"]
}

proc motionTooltip { x y id } {
    update idletasks
    global Tooltip
    set Tooltip(X) $x
    set Tooltip(Y) $y
    if ![winfo exists .tooltip] { return }
    wm geometry .tooltip +$Tooltip(X)+[expr $Tooltip(Y)-2-[winfo height .tooltip]]
}

proc leaveTooltip { } {
    if [winfo exists .tooltip] { destroy .tooltip }
    global Tooltip
    after cancel $Tooltip(ID)
}

proc canvasTooltip { id text } {
    return
    global Tooltip maincanvas
    set Tooltip($id) $text
    $maincanvas bind $id <Enter> "enterTooltip %X %Y $id"
    $maincanvas bind $id <Motion> "motionTooltip %X %Y $id"
    $maincanvas bind $id <Leave> "leaveTooltip"
    $maincanvas bind $id <Button> "leaveTooltip"
}

proc Tooltip { id text } {
    return
    global Tooltip
    set Tooltip($id) $text
    bind Tooltip$id <Enter> "enterTooltip %X %Y $id"
    bind Tooltip$id <Motion> "motionTooltip %X %Y $id"
    bind Tooltip$id <Leave> "leaveTooltip"
    bind Tooltip$id <Button> "leaveTooltip"
    bindtags $id "[bindtags $id] Tooltip$id"
}