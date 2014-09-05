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
global tooltipDelayMS
set tooltipDelayMS 1000
global tooltipID
set tooltipID 0
global tooltipsOn
set tooltipsOn 1
global tooltipsDontShow
set tooltipsDontShow 0


# TooltipMultiline allows the caller to put the tool tip string on
# separate lines (as separate substrings)
#
# eg:
#     TooltipMultiline window \
#        "this is line one\n" \
#        "this is line two\n" \
#        "etc"
#
proc TooltipMultiline { w args } {

    set message [lindex $args 0]

    for {set arg 1} {$arg < [llength $args] } { incr arg } {
	set message "$message[lindex $args $arg]"
    }
    Tooltip $w "$message"
}

#  TooltipMultiWidget widgets msg
#
#     This convenience function creates the same tooltip for multiple
#     widgets.  "widgets" must be a list of widgets (Use "" around the
#     names of each widget.).
#
proc TooltipMultiWidget {widgets msg} {
    for {set arg 0} {$arg < [llength $widgets] } { incr arg } {
	set widget [lindex $widgets $arg]
	Tooltip $widget $msg
    }
}

proc Tooltip {w args} {
    set msg [join $args ""]
    set msg [join [split $msg \"] \\\"]
    global tooltipDelayMS tooltipID tooltipsOn
    bind $w <Enter> "global tooltipID tooltipsOn
                     if \[set tooltipsOn\] \{
		        after cancel \[set tooltipID\]
                        after 100 {catch {destroy .balloon_help}}
                        set tooltipID \[after $tooltipDelayMS \"balloon_aux %W %X %Y [list $msg]\"\]
                     \}"
    bind $w <Leave> "global tooltipID tooltipsOn
                     if \[set tooltipsOn\] \{
                        after cancel \[set tooltipID\]
                        after 100 {catch {destroy .balloon_help}}
                     \}"
}

proc TooltipDisabled {w args} {
    set msg [join $args ""]
    set msg [join [split $msg \"] \\\"]
    global tooltipDelayMS tooltipID tooltipsOn
    bind $w <Enter> "+;global tooltipID tooltipsOn
                     if \{ \[set tooltipsOn\] && \[%W cget -state\] == \"disabled\"\} \{
		        after cancel \[set tooltipID\]
                        set tooltipID \[after $tooltipDelayMS \"balloon_aux %W %X %Y [list $msg]\"\]
                     \}"
}

proc canvasTooltip {canvas w msg} {       
    global tooltipDelayMS tooltipID tooltipsOff
    $canvas bind $w <Enter> "global tooltipID tooltipsOn
                             if \[set tooltipsOn\] \{
		                after cancel \[set tooltipID\]
                                after 100 {catch {destroy .balloon_help}}
                                set tooltipID \[after $tooltipDelayMS \"balloon_aux %W %X %Y [list $msg]\"\]
                             \}"
    $canvas bind $w <Leave> "global tooltipID tooltipsOn
                             if \[set tooltipsOn\] \{
                                after cancel \[set tooltipID\]
                                after 100 {catch {destroy .balloon_help}}
                             \}"
}


# the following code is courtesy the TCLer's Wiki (http://mini.net/tcl/)
proc balloon_aux {w x y msg} {
    global tooltipsDontShow
    if $tooltipsDontShow return
    set t .balloon_help
    catch {destroy $t}
    toplevel $t
    wm overrideredirect $t 1
    if {$::tcl_platform(os) == "Darwin"} {
#	unsupported1 style $t floating sideTitlebar
    }
    pack [label $t.l -text $msg -justify left -relief groove -bd 1 -bg white] -fill both
#    set x [expr [winfo rootx $w]+6+[winfo width $w]/2]
#    set y [expr [winfo rooty $w]+6+[winfo height $w]/2]
    set padding 12
    set x [expr $x+$padding]
    set y [expr $y+$padding]

# let's move the tool tip to the left if it is going
# to be cutoff on the right of the screen

    set cutoff_right [expr [expr $x+[winfo reqwidth $t.l]]-[winfo screenwidth .]]
    if {$cutoff_right > 0} {set x [expr $x-$cutoff_right-$padding]}

# let's put the tooltip above the cursor if it is going
# to be cutoff on the bottom of the screen

    set cutoff_bottom [expr [expr $y+[winfo reqheight $t.l]]-[winfo screenheight .]]
    if {$cutoff_bottom > 0} {set y [expr $y-[winfo reqheight $t.l]-(2*$padding)]}

    wm geometry $t +$x\+$y
    bind $t <Enter> {after cancel {catch {destroy .balloon_help}}}
    bind $t <Leave> "catch {destroy .balloon_help}"
}

proc pressProc {} {
    global tooltipsDontShow
    set tooltipsDontShow 1
}

proc releaseProc {} {
    global tooltipsDontShow
    set tooltipsDontShow 0
}

bind all <ButtonPress> "+pressProc"
bind all <ButtonRelease> "+releaseProc"
