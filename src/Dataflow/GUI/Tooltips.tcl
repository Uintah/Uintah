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
global tooltipDelayMS
set tooltipDelayMS 1000
global tooltipID
set tooltipID 0
global tooltipsOn
set tooltipsOn 1


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
    puts "message: $message"
    Tooltip $w "$message"
}

#  TooltipMultiWidget widgets msg
#
#     This convenience function creates the same tooltip for multiple
#     widgets.  "widgets" must be a list of widgets.
#
proc TooltipMultiWidget {widgets msg} {
    for {set arg 0} {$arg < [llength $widgets] } { incr arg } {
	set widget [lindex $widgets $arg]
	Tooltip $widget $msg
    }
}

proc Tooltip {w msg} {

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
    set t .balloon_help
    catch {destroy $t}
    toplevel $t
    wm overrideredirect $t 1
    if {$::tcl_platform(platform) == "macintosh"} {
	unsupported1 style $t floating sideTitlebar
    }
    pack [label $t.l -text $msg -justify left -relief groove -bd 1 -bg white] -fill both
#    set x [expr [winfo rootx $w]+6+[winfo width $w]/2]
#    set y [expr [winfo rooty $w]+6+[winfo height $w]/2]
    set x [expr $x+6]
    set y [expr $y+6]

    wm geometry $t +$x\+$y
    bind $t <Enter> {after cancel {catch {destroy .balloon_help}}}
    bind $t <Leave> "catch {destroy .balloon_help}"
}