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


itcl_class SCIRun_Visualization_ShowColorMap {
    inherit Module
    constructor {config} {
        set name ShowColorMap
	global $this-length       # One of three lengths to use.
	global $this-side         # Which side to put the map on.
	global $this-numlabels    # How many labels to use on the map.
	global $this-scale
	global $this-numsigdigits
	global $this-units
	global $this-text_color
	global $this-text-fontsize
	global $this-extra-padding

        set_defaults
    }

    method set_defaults {} {
	set $this-length half2
	set $this-side left
	set $this-numlabels 5
	set $this-scale 1.0
	set $this-numsigdigits 2
	set $this-units ""
	set $this-text_color 1
	set $this-text-fontsize 2
	set $this-extra-padding 0
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w

	frame $w.side_length
	frame $w.tc_uns

	frame $w.tc_uns.uns

	frame $w.side_length.side -relief groove -borderwidth 2
	frame $w.side_length.length -relief groove -borderwidth 2
	frame $w.tc_uns.uns.units -borderwidth 2
	frame $w.tc_uns.uns.nlabs -borderwidth 2
	frame $w.tc_uns.uns.scale -borderwidth 2
	frame $w.tc_uns.uns.ndigits -borderwidth 2
	
	label $w.side_length.side.label -text "Display Side"
	radiobutton $w.side_length.side.left -text "Left" -variable $this-side -value left -command "$this-c needexecute"
	radiobutton $w.side_length.side.bottom -text "Bottom" -variable $this-side -value bottom -command "$this-c needexecute"

	label $w.side_length.length.label -text "Display Length"
	radiobutton $w.side_length.length.full -text "Full" -variable $this-length -value full -command "$this-c needexecute"
	radiobutton $w.side_length.length.half1 -text "First Half" -variable $this-length -value half1 -command "$this-c needexecute"
	radiobutton $w.side_length.length.half2 -text "Second Half" -variable $this-length -value half2 -command "$this-c needexecute"

	frame $w.tc_uns.cf -relief groove -borderwidth 2
	label $w.tc_uns.cf.tcolor -text "Text Color"
	radiobutton $w.tc_uns.cf.white -text "White" -variable $this-text_color \
	    -value 1 -command "$this-c needexecute"
	radiobutton $w.tc_uns.cf.black -text "Black" -variable $this-text_color \
	    -value 0 -command "$this-c needexecute"

	label $w.tc_uns.uns.nlabs.label -text "Labels"
	entry $w.tc_uns.uns.nlabs.entry -width 10 -textvariable $this-numlabels
	bind $w.tc_uns.uns.nlabs.entry <KeyPress-Return> "$this-c needexecute"
	
	label $w.tc_uns.uns.units.label -text "Units"
	entry $w.tc_uns.uns.units.entry -width 10 -textvariable $this-units
	bind $w.tc_uns.uns.units.entry <KeyPress-Return> "$this-c needexecute"


	label $w.tc_uns.uns.scale.label -text "Scale"
	entry $w.tc_uns.uns.scale.entry -width 10 -textvariable $this-scale
	bind $w.tc_uns.uns.scale.entry <KeyPress-Return> "$this-c needexecute"

	label $w.tc_uns.uns.ndigits.label -text "Significant\nDigits"
	entry $w.tc_uns.uns.ndigits.entry -width 10 \
	    -textvariable $this-numsigdigits
	bind $w.tc_uns.uns.ndigits.entry \
	    <KeyPress-Return> "$this-c needexecute"

	# Pack Display Length Frame
	pack $w.side_length.length.label $w.side_length.length.full $w.side_length.length.half1 \
	    $w.side_length.length.half2 -anchor w -padx 4 -pady 2

	# Pack the Display Side Frame
	pack $w.side_length.side.label $w.side_length.side.left $w.side_length.side.bottom -anchor w \
	    -padx 4 -pady 2

	# Pack the Text Color Frame
	pack $w.tc_uns.cf.tcolor $w.tc_uns.cf.white $w.tc_uns.cf.black -anchor w -padx 4 -pady 2

	# Pack the Labels/Units/Scale Widgets Frame
	pack $w.tc_uns.uns.nlabs.label $w.tc_uns.uns.nlabs.entry -side left -anchor e
	pack $w.tc_uns.uns.units.label $w.tc_uns.uns.units.entry -side left -anchor e
	pack $w.tc_uns.uns.scale.label $w.tc_uns.uns.scale.entry -side left -anchor e
	pack $w.tc_uns.uns.ndigits.label $w.tc_uns.uns.ndigits.entry -side left -anchor e
	pack $w.tc_uns.uns.nlabs $w.tc_uns.uns.units $w.tc_uns.uns.scale $w.tc_uns.uns.ndigits -side top \
          -padx 5 -pady 2 -anchor e

	# Pack the "Display Length" and "Display Side" Frame
	pack $w.side_length.side $w.side_length.length -padx 4  -fill y -side right

	# Pack the "Text Color" and "Units/Labels/Scale" Frame
	pack $w.tc_uns.uns $w.tc_uns.cf -side right -padx 4 -fill y -side right

	pack $w.side_length -padx 4 -pady 4 -anchor w
	pack $w.tc_uns      -padx 4 -pady 4 -anchor w

	make_labeled_radio $w.size \
	    "Text Size:" "$this-c needexecute" left \
	    $this-text-fontsize \
	    {{"T" 0} {"S" 1} {"M" 2} {"L" 3} {"XL" 4}}

	pack $w.size -side top -fill x -pady 3

	checkbutton $w.extra -text "Add Extra Space" -variable $this-extra-padding -command "$this-c needexecute"

	pack $w.extra -anchor w -side top -pady 3 -padx 10

	makeSciButtonPanel $w $w $this
	moveToCursor $w
    }
}



