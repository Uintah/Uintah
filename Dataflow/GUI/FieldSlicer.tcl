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


# GUI for FieldSlicer module
# by Michael Callahan &&
#    Allen Sanderson
# December 2002

# This GUI interface is for selecting an axis and index for sub sampling a
# topologically structured field

itcl_class SCIRun_FieldsCreate_FieldSlicer {
    inherit Module
    constructor {config} {
        set name FieldSlicer
        set_defaults
    }

    method set_defaults {} {

	global $this-dims
	global $this-axis

	set $this-axis 2
	set $this-dims 3

	for {set i 0} {$i < 3} {incr i 1} {
	    if { $i == 0 } {
		set index i
	    } elseif { $i == 1 } {
		set index j
	    } elseif { $i == 2 } {
		set index k
	    }

	    global $this-$index-dim
	    global $this-$index-index
	    global $this-$index-index2

	    set $this-$index-dim 1
	    set $this-$index-index 1
	    set $this-$index-index2 "0"

	    trace variable $this-$index-dim w "$this update_setsize_callback"
	}

	trace variable $this-dims w "$this update_setsize_callback"
    }

    method ui {} {

	global $this-axis
	global $this-dims

        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }

        toplevel $w

	frame $w.main

	frame $w.main.l
	label $w.main.l.direction -text "Direction"       -width 9 -anchor w -just left
	label $w.main.l.index     -text "Slice Node"      -width 11 -anchor w -just left

	pack $w.main.l.direction -side left
	pack $w.main.l.index     -side left -padx 75

	for {set i 0} {$i < 3} {incr i 1} {
	    if { $i == 0 } {
		set index i
	    } elseif { $i == 1 } {
		set index j
	    } elseif { $i == 2 } {
		set index k
	    }

	    global $this-$index-dim
	    global $this-$index-index
	    global $this-$index-index2

	    frame $w.main.$index

	    radiobutton $w.main.$index.l -text "$index axis" -width 6 \
		-anchor w -just left -variable $this-axis -value $i

	    pack $w.main.$index.l -side left

	    scaleEntry2 $w.main.$index.index \
		0 [expr [set $this-$index-dim] - 1] 200 \
		$this-$index-index $this-$index-index2

	    pack $w.main.$index.l $w.main.$index.index -side left
	}

	if { [set $this-dims] == 3 } {
	    pack $w.main.l $w.main.i $w.main.j $w.main.k -side top -padx 10 -pady 5

	} elseif { [set $this-dims] == 2 } {
	    pack $w.main.l $w.main.i $w.main.j -side top -padx 10 -pady 5
	} elseif { [set $this-dims] == 1 } {
	    pack $w.main.l $w.main.i -side top -padx 10 -pady 5
	}

	pack $w.main -side top
	
	makeSciButtonPanel $w $w $this
	moveToCursor $w
    }


    method scaleEntry2 { win start stop length var1 var2 } {
	frame $win 
	pack $win -side top -padx 5

	scale $win.s -from $start -to $stop -length $length \
	    -variable $var1 -orient horizontal -showvalue false \
	    -command "$this updateSliderEntry $var1 $var2"

	entry $win.e -width 4 -text $var2

	bind $win.e <KeyRelease> "$this manualSliderEntry $start $stop $var1 $var2"

	pack $win.s -side left
	pack $win.e -side bottom -padx 5
    }

    method updateSliderEntry {var1 var2 someUknownVar} {
	set $var2 [set $var1]
    }

    method manualSliderEntry { start stop var1 var2 } {

	if { ![string is integer [set $var2]] } {
	    set $var2 [set $var1] }

	if { [set $var2] < $start } {
	    set $var2 $start }
	
	if { [set $var2] > $stop } {
	    set $var2 $stop }
	
	set $var1 [set $var2]
    }

    method update_setsize_callback { name1 name2 op } {
	global $this-dims
	global $this-i-dim
	global $this-j-dim
	global $this-k-dim

	set_size [set $this-dims] \
	    [set $this-i-dim] \
	    [set $this-j-dim] \
	    [set $this-k-dim]
    }

    method set_index { axis iindex jindex kindex } {
	global $this-axis

	global $this-i-index
	global $this-j-index
	global $this-k-index

	global $this-i-index2
	global $this-j-index2
	global $this-k-index2

	set $this-axis $axis

	set $this-i-index $iindex
	set $this-j-index $jindex
	set $this-k-index $kindex

	set $this-i-index2 $iindex
	set $this-j-index2 $jindex
	set $this-k-index2 $kindex
    }

    method set_size { dims idim jdim kdim } {
	global $this-dims
	global $this-i-dim
	global $this-j-dim
	global $this-k-dim
	global $this-axis

	set $this-dims  $dims
	set $this-i-dim $idim
	set $this-j-dim $jdim
	set $this-k-dim $kdim

	if { [set $this-axis] >= [set $this-dims] } {
	    set $this-axis [expr [set $this-dims]-1]
	}

	set w .ui[modname]

	if {[winfo exists $w]} {

	    pack forget $w.main.i
	    pack forget $w.main.k
	    pack forget $w.main.j

	    if { [set $this-dims] == 3 } {
		pack $w.main.l $w.main.i $w.main.j $w.main.k -side top -padx 10 -pady 5
	    } elseif { [set $this-dims] == 2 } {
		pack $w.main.l $w.main.i $w.main.j -side top -padx 10 -pady 5	    
	    } elseif { [set $this-dims] == 1 } {
		pack $w.main.l $w.main.i -side top -padx 10 -pady 5	    
	    }
	}

	for {set i 0} {$i < 3} {incr i 1} {
	    if { $i == 0 } {
		set index i
	    } elseif { $i == 1 } {
		set index j
	    } elseif { $i == 2 } {
		set index k
	    }

	    global $this-$index-index
	    global $this-$index-index2

	    set stop_val [expr [set $this-$index-dim]-1]

	    if [ expr [winfo exists $w] ] {

		# Update the sliders to the new bounds.
		$w.main.$index.index.s configure -from 0 -to $stop_val

		bind $w.main.$index.index.e \
		    <KeyRelease> "$this manualSliderEntry 0 $stop_val $this-$index-index $this-$index-index2"
	    }

	    # Reset all of the slider values to the index values.
	    if { [set $this-$index-index] > $stop_val } {
		set $this-$index-index $stop_val
	    }

	    # Update the text values.
	    set $this-$index-index2 [set $this-$index-index]
	}
    }
}

