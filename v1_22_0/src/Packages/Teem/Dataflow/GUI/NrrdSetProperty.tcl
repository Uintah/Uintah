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
 #  NrrdSetProperty.tcl: The NrrdSetProperty UI
 #  Written by:
 #   David Weinstein
 #   Department of Computer Science
 #   University of Utah
 #   Jan 2000
 #  Copyright (C) 2000 SCI Group
 ##

catch {rename Teem_NrrdData_NrrdSetProperty ""}

itcl_class Teem_NrrdData_NrrdSetProperty {
    inherit Module
    constructor {config} {
        set name NrrdSetProperty
        set_defaults
    }

    method set_defaults {} {
        global $this-num-entries
        set $this-num-entries 0

	global $this-check
	global $this-property
	global $this-type
	global $this-value
	global $this-readonly

	set $this-check 0
	set $this-property ""
	set $this-type "unknown"
	set $this-value ""
	set $this-readonly 0
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }

        toplevel $w


	iwidgets::scrolledframe $w.entries -hscrollmode none

	frame $w.title
	label $w.title.check    -text ""         -width  3 -relief groove
	label $w.title.property -text "Property" -width 16 -relief groove
	label $w.title.type     -text "Type"     -width 15 -relief groove
	label $w.title.value    -text "Value"    -width 16 -relief groove
	label $w.title.empty    -text ""         -width  3 -relief groove

	pack $w.title.check $w.title.property $w.title.type \
	    $w.title.value $w.title.empty \
	    -side left 

	pack $w.title  -fill x
	pack $w.entries -side top -fill both -expand yes

	create_entries

	frame $w.controls
	button $w.controls.add -text "Add Entry" \
	    -command "$this addEntry"
	button $w.controls.delete -text "Delete Entry" \
	    -command "$this deleteEntry"
	pack $w.controls.add $w.controls.delete \
	    -side left -fill x -expand y

	pack $w.controls -side top -fill both -expand yes -pady 10

	makeSciButtonPanel $w $w $this
	moveToCursor $w	
     }

    method create_entries {} {
	set w .ui[modname]
	if {[winfo exists $w]} {

	    set entries [$w.entries childsite]

	    # Create the new variables and entries if needed.
	    for {set i 0} {$i < [set $this-num-entries]} {incr i} {
		
		if { [catch { set t [set $this-check-$i] } ] } {
		    set $this-check-$i [set $this-check]
		}
		if { [catch { set t [set $this-property-$i] } ] } {
		    set $this-property-$i [set $this-property]
		}
		if { [catch { set t [set $this-type-$i]}] } {
		    set $this-type-$i [set $this-type]
		}
		if { [catch { set t [set $this-value-$i]}] } {
		    set $this-value-$i [set $this-value]
		}
		if { [catch { set t [set $this-readonly-$i]}] } {
		    set $this-readonly-$i [set $this-readonly]
		}

		if {![winfo exists $entries.e-$i]} {

		    if { [set $this-readonly-$i] == 1 } {
			set state "disabled"
		    } else {
			set state "normal"
		    }

		    frame $entries.e-$i
		    checkbutton $entries.e-$i.check -variable $this-check-$i \
			-state $state
		    entry $entries.e-$i.property \
			-textvariable $this-property-$i -width 16 \
			-state $state
		    labelcombo $entries.e-$i.type \
			{unknown int float double string other} \
			$this-type-$i \
			$state
		    entry $entries.e-$i.value \
			-textvariable $this-value-$i    -width 16 \
			-state $state

		    pack $entries.e-$i.check \
			$entries.e-$i.property \
			$entries.e-$i.type \
			$entries.e-$i.value \
			-side left
		    pack $entries.e-$i 
		}
	    }

	    # Destroy all the left over entries from prior runs.
	    while {[winfo exists $entries.e-$i]} {
		destroy $entries.e-$i
		incr i
	    }
	}
    }

    method labelcombo { win arglist var state} {
	frame $win 
	iwidgets::optionmenu $win.c -foreground darkred \
	    -command "$this comboget $win.c $var" \
	    -state $state

	foreach elem $arglist {
	    $win.c insert end $elem
	}

	# hack to associate optionmenus with a textvariable
	bind $win.c <Map> "$win.c select {[set $var]}"

	pack $win.c -side left	
    }

    method comboget { win var } {
	if {[winfo exists $win]} {
	    set $var [$win get]
	}
    }

    method addEntry {} {
	global $this-num-entries

# Save the defaults for the next new entry.
	if { [set $this-num-entries] > 0 } {
	    set i [expr [set $this-num-entries] - 1]
	}

	set i [set $this-num-entries]

# Add in the new entry using the defaults.

	set $this-check-$i 0
	set $this-property-$i [set $this-property]
	set $this-type-$i     [set $this-type]
	set $this-value-$i    [set $this-value]

	incr $this-num-entries

	create_entries
    }

    method deleteEntry {} {
	global $this-num-entries

	set j 0

	for {set i 0} {$i < [set $this-num-entries]} {incr i} {

# Shift the enties in the list. 
	    if { [set $this-check-$i] == 0 } {
		set $this-check-$j 0
		set $this-property-$j [set $this-property-$i]
		set $this-type-$j     [set $this-type-$i]
		set $this-value-$j    [set $this-value-$i]
		incr j
	    }
	}

	set $this-num-entries $j
	
	create_entries
    }

    method setEntry { property type value readonly } {

	for {set i 0} {$i < [set $this-num-entries]} {incr i} {
	    if { $property == [set $this-property-$i] } {
		break;
	    }
	}

	if { $i == [set $this-num-entries] } {

	    set $this-check-$i 0
	    set $this-property-$i $property
	    set $this-type-$i     $type
	    set $this-value-$i    $value
	    set $this-readonly-$i $readonly
	    
	    set w .ui[modname]
	    if {[winfo exists $w]} {
		    
		set entries [$w.entries childsite]
		if {[winfo exists $entries.e-$i.type.c]} {
		    $entries.e-$i.type.c select $type
		}
	    }

	    incr $this-num-entries
	}

	create_entries
    }
}
