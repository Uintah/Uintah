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

#    File   : Unu2op.tcl
#    Author : Martin Cole
#    Date   : Mon Sep  8 09:46:23 2003

catch {rename Teem_UnuAtoM_Unu2op ""}

itcl_class Teem_UnuAtoM_Unu2op {
    inherit Module
    constructor {config} {
        set name Unu2op
        set_defaults
    }
    method set_defaults {} {
        global $this-operator
        set $this-operator "eq"

	global $this-float_input
	set $this-float_input {1.0}

	global $this-type
	set $this-type nrrdTypeFloat

	global $this-usetype
	set $this-usetype 1

	trace variable $this-operator w "$this set_operator" 
	trace variable $this-type w "$this set_type" 

    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }

        toplevel $w

        frame $w.f
	pack $w.f -padx 2 -pady 2 -side top -expand yes
	
	frame $w.f.options
	pack $w.f.options -side top -expand yes

	iwidgets::optionmenu $w.f.options.operator -labeltext "Operator:" \
	    -labelpos w -command "$this update_operator $w.f.options.operator"
	$w.f.options.operator insert end "+" "-" "x" "/" "^" "%" \
	    "fmod" "atan2" "min" "max" "lt" "lte" "gt" "gte" "eq" "neq" "comp"\
	    "exists"
	pack $w.f.options.operator -side top -anchor nw -padx 3 -pady 3
	$w.f.options.operator select [set $this-operator]
	
	iwidgets::entryfield $w.f.options.float_input -labeltext "Float Input:" -textvariable $this-float_input
        pack $w.f.options.float_input -side top -expand yes -fill x

	iwidgets::optionmenu $w.f.options.type -labeltext "Type:" \
	    -labelpos w -command "$this update_type $w.f.options.type"
	$w.f.options.type insert end nrrdTypeChar nrrdTypeUChar nrrdTypeShort nrrdTypeUShort nrrdTypeInt \
	    nrrdTypeUInt nrrdTypeLLong nrrdTupeULLong nrrdTypeFloat nrrdTypeDouble
	pack $w.f.options.type -side top -anchor nw -padx 3 -pady 3
	$w.f.options.type select [set $this-type]

	checkbutton $w.f.options.usetype -text "Same as input type" \
	    -variable $this-usetype
	pack $w.f.options.usetype -side top -anchor nw -padx 3 -pady 3 

	makeSciButtonPanel $w.f $w $this
	moveToCursor $w

	pack $w.f -expand 1 -fill x
    }

    method update_operator {menu} {
	global $this-operator
	set which [$menu get]
	set $this-operator $which
    }

    method set_operator { name1 name2 op } {
	set w .ui[modname]
	set menu $w.f.options.operator
	if {[winfo exists $menu]} {
	    $menu select [set $this-operator]
	}
    }

    method update_type {menu} {
	global $this-type
	set which [$menu get]
	set $this-type $which
    }

    method set_type { name1 name2 op } {
	set w .ui[modname]
	set menu $w.f.options.type
	if {[winfo exists $menu]} {
	    $menu select [set $this-type]
	}
    }
}
