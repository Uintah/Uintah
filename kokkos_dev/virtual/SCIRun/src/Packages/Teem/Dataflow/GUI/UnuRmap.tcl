#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  
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

#    File   : UnuRmap.tcl
#    Author : Martin Cole
#    Date   : Mon Sep  8 09:46:23 2003

catch {rename Teem_UnuNtoZ_UnuRmap ""}

itcl_class Teem_UnuNtoZ_UnuRmap {
    inherit Module
    constructor {config} {
        set name UnuRmap
        set_defaults
    }
    method set_defaults {} {
        global $this-rescale
        set $this-rescale 0
	
	global $this-min
	set $this-min 0

	global $this-useinputmin
	set $this-useinputmin 1

	global $this-max
	set $this-max 0

	global $this-useinputmax
	set $this-useinputmax 1

        global $this-type
        set $this-type "nrrdTypeFloat"

	global $this-usetype
	set $this-usetype 1

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
	
        checkbutton $w.f.options.rescale \
	    -text "Rescale to the Map Domain" \
	    -variable $this-rescale
        pack $w.f.options.rescale -side top -anchor nw 

	frame $w.f.options.min -relief groove -borderwidth 2
	pack $w.f.options.min -side top -expand yes -fill x

        iwidgets::entryfield $w.f.options.min.v -labeltext "Min:" \
	    -textvariable $this-min
        pack $w.f.options.min.v -side top -expand yes -fill x

        checkbutton $w.f.options.min.useinputmin \
	    -text "Use lowest value of input nrrd as min" \
	    -variable $this-useinputmin
        pack $w.f.options.min.useinputmin -side top -anchor nw 

	frame $w.f.options.max -relief groove -borderwidth 2
	pack $w.f.options.max -side top -expand yes -fill x

        iwidgets::entryfield $w.f.options.max.v -labeltext "Max:" \
	    -textvariable $this-max
        pack $w.f.options.max.v -side top -anchor nw -expand yes -fill x

        checkbutton $w.f.options.max.useinputmax \
	    -text "Use highest value of input nrrd as max" \
	    -variable $this-useinputmax
        pack $w.f.options.max.useinputmax -side top -anchor nw 

	iwidgets::optionmenu $w.f.options.type -labeltext "Type:" \
	    -labelpos w -command "$this update_type $w.f.options.type"
	$w.f.options.type insert end nrrdTypeChar nrrdTypeUChar \
	    nrrdTypeShort nrrdTypeUShort nrrdTypeInt nrrdTypeUInt \
	    nrrdTypeLLong nrrdTupeULLong nrrdTypeFloat nrrdTypeDouble
	pack $w.f.options.type -side top -anchor nw -padx 3 -pady 3
	$w.f.options.type select [set $this-type]

	checkbutton $w.f.options.usetype \
	    -text "Use map's type as output type" \
	    -variable $this-usetype
	pack $w.f.options.usetype -side top -anchor nw -padx 3 -pady 3

	makeSciButtonPanel $w.f $w $this
	moveToCursor $w

	pack $w.f -expand 1 -fill x
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



