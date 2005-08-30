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



catch {rename ChangeCoordinates ""}

itcl_class SCIRun_FieldsGeometry_ChangeCoordinates {
    inherit Module
    
    constructor {config} {
	set name ChangeCoordinates
	set_defaults
    }
    
    method set_defaults {} {
	global $this-oldsystem
	global $this-newsystem
	set $this-oldsystem "Cartesian"
	set $this-newsystem "Spherical"
    }

    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	
	toplevel $w
	frame $w.f 
	pack $w.f -padx 2 -pady 2 -expand 1 -fill x
	set n "$this-c needexecute "

	frame $w.f.old -relief groove -borderwidth 2
	frame $w.f.new -relief groove -borderwidth 2
	
	label $w.f.old.l -text "Input coordinate system: "
	radiobutton $w.f.old.e -text "Cartesian   " -variable $this-oldsystem \
	    -value "Cartesian"
	radiobutton $w.f.old.s -text "Spherical   " -variable $this-oldsystem \
	    -value "Spherical"
	radiobutton $w.f.old.p -text "Polar   " -variable $this-oldsystem \
	    -value "Polar"
	radiobutton $w.f.old.r -text "Range" -variable $this-oldsystem \
	    -value "Range"
	pack $w.f.old.l $w.f.old.e $w.f.old.s $w.f.old.p $w.f.old.r -side top -anchor w

	label $w.f.new.l -text "Output coordinate system: "
	radiobutton $w.f.new.e -text "Cartesian   " -variable $this-newsystem \
	    -value "Cartesian"
	radiobutton $w.f.new.s -text "Spherical   " -variable $this-newsystem \
	    -value "Spherical"
	radiobutton $w.f.new.p -text "Polar   " -variable $this-newsystem \
	    -value "Polar"
	radiobutton $w.f.new.r -text "Range" -variable $this-newsystem \
	    -value "Range"
	pack $w.f.new.l $w.f.new.e $w.f.new.s $w.f.new.p $w.f.new.r -side top -anchor w

	pack $w.f.old $w.f.new -side left -pady 2 -padx 2

	makeSciButtonPanel $w $w $this
	moveToCursor $w
    }
}
