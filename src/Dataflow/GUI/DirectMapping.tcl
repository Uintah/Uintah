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


itcl_class SCIRun_FieldsData_DirectMapping {
    inherit Module
    constructor {config} {
        set name DirectMapping
        set_defaults
    }

    method set_defaults {} {
	global $this-interpolation_basis
	global $this-map_source_to_single_dest
	global $this-exhaustive_search
	global $this-exhaustive_search_max_dist
	global $this-np
	set $this-interpolation_basis linear
	set $this-map_source_to_single_dest 0
	set $this-exhaustive_search 0
	set $this-exhaustive_search_max_dist -1
	set $this-np 1
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w

	frame $w.basis
	label $w.basis.label -text "Interpolation Basis:"
	radiobutton $w.basis.const -text "Constant ('find closest')" \
		-variable $this-interpolation_basis -value constant
	frame $w.basis.cframe 
	label $w.basis.cframe.label -text "Constant Mapping:"
	radiobutton $w.basis.cframe.onetomany -text \
		"Each destination gets nearest source value" \
		-variable $this-map_source_to_single_dest -value 0
	radiobutton $w.basis.cframe.onetoone -text \
		"Each source projects to just one destination" \
		-variable $this-map_source_to_single_dest -value 1
	pack $w.basis.cframe.label -side top -anchor w -padx 4 -pady 4
	pack $w.basis.cframe.onetomany $w.basis.cframe.onetoone \
		-side top -anchor w -padx 15
	radiobutton $w.basis.lin -text "Linear (`weighted')" \
		-variable $this-interpolation_basis -value linear
	pack $w.basis.label -side top -anchor w
	pack $w.basis.const -padx 15 -side top -anchor w
	pack $w.basis.cframe -padx 30 -side top -anchor w
	pack $w.basis.lin -padx 15 -side top -anchor w
	
	frame $w.exhaustive
	label $w.exhaustive.label -text "Exhaustive Search Options:"
	checkbutton $w.exhaustive.check \
	    -text "Use Exhaustive Search if Fast Search Fails" \
	    -variable $this-exhaustive_search
	frame $w.exhaustive.dist
	label $w.exhaustive.dist.label -text \
		"Maximum Distance (negative value -> 'no max'):"
	entry $w.exhaustive.dist.entry -textvariable \
	    $this-exhaustive_search_max_dist -width 8
	pack $w.exhaustive.dist.label $w.exhaustive.dist.entry \
	    -side left -anchor n
	pack $w.exhaustive.label -side top -anchor w
	pack $w.exhaustive.check -side top -anchor w -padx 15
	pack $w.exhaustive.dist -side top -anchor w -padx 30

	scale $w.scale -orient horizontal -variable $this-np -from 1 -to 32 \
		-showvalue true -label "Number of Threads"
	
	pack $w.basis -side top -anchor w
	pack $w.exhaustive -side top -anchor w -pady 15
	pack $w.scale -side top -expand 1 -fill x -padx 4 -pady 4

	makeSciButtonPanel $w $w $this
	moveToCursor $w
    }
}
