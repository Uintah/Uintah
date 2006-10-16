#
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


itcl_class SCIRun_FieldsCreate_SeedPoints {
    inherit Module

    constructor {config} {
        set name SeedPoints	
    }

    method send {} {
	set $this-send 1
	$this-c needexecute
    }

    method make_seed {i} {
	set w .ui[modname]
        if {[winfo exists $w]} {
  
	    if {[winfo exists $w.f.t]} {
		destroy $w.f.t
	    }
	    if {! [winfo exists $w.f.f.a$i]} {

		global $this-seedX$i
		global $this-seedY$i
		global $this-seedZ$i

		set $this-seedX$i "0"
		set $this-seedY$i "0"
		set $this-seedZ$i "0"
	    }
	}
    }

    method set_seed {i x y z} {
	set $this-seedX$i $x
	set $this-seedY$i $y
	set $this-seedZ$i $z
    }

    method clear_seed {i} {
	set w .ui[modname]
        if {[winfo exists $w]} {
	    
	    if {[winfo exists $w.f.t]} {
		destroy $w.f.t
	    }
	    if {[winfo exists $w.f.f.a$i]} {
		destroy $w.f.f.a$i
	    }
	    unset $this-seedX$i
	    unset $this-seedY$i
	    unset $this-seedZ$i
	}
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }

        toplevel $w

	build_ui $w

	makeSciButtonPanel $w $w $this "\"Send Seeds\" \"$this send\" \"\""

	moveToCursor $w
    }

    method build_ui { w } {

	frame $w.numSeeds
        label $w.numSeeds.label -text "Number of Seeds"
        entry $w.numSeeds.entry \
            -textvariable $this-num_seeds
        pack $w.numSeeds.label $w.numSeeds.entry -side left
        pack $w.numSeeds

	frame $w.f
	frame $w.f.g
	scale $w.f.slide -orient horizontal -label "Seed Size" -from 0 -to 100 -showvalue true \
	     -variable $this-probe_scale -resolution 0.25 -tickinterval 25
	set $w.f.slide $this-probe_scale

	bind $w.f.slide <ButtonRelease> "$this-c needexecute"
	bind $w.f.slide <B1-Motion> "$this-c needexecute"

	pack $w.f.slide $w.f.g -side bottom -expand yes -fill x
	pack $w.f -side top -expand yes -fill both -padx 5 -pady 5

	checkbutton $w.ex -text "Auto execute" \
	    -variable $this-auto_execute
	pack $w.ex -side top -anchor n
	
    }
}




