#
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

itcl_class SCIRun_FieldsOther_FieldInfo {
    inherit Module
    constructor {config} {
        set name FieldInfo
        set_defaults
    }

    method set_defaults {} {
	# the width of the first column of the data display
	global $this-firstwidth
	set $this-firstwidth 12

	# these won't be saved 
	global $this-fldname
	global $this-typename
	global $this-datamin
	global $this-datamax
	global $this-numnodes
	global $this-numelems
	global $this-dataat
        global $this-cx
        global $this-cy
        global $this-cz
        global $this-sizex
        global $this-sizey
        global $this-sizez
	set $this-fldname "---"
	set $this-typename "---"
	set $this-datamin "---"
	set $this-datamax "---"
	set $this-numnodes "---"
	set $this-numelems "---"
	set $this-dataat "---"
        set $this-cx "---"
        set $this-cy "---"
        set $this-cz "---"
        set $this-sizex "---"
        set $this-sizey "---"
        set $this-sizez "---"

    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }
        toplevel $w

#	wm minsize $w 416 408
#	wm maxsize $w 416 1000

	iwidgets::Labeledframe $w.att -labelpos nw \
		               -labeltext "Input Field Attributes" 
			       
	pack $w.att 
	set att [$w.att childsite]
	
	labelpair $att.l1 "Name" $this-fldname
	labelpair $att.l2 "Typename" $this-typename
        labelpair3 $att.l3 "Center (x,y,z)" $this-cx $this-cy $this-cz
        labelpair3 $att.l4 "Size (x,y,z)" $this-sizex $this-sizey $this-sizez
	labelpair2 $att.l5 "Data min,max" $this-datamin $this-datamax
	labelpair $att.l7 "# Nodes" $this-numnodes
	labelpair $att.l8 "# Elements" $this-numelems
	labelpair $att.l9 "Data at" $this-dataat
	pack $att.l1 $att.l2 $att.l3 $att.l4 $att.l5 \
	     $att.l7 $att.l8 $att.l9 -side top -expand y -fill x

	frame $w.exec
	pack $w.exec -side bottom -padx 5 -pady 5
	button $w.exec.execute -text "Execute" -command "$this-c needexecute"
	pack $w.exec.execute -side top -e n
    }

    method labelpair { win text1 text2 } {
	frame $win 
	pack $win -side top -padx 5
	label $win.l1 -text $text1 -width [set $this-firstwidth] \
		      -anchor w -just left
	label $win.colon  -text ":" -width 2 -anchor w -just left 
	label $win.l2 -textvar $text2 -width 40 -anchor w -just left \
		-fore darkred -borderwidth 0
	pack $win.l1 $win.colon $win.l2 -side left
    } 

    method labelpair2 { win text1 text2x text2y } {
	frame $win 
	pack $win -side top -padx 5
	label $win.l1 -text $text1 -width [set $this-firstwidth] \
		      -anchor w -just left
	label $win.colon  -text ": " -width 2 -anchor w -just left
	label $win.l2x -textvar $text2x -anchor w -just left \
	    -fore darkred -borderwidth 0
	label $win.comma1  -text ", " -anchor w -just left  \
	    -fore darkred -borderwidth 0
	label $win.l2y -textvar $text2y -anchor w -just left \
	    -fore darkred -borderwidth 0
	pack $win.l1 $win.colon \
	    $win.l2x $win.comma1 $win.l2y -side left -padx 0
    } 

    method labelpair3 { win text1 text2x text2y text2z } {
	frame $win 
	pack $win -side top -padx 5
	label $win.l1 -text $text1 -width [set $this-firstwidth] \
		      -anchor w -just left
	label $win.colon  -text ": " -width 2 -anchor w -just left
	label $win.l2x -textvar $text2x -anchor w -just left \
	    -fore darkred -borderwidth 0
	label $win.comma1  -text ", " -anchor w -just left  \
	    -fore darkred -borderwidth 0
	label $win.l2y -textvar $text2y -anchor w -just left \
	    -fore darkred -borderwidth 0
	label $win.comma2  -text ", " -anchor w -just left \
	    -fore darkred -borderwidth 0
	label $win.l2z -textvar $text2z -anchor w -just left \
	    -fore darkred -borderwidth 0
	pack $win.l1 $win.colon \
	    $win.l2x $win.comma1 $win.l2y $win.comma2 $win.l2z \
	    -side left -padx 0
    } 
}




