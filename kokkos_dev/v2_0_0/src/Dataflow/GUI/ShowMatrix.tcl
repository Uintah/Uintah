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

itcl_class SCIRun_Visualization_ShowMatrix {
    inherit Module
    constructor {config} {
        set name ShowMatrix

	global $this-xpos
	set $this-xpos 0.0

	global $this-ypos
	set $this-ypos 0.0

	global $this-xscale
	set $this-xscale 1.0

	global $this-yscale
	set $this-yscale 2.0

	global $this-col_begin
	set $this-col_begin 0
	
	global $this-col_end
#	set $this-col_end 1

	global $this-row_begin
	set $this-row_begin 0

	global $this-row_end
#	set $this-row_end 1
	
	global $this-displaymode
	set $this-displaymode 3D

	global $this-gmode
	set $this-gmode 1

	global $this-colormapmode
	set $this-colormapmode 0

	global $this-showtext
	set $this-showtext 0

#	global $this-swap
#	set $this-swap
    }

    method set_defaults {} {
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }
        toplevel $w


	frame $w.length -relief groove -borderwidth 2
	frame $w.nlabs -borderwidth 2

	
	frame $w.mode -relief groove -borderwidth 2
	label $w.mode.label -text "Display Mode"
	radiobutton $w.mode.two -text "2D" -variable $this-displaymode -value 2D -command "$this-c needexecute"
	radiobutton $w.mode.three -text "3D" -variable $this-displaymode -value 3D -command "$this-c needexecute"
	pack $w.mode.label -side top -expand yes -fill both
	pack $w.mode.two $w.mode.three -side top -anchor w

	frame $w.text -relief groove -borderwidth 2
	checkbutton $w.text.text -text "Show Text" -variable $this-showtext -command "$this-c needexecute"
	set $w.text.text $this-showtext
	pack $w.text.text -side top -anchor w


	frame $w.graph -relief groove -borderwidth 2
	label $w.graph.label -text "Graph Mode"
	radiobutton $w.graph.1 -text "Line" -variable $this-gmode -value 1 -command "$this-c needexecute"
	radiobutton $w.graph.2 -text "Bar" -variable $this-gmode -value 2 -command "$this-c needexecute"
	radiobutton $w.graph.3 -text "Sheet" -variable $this-gmode -value 3 -command "$this-c needexecute"
	radiobutton $w.graph.4 -text "Ribbon" -variable $this-gmode -value 4 -command "$this-c needexecute"
	radiobutton $w.graph.5 -text "Filled Ribbon" -variable $this-gmode -value 5 -command "$this-c needexecute"

	pack $w.graph.label -side top -expand yes -fill both
	pack $w.graph.1 $w.graph.2 $w.graph.3 $w.graph.4 $w.graph.5 -side top -anchor w


	frame $w.cmap -relief groove -borderwidth 2
	label $w.cmap.label -text "Color Mode"
	radiobutton $w.cmap.0 -text "Color By Value" -variable $this-colormapmode -value 0 -command "$this-c needexecute"
	radiobutton $w.cmap.1 -text "Color By Row" -variable $this-colormapmode -value 1 -command "$this-c needexecute"
	radiobutton $w.cmap.2 -text "Color By Column" -variable $this-colormapmode -value 2 -command "$this-c needexecute"
	pack $w.cmap.label -side top -expand yes -fill both
	pack $w.cmap.0 $w.cmap.1 $w.cmap.2 -side top -anchor w



	frame $w.pos
	label $w.poslabel -text "2D Positioning"
	expscale $w.pos.xslide -orient horizontal -label "X Tanslate" -variable $this-xpos
#	set $w.pos.xslide.scale $this-xpos
	expscale $w.pos.yslide -orient horizontal -label "Y Tanslate" -variable $this-ypos
#	set $w.pos.yslide.scale $this-ypos

	expscale $w.pos.xscaleslide -orient horizontal -label "Scale" -variable $this-xscale
#	set $w.pos.xscaleslide.scale $this-xscale
#	expscale $w.pos.yscaleslide -orient horizontal -label "Y Scale" -variable $this-yscale
#	set $w.pos.yscaleslide.scale $this-yscale
	

	pack $w.pos.xslide $w.pos.yslide $w.pos.xscaleslide -side top -expand yes -fill x
#	-from -1 -to 1 -showvalue true -variable -resolution 0.01 -tickinterval 0.25
#	scale $w.pos.yslide -orient horizontal -label "Y Translate" -from -1 -to 1 
#	-showvalue true -variable $this-ypos -resolution 0.01 -tickinterval 0.25





	frame $w.col -relief groove -borderwidth 2
	frame $w.col.label
	
	label $w.col.label.from -text "Column Begin:"
	label $w.col.label.to -text "Column End:"
	pack $w.col.label.from $w.col.label.to -side top

	frame $w.col.entry
	entry $w.col.entry.from -textvariable $this-col_begin -width 4
	set $w.col.entry.from $this-col_begin

	entry $w.col.entry.to -textvariable $this-col_end -width 4
#	set $w.col.entry.to $this-col_end
	pack $w.col.entry.from $w.col.entry.to -side top
	pack $w.col.label $w.col.entry -side left -expand yes -fill x


	frame $w.row -relief groove -borderwidth 2
	frame $w.row.label
	
	label $w.row.label.from -text "Row Begin:"
	label $w.row.label.to -text "Row End:"
	pack $w.row.label.from $w.row.label.to -side top

	frame $w.row.entry
	entry $w.row.entry.from -textvariable $this-row_begin -width 4
	set $w.row.entry.from $this-row_begin

	entry $w.row.entry.to -textvariable $this-row_end -width 4
#	set $w.row.entry.to $this-row_end
	pack $w.row.entry.from $w.row.entry.to -side top
	pack $w.row.label $w.row.entry -side left -expand yes -fill x


	bind $w.pos <ButtonRelease> "$this-c needexecute"
	
	bind $w.pos.xslide.scale <ButtonRelease> "$this-c needexecute"
	bind $w.pos.yslide.scale <ButtonRelease> "$this-c needexecute"
	bind $w.pos.xslide.scale <B1-Motion> "$this-c needexecute"
	bind $w.pos.yslide.scale <B1-Motion> "$this-c needexecute"

	bind $w.pos.xscaleslide.scale <ButtonRelease> "$this-c needexecute"
#	bind $w.pos.yscaleslide.scale <ButtonRelease> "$this-c needexecute"
	bind $w.pos.xscaleslide.scale <B1-Motion> "$this-c needexecute"
#	bind $w.pos.yscaleslide.scale <B1-Motion> "$this-c needexecute"

	bind $w.col.entry.from <Return> "$this-c needexecute"
	bind $w.col.entry.to <Return> "$this-c needexecute"
	bind $w.row.entry.from <Return> "$this-c needexecute"
	bind $w.row.entry.to <Return> "$this-c needexecute"

#	bind $w.col.cto <Return> "$this-c needexecute"
#	bind $w.row.rfrom <Return> "$this-c needexecute"
#	bind $w.row.rto <Return> "$this-c needexecute"

	button $w.execbutton -text "Execute" -command "$this-c needexecute"

	pack $w.row $w.col $w.graph $w.cmap $w.text $w.mode $w.poslabel $w.pos $w.execbutton -side top -e y -f both -padx 5 -pady 5
    }
}


