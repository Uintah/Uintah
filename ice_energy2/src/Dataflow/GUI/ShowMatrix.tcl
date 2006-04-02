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


itcl_class SCIRun_Visualization_ShowMatrix {
    inherit Module
    constructor {config} {
        set name ShowMatrix
	set_defaults
    }

    method set_defaults {} {
	initGlobal $this-xpos 0.0
	initGlobal $this-ypos 0.0
	initGlobal $this-xscale 1.0
	initGlobal $this-yscale 2.0
	initGlobal $this-col_begin 0
	initGlobal $this-row_begin 0
	initGlobal $this-cols 10000
	initGlobal $this-rows 10000
	initGlobal $this-3d_mode 1
	initGlobal $this-gmode 1
	initGlobal $this-colormapmode 0
	initGlobal $this-showtext 0
	trace variable $this-cols w "$this maxChanged"
	trace variable $this-rows w "$this maxChanged"
    }

    method maxChanged {args} {
	upvar \#0 $this-rows rows $this-cols cols
	set w .ui[modname].row.f.s
	if { [winfo exists $w] } {
	    $w configure -to $rows
	}

	set w .ui[modname].col.f.s
	if { [winfo exists $w] } {
	    $w configure -to $cols
	}
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w


	frame $w.length -relief groove -borderwidth 2
	frame $w.nlabs -borderwidth 2

	
	frame $w.mode -relief groove -borderwidth 2
	label $w.mode.label -text "Display Mode"
	radiobutton $w.mode.two -text "2D" -variable $this-3d_mode -value 0 -command "$this-c needexecute"
	radiobutton $w.mode.three -text "3D" -variable $this-3d_mode -value 1 -command "$this-c needexecute"
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

	foreach {dim title} {row Row col Column} {
	    set f $w.${dim}
	    frame $f -bd 2 -relief groove
	    label $f.label -text "$title Range"
	    pack $f.label -side top -expand 1 -fill x
	    set f $f.f
	    # Create range widget for slab mode
	    frame $f
	    # min range value label
	    entry $f.min -textvariable $this-${dim}_begin \
		-justify right -width 3 
	    bind $f.min <Return> "$this-c needexecute"
	    # MIP slab range widget
	    upvar \#0 $this-${dim}s max
	    range $f.s -from 0 -to 100 -orient horizontal -showvalue false \
		-rangecolor "#830101" -width 16 -command "$this-c needexecute"\
		-varmin $this-${dim}_begin -varmax $this-${dim}_end
	    # max range value label
	    entry $f.max -textvariable $this-${dim}_end -justify left -width 3
	    bind $f.max <Return> "$this-c needexecute"
	    pack $f.min -anchor w -side left -padx 0 -pady 0 -expand 0 
	    pack $f.max -anchor e -side right -padx 0 -pady 0 -expand 0 
	    pack $f.s -side left -anchor n -padx 0 -pady 0 -expand 1 -fill x
	    pack $f -side top -expand 1 -fill x
	}


	bind $w.pos <ButtonRelease> "$this-c needexecute"
	
	bind $w.pos.xslide.scale <ButtonRelease> "$this-c needexecute"
	bind $w.pos.yslide.scale <ButtonRelease> "$this-c needexecute"
	bind $w.pos.xslide.scale <B1-Motion> "$this-c needexecute"
	bind $w.pos.yslide.scale <B1-Motion> "$this-c needexecute"

	bind $w.pos.xscaleslide.scale <ButtonRelease> "$this-c needexecute"
#	bind $w.pos.yscaleslide.scale <ButtonRelease> "$this-c needexecute"
	bind $w.pos.xscaleslide.scale <B1-Motion> "$this-c needexecute"
#	bind $w.pos.yscaleslide.scale <B1-Motion> "$this-c needexecute"


#	bind $w.col.cto <Return> "$this-c needexecute"
#	bind $w.row.rfrom <Return> "$this-c needexecute"
#	bind $w.row.rto <Return> "$this-c needexecute"

	pack $f $w.row $w.col $w.graph $w.cmap $w.text $w.mode $w.poslabel $w.pos -side top -e y -f both -padx 5 -pady 5

	makeSciButtonPanel $w $w $this
	moveToCursor $w
    }
}


