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



catch {rename EditTransferFunc2 ""}

itcl_class Volume_Visualization_EditTransferFunc2 {
    inherit Module
    constructor {config} {
	set name EditTransferFunc2
	set_defaults
    }

    method set_defaults {} {
	global $this-faux
	set $this-faux 0
	global $this-histo
	set $this-histo 0.5
    }

    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    create_gl
	    return
	}
	toplevel $w
	frame $w.f
	pack $w.f -padx 2 -pady 2
        create_gl

	iwidgets::scrolledframe $w.widgets -hscrollmode none -vscrollmode dynamic

	frame $w.title
	label $w.title.name -text "Widget Name" \
	    -width 16 -relief groove
	label $w.title.color -text "Color" -width 8 -relief groove
	label $w.title.opacity -text "Opacity" -width 8 -relief groove
	label $w.title.empty -text "" -width 3
	pack $w.title.name $w.title.color $w.title.opacity \
	    $w.title.empty \
	    -side left 

	frame $w.controls
	button $w.controls.addtriangle -text "Add Triangle" \
	    -command "$this-c addtriangle"
	button $w.controls.addrectangle -text "Add Rectangle" \
	    -command "$this-c addrectangle"
	button $w.controls.delete -text "Delete" \
	    -command "$this-c deletewidget"
	button $w.controls.undo -text "Undo" \
	    -command "$this-c undowidget"
	pack $w.controls.addtriangle $w.controls.addrectangle \
	    $w.controls.delete $w.controls.undo \
	    -padx 20 -pady 4 -fill x -expand yes -side left

	pack $w.title  -fill x -padx 2 -pady 2
	pack $w.widgets -side top -fill both -expand yes -padx 2
	pack $w.controls -fill x 

	makeSciButtonPanel $w $w $this "\"Reset\" \"$this-c reset_gui\" \"\""
	moveToCursor $w
    }

    method create_gl {} {
        set w .ui[modname]
        if {[winfo exists $w.f.gl]} {
            raise $w
        } else {
            set n "$this-c needexecute"
	    
            frame $w.f.gl -relief groove -borderwidth 2
            pack $w.f.gl -padx 2 -pady 2
            # create an OpenGL widget
            opengl $w.f.gl.gl -geometry 512x256 -doublebuffer true -direct true \
                -rgba true -redsize 1 -greensize 1 -bluesize 1 -depthsize 2
            # every time the OpenGL widget is displayed, redraw it
            bind $w.f.gl.gl <Expose> "$this-c expose"
            #bind $w.f.gl.gl <Configure> "$this-c resize"
            bind $w.f.gl.gl <ButtonPress> "$this-c mouse push %x %y %b"
            bind $w.f.gl.gl <ButtonRelease> "$this-c mouse release %x %y %b"
            bind $w.f.gl.gl <Motion> "$this-c mouse motion %x %y"
            bind $w.f.gl.gl <Destroy> "$this-c closewindow"
            # place the widget on the screen
            pack $w.f.gl.gl -fill both -expand 1
            # histogram opacity
            scale $w.f.gl.shisto -variable $this-histo \
                -from 0.0 -to 1.0 -label "Histogram Opacity" \
                -showvalue true -resolution 0.001 \
                -orient horizontal -command "$this-c redraw"
            pack $w.f.gl.shisto -side top -fill x -padx 4
            # faux shading
            frame $w.f.f0 -relief groove -borderwidth 2
            pack $w.f.f0 -padx 2 -pady 2 -fill x
            checkbutton $w.f.f0.faux -text "Opacity Modulation (Faux Shading)" \
                -relief flat -variable $this-faux -onvalue 1 -offvalue 0 \
                -anchor w -command "$n; $this-c redrawcmap"
            pack $w.f.f0.faux -side top -fill x -padx 4
        }
    }
}
