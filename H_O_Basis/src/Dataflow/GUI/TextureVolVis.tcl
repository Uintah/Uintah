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



catch {rename TextureVolVis ""}

itcl_class SCIRun_Visualization_TextureVolVis {
    inherit Module
    constructor {config} {
	set name TextureVolVis
	set_defaults
    }
    method set_defaults {} {
	global $this-draw_mode
	set $this-draw_mode 0
	global $this-num_slices
	set $this-num_slices 64
	global $this-alpha_scale
	set $this-alpha_scale 0
	global $this-render_style
	set $this-render_style 0
	global $this-interp_mode 
	set $this-interp_mode 1
	global $this-contrast
	set $this-contrast 0.5
	global $this-contrastfp
	set $this-contrastfp 0.5
    }
    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    return
	}
	toplevel $w
	wm minsize $w 250 300
	frame $w.f -relief groove -borderwidth 2 
	pack $w.f -padx 2 -pady 2 -fill x
	set n "$this-c needexecute "

	global $this-render_style
	label $w.f.l -text "Rendering Style"
	radiobutton $w.f.modeo -text "Over Operator" -relief flat \
		-variable $this-render_style -value 0 \
		-anchor w -command $n

	radiobutton $w.f.modem -text "MIP" -relief flat \
		-variable $this-render_style -value 1 \
		-anchor w -command $n

	radiobutton $w.f.modea -text "Attenuate" -relief flat \
		-variable $this-render_style -value 2 \
		-anchor w -command $n

	pack $w.f.l $w.f.modeo $w.f.modem $w.f.modea -side top -fill x -padx 4 -pady 4

	frame $w.f2 -relief groove -borderwidth 2
	pack $w.f2 -padx 2 -pady 2 -fill x
	
	label $w.f2.l -text "View Mode"
	radiobutton $w.f2.full -text "Full Resolution" -relief flat \
		-variable $this-draw_mode -value 0 \
		-anchor w -command $n

	radiobutton $w.f2.los -text "Line of Sight" -relief flat \
		-variable $this-draw_mode -value 1 \
		-anchor w -command $n

	radiobutton $w.f2.roi -text "Region of Influence" -relief flat \
		-variable $this-draw_mode -value 2 \
		-anchor w -command $n

	pack $w.f2.l $w.f2.full $w.f2.los $w.f2.roi -side top -fill x -padx 4

	frame $w.f3 -relief groove -borderwidth 2
	pack $w.f3 -padx 2 -pady 2 -fill x

	label $w.f3.l -text "Interpolation Mode"
	radiobutton $w.f3.interp -text "Interpolate" -relief flat \
		-variable $this-interp_mode -value 1 \
		-anchor w -command $n

	radiobutton $w.f3.near -text "Nearest" -relief flat \
		-variable $this-interp_mode -value 0 \
		-anchor w -command $n

	pack $w.f3.l $w.f3.interp $w.f3.near -side top -fill x -padx 4
 
	global $this-num_slices
	scale $w.nslice -variable $this-num_slices \
		-from 64 -to 1024 -label "Number of Slices" \
		-showvalue true \
		-orient horizontal \

	global $this-alpha_scale
	
	scale $w.stransp -variable $this-alpha_scale \
		-from -1.0 -to 1.0 -label "Slice Transparency" \
		-showvalue true -resolution 0.001 \
		-orient horizontal 

	pack $w.stransp $w.nslice  -side top -fill x -padx 4 -pady 2

	bind $w.nslice <ButtonRelease> $n
	bind $w.stransp <ButtonRelease> $n
	
	makeSciButtonPanel $w $w $this
	moveToCursor $w
    }
}
