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


catch {rename VolumeVisualizer ""}

itcl_class SCIRun_Visualization_VolumeVisualizer {
    inherit Module
    constructor {config} {
	set name VolumeVisualizer
	set_defaults
    }
    method set_defaults {} {
	global $this-use_stencil
	set $this-use_stencil 0
	global $this-invert_opacity
	set $this-invert_opacity 0
	global $this-multi_level
	set $this-multi_level 1
        global $this-blend_res
        set $this-blend_res 8
	global $this-sampling_rate_lo
	set $this-sampling_rate_lo 1.0
	global $this-sampling_rate_hi
	set $this-sampling_rate_hi 4.0
        global $this-adaptive
        set $this-adaptive 1
        global $this-cmap_size
        set $this-cmap_size 7
        global $this-sw_raster
        set $this-sw_raster 0
	global $this-alpha_scale
	set $this-alpha_scale 0
	global $this-render_style
	set $this-render_style 0
	global $this-interp_mode 
	set $this-interp_mode 1
        global $this-shading
        set $this-shading 0
        global $this-ambient
        set $this-ambient 0.5
        global $this-diffuse
        set $this-diffuse 0.5
        global $this-specular
        set $this-specular 0.0
        global $this-shine
        set $this-shine 30.0
        global $this-light
        set $this-light 0

	# For backwards compatability
	global $this-contrast
	global $this-contrastfp
	global $this-draw_mode
	global $this-num_slices
	set $this-num_slices -1
    }

#      method ui {} {
#  	set w .ui[modname]
#  	if {[winfo exists $w]} {
#  	    return
#  	}
#  	toplevel $w
#  	build_ui

#      }
    

    method ui {} { 
        set w .ui[modname] 

        if {[winfo exists $w]} {
            return
        } else {
	    buildTopLevel
	}
    }

    method buildTopLevel {} {
        set w .ui[modname] 

        if {[winfo exists $w]} { 
            return
        } 
	
        toplevel $w 
	wm withdraw $w

	build_ui
    }

    method build_ui {} {
	set w .ui[modname]
	wm minsize $w 250 300
	frame $w.main -relief flat
	pack $w.main -fill both -expand yes
	frame $w.main.f -relief groove -borderwidth 2 
	pack $w.main.f -padx 2 -pady 2 -fill x
	set n "$this-c needexecute"
        set s "$this state"

	global $this-render_style
	label $w.main.f.l -text "Blending Mode"
	pack $w.main.f.l -side top -fill x

        frame $w.main.f.fmode
        pack $w.main.f.fmode -padx 2 -pady 2 -fill x
	radiobutton $w.main.f.fmode.modeo -text "Over Operator" -relief flat \
		-variable $this-render_style -value 0 \
		-anchor w -command $n
	radiobutton $w.main.f.fmode.modem -text "MIP" -relief flat \
		-variable $this-render_style -value 1 \
		-anchor w -command $n
	pack $w.main.f.fmode.modeo $w.main.f.fmode.modem \
            -side left -fill x -padx 4 -pady 4
	pack $w.main.f.fmode.modeo $w.main.f.fmode.modem \
            -side left -fill x -padx 10 -pady 4 -expand y

        frame $w.main.f.fres
        pack $w.main.f.fres -padx 2 -pady 2 -fill x
        label $w.main.f.fres.res -text "Resolution (bits)"
	radiobutton $w.main.f.fres.b0 -text 8 -variable $this-blend_res -value 8 \
	    -command $n
	radiobutton $w.main.f.fres.b1 -text 16 -variable $this-blend_res -value 16 \
	    -command $n
	radiobutton $w.main.f.fres.b2 -text 32 -variable $this-blend_res -value 32 \
	    -command $n
	pack $w.main.f.fres.res $w.main.f.fres.b0 $w.main.f.fres.b1 $w.main.f.fres.b2 \
            -side left -fill x -padx 4 -pady 4

	frame $w.main.f3 -relief flat -borderwidth 0
	pack $w.main.f3 -fill x
	if { [set $this-multi_level] > 1 } {
	    $this build_multi_level
	}
 
        #-----------------------------------------------------------
        # Shading
        #-----------------------------------------------------------
	frame $w.main.f4 -relief groove -borderwidth 2
	pack $w.main.f4 -padx 2 -pady 2 -fill x
	checkbutton $w.main.f4.shading -text "Shading" -relief flat \
            -variable $this-shading -onvalue 1 -offvalue 0 \
            -anchor w -command "$s; $n"
        pack $w.main.f4.shading -side top -fill x -padx 4

        #-----------------------------------------------------------
        # Light
        #-----------------------------------------------------------
 	frame $w.main.f5
 	pack $w.main.f5 -padx 2 -pady 2 -fill x
 	label $w.main.f5.light -text "Attach Light to"
 	radiobutton $w.main.f5.light0 -text "Light 0" -relief flat \
            -variable $this-light -value 0 \
            -anchor w -command $n
 	radiobutton $w.main.f5.light1 -text "Light 1" -relief flat \
            -variable $this-light -value 1 \
            -anchor w -command $n
#  	radiobutton $w.f5.light2 -text "Light 2" -relief flat \
#             -variable $this-light -value 2 \
#             -anchor w -command $n
#  	radiobutton $w.f5.light3 -text "Light 3" -relief flat \
#             -variable $this-light -value 3 \
#             -anchor w -command $n
        pack $w.main.f5.light $w.main.f5.light0 $w.main.f5.light1 \
            -side left -fill x -padx 4

        #-----------------------------------------------------------
        # Material
        #-----------------------------------------------------------
	frame $w.main.f6 -relief groove -borderwidth 2
	pack $w.main.f6 -padx 2 -pady 2 -fill x
 	label $w.main.f6.material -text "Material"
	global $this-ambient
	scale $w.main.f6.ambient -variable $this-ambient \
            -from 0.0 -to 1.0 -label "Ambient" \
            -showvalue true -resolution 0.001 \
            -orient horizontal
	global $this-diffuse
	scale $w.main.f6.diffuse -variable $this-diffuse \
		-from 0.0 -to 1.0 -label "Diffuse" \
		-showvalue true -resolution 0.001 \
		-orient horizontal
	global $this-specular
	scale $w.main.f6.specular -variable $this-specular \
		-from 0.0 -to 1.0 -label "Specular" \
		-showvalue true -resolution 0.001 \
		-orient horizontal
	global $this-shine
	scale $w.main.f6.shine -variable $this-shine \
		-from 1.0 -to 128.0 -label "Shine" \
		-showvalue true -resolution 1.0 \
		-orient horizontal
        pack $w.main.f6.material $w.main.f6.ambient $w.main.f6.diffuse \
            $w.main.f6.specular $w.main.f6.shine \
            -side top -fill x -padx 4


 	frame $w.f3 -relief groove -borderwidth 2
 	pack $w.f3 -padx 2 -pady 2 -fill x
 	label $w.f3.l -text "Interpolation Mode"
	frame $w.f3.f
 	radiobutton $w.f3.f.interp -text "Trilinear" -relief flat \
 		-variable $this-interp_mode -value 1 \
 		-anchor w -command $n
 	radiobutton $w.f3.f.near -text "Nearest" -relief flat \
 		-variable $this-interp_mode -value 0 \
 		-anchor w -command $n
	pack $w.f3.f.interp $w.f3.f.near -side left -fill x -padx 10 -expand y
 	pack $w.f3.l $w.f3.f -side top -fill x -padx 4

        #-----------------------------------------------------------
        # Sampling
        #-----------------------------------------------------------
        frame $w.main.sampling -relief groove -borderwidth 2
        pack $w.main.sampling -padx 2 -pady 2 -fill x
        label $w.main.sampling.l -text "Sampling"

	scale $w.main.sampling.srate_hi -variable $this-sampling_rate_hi \
            -from 0.5 -to 10.0 -label "Sampling Rate" \
            -showvalue true -resolution 0.1 \
            -orient horizontal \

	checkbutton $w.main.sampling.adaptive -text "Adaptive Sampling" -relief flat \
            -variable $this-adaptive -onvalue 1 -offvalue 0 \
            -anchor w -command "$s; $n"

	scale $w.main.sampling.srate_lo -variable $this-sampling_rate_lo \
            -from 0.1 -to 5.0 -label "Interactive Sampling Rate" \
            -showvalue true -resolution 0.1 \
            -orient horizontal \

	pack $w.main.sampling.l $w.main.sampling.srate_hi $w.main.sampling.adaptive \
            $w.main.sampling.srate_lo -side top -fill x -padx 4 -pady 2
        
        #-----------------------------------------------------------
        # Transfer Function
        #-----------------------------------------------------------
        frame $w.main.tf -relief groove -borderwidth 2
        pack $w.main.tf -padx 2 -pady 2 -fill x
        label $w.main.tf.l -text "Transfer Function"

	scale $w.main.tf.stransp -variable $this-alpha_scale \
		-from -1.0 -to 1.0 -label "Global Opacity" \
		-showvalue true -resolution 0.001 \
		-orient horizontal 

# 	scale $w.tf.cmap_size -variable $this-cmap_size \
# 		-from 4 -to 10 -label "Table Size (2^n)" \
# 		-showvalue true -resolution 1 \
# 		-orient horizontal \

	checkbutton $w.main.tf.sw -text "Software Rasterization" -relief flat \
            -variable $this-sw_raster -onvalue 1 -offvalue 0 \
            -anchor w -command "$n"

	pack $w.main.tf.l $w.main.tf.stransp $w.main.tf.sw \
            -side top -fill x -padx 4 -pady 2

        bind $w.main.f6.ambient <ButtonRelease> $n
        bind $w.main.f6.diffuse <ButtonRelease> $n
        bind $w.main.f6.specular <ButtonRelease> $n
        bind $w.main.f6.shine <ButtonRelease> $n

	bind $w.main.sampling.srate_hi <ButtonRelease> $n
	bind $w.main.sampling.srate_lo <ButtonRelease> $n

#	bind $w.main.tf.cmap_size <ButtonRelease> $n
	bind $w.main.tf.stransp <ButtonRelease> $n

	makeSciButtonPanel $w.main $w $this
        $this state
	moveToCursor $w

    }

    method state {} {
	set w .ui[modname]
	if {[set $this-shading] == 1} {
            $this activate $w.main.f6.ambient
            $this activate $w.main.f6.diffuse
            $this activate $w.main.f6.specular
            $this activate $w.main.f6.shine
            $this activate $w.main.f5.light
            $this activate $w.main.f5.light0
            $this activate $w.main.f5.light1
#             $this activate $w.f5.light2
#             $this activate $w.f5.light3
	} else {
            $this deactivate $w.main.f6.ambient
            $this deactivate $w.main.f6.diffuse
            $this deactivate $w.main.f6.specular
            $this deactivate $w.main.f6.shine
            $this deactivate $w.main.f5.light
            $this deactivate $w.main.f5.light0
            $this deactivate $w.main.f5.light1
#             $this deactivate $w.f5.light2
#             $this deactivate $w.f5.light3
	}
	if {[set $this-adaptive] == 1} {
            $this activate $w.main.sampling.srate_lo
        } else {
            $this deactivate $w.main.sampling.srate_lo
        }
    }
    method activate { w } {
	if {[winfo exists $w]} {
	    $w configure -state normal -foreground black
	}
    }
    method deactivate { w } {
	if {[winfo exists $w]} {
	    $w configure -state disabled -foreground darkgrey
	}
    }

    method build_multi_level { } {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    puts -nonewline "building ml frame"
	    frame $w.main.f3.f -relief groove -borderwidth 2
	    pack $w.main.f3.f -padx 2 -pady 2 -fill x -expand yes	
	    frame $w.main.f3.f.f1 -relief flat -borderwidth 2
	    pack $w.main.f3.f.f1 -padx 2 -pady 2 -fill x -expand yes
	    checkbutton $w.main.f3.f.f1.stencil -text "Use Stencil" \
		-variable $this-use_stencil -command "$this-c needexecute"
	    checkbutton $w.main.f3.f.f1.opacity -text "Highlight Levels" \
		-variable $this-invert_opacity -command "$this-c needexecute"
	    pack $w.main.f3.f.f1.stencil $w.main.f3.f.f1.opacity -side left
	    
	    frame $w.main.f3.f.f2 -relief flat -borderwidth 2
	    pack $w.main.f3.f.f2 -padx 2 -pady 2 -fill x -expand yes
	    label $w.main.f3.f.f2.l -text "Show level"
	    pack $w.main.f3.f.f2.l -side left
	    frame $w.main.f3.f.f2.f -relief flat -borderwidth 2
	    pack $w.main.f3.f.f2.f -side right
	    set selected 0
	    for { set i 0 } { $i < [set $this-multi_level] } { incr i } {
		frame $w.main.f3.f.f2.f.f$i -relief flat
		pack $w.main.f3.f.f2.f.f$i -fill x -expand yes -side top
		checkbutton $w.main.f3.f.f2.f.f$i.b -text $i  \
		-variable $this-l$i -command "$this-c needexecute" 
		scale $w.main.f3.f.f2.f.f$i.s -variable $this-s$i \
		    -from -1.0 -to 1.0 -orient horizontal -resolution 0.01

		pack $w.main.f3.f.f2.f.f$i.b $w.main.f3.f.f2.f.f$i.s -side left
		if { [isOn l$i] } {
		    set selected 1
		}
		bind $w.main.f3.f.f2.f.f$i.s <ButtonRelease> "$this-c needexecute" 
	    }
	    if { !$selected && [winfo exists $w.main.f3.f.f2.f.f0.b] } {  
		$w.main.f3.f.f2.f.f0.b select 
	    }
	}
    }
    
    method destroy_multi_level { } {
	set w .ui[modname]
	if {[winfo exists $w.main]} {
	    destroy $w.main
	}
	build_ui
    }

    method hasUI {} {
	return [winfo exists .ui[modname]]
    }
    
    method isOn { bval } {
	return  [set $this-$bval]
    }

    method alphaVal { sval } {
	return [set $this-$sval]
    }

}
