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


catch {rename VolumeVisualizer ""}

itcl_class Volume_Visualization_VolumeVisualizer {
    inherit Module
    constructor {config} {
	set name VolumeVisualizer
	set_defaults
    }
    method set_defaults {} {
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
	global $this-contrast
	set $this-contrast 0.5
	global $this-contrastfp
	set $this-contrastfp 0.5
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
	set n "$this-c needexecute"
        set s "$this state"

	global $this-render_style
	label $w.f.l -text "Blending"
	pack $w.f.l -side top -fill x

        frame $w.f.fmode
        pack $w.f.fmode -padx 2 -pady 2 -fill x
        label $w.f.fmode.mode -text "Mode"
	radiobutton $w.f.fmode.modeo -text "Over Operator" -relief flat \
		-variable $this-render_style -value 0 \
		-anchor w -command $n
	radiobutton $w.f.fmode.modem -text "MIP" -relief flat \
		-variable $this-render_style -value 1 \
		-anchor w -command $n
	pack $w.f.fmode.mode $w.f.fmode.modeo $w.f.fmode.modem \
            -side left -fill x -padx 4 -pady 4

        frame $w.f.fres
        pack $w.f.fres -padx 2 -pady 2 -fill x
        label $w.f.fres.res -text "Resolution (bits)"
	radiobutton $w.f.fres.b0 -text 8 -variable $this-blend_res -value 8 \
	    -command $n
	radiobutton $w.f.fres.b1 -text 16 -variable $this-blend_res -value 16 \
	    -command $n
	radiobutton $w.f.fres.b2 -text 32 -variable $this-blend_res -value 32 \
	    -command $n
	pack $w.f.fres.res $w.f.fres.b0 $w.f.fres.b1 $w.f.fres.b2 \
            -side left -fill x -padx 4 -pady 4

        #-----------------------------------------------------------
        # Shading
        #-----------------------------------------------------------
	frame $w.f4 -relief groove -borderwidth 2
	pack $w.f4 -padx 2 -pady 2 -fill x
	checkbutton $w.f4.shading -text "Shading" -relief flat \
            -variable $this-shading -onvalue 1 -offvalue 0 \
            -anchor w -command "$s; $n"
        pack $w.f4.shading -side top -fill x -padx 4

        #-----------------------------------------------------------
        # Light
        #-----------------------------------------------------------
 	frame $w.f5
 	pack $w.f5 -padx 2 -pady 2 -fill x
 	label $w.f5.light -text "Attach Light to"
 	radiobutton $w.f5.light0 -text "Light 0" -relief flat \
            -variable $this-light -value 0 \
            -anchor w -command $n
 	radiobutton $w.f5.light1 -text "Light 1" -relief flat \
            -variable $this-light -value 1 \
            -anchor w -command $n
#  	radiobutton $w.f5.light2 -text "Light 2" -relief flat \
#             -variable $this-light -value 2 \
#             -anchor w -command $n
#  	radiobutton $w.f5.light3 -text "Light 3" -relief flat \
#             -variable $this-light -value 3 \
#             -anchor w -command $n
        pack $w.f5.light $w.f5.light0 $w.f5.light1 \
            -side left -fill x -padx 4

        #-----------------------------------------------------------
        # Material
        #-----------------------------------------------------------
	frame $w.f6 -relief groove -borderwidth 2
	pack $w.f6 -padx 2 -pady 2 -fill x
 	label $w.f6.material -text "Material"
	global $this-ambient
	scale $w.f6.ambient -variable $this-ambient \
            -from 0.0 -to 1.0 -label "Ambient" \
            -showvalue true -resolution 0.001 \
            -orient horizontal
	global $this-diffuse
	scale $w.f6.diffuse -variable $this-diffuse \
		-from 0.0 -to 1.0 -label "Diffuse" \
		-showvalue true -resolution 0.001 \
		-orient horizontal
	global $this-specular
	scale $w.f6.specular -variable $this-specular \
		-from 0.0 -to 1.0 -label "Specular" \
		-showvalue true -resolution 0.001 \
		-orient horizontal
	global $this-shine
	scale $w.f6.shine -variable $this-shine \
		-from 1.0 -to 128.0 -label "Shine" \
		-showvalue true -resolution 1.0 \
		-orient horizontal
        pack $w.f6.material $w.f6.ambient $w.f6.diffuse \
            $w.f6.specular $w.f6.shine \
            -side top -fill x -padx 4


# 	frame $w.f3 -relief groove -borderwidth 2
# 	pack $w.f3 -padx 2 -pady 2 -fill x
# 	label $w.f3.l -text "Interpolation Mode"
# 	radiobutton $w.f3.interp -text "Trilinear" -relief flat \
# 		-variable $this-interp_mode -value 1 \
# 		-anchor w -command $n
# 	radiobutton $w.f3.near -text "Nearest" -relief flat \
# 		-variable $this-interp_mode -value 0 \
# 		-anchor w -command $n
# 	pack $w.f3.l $w.f3.interp $w.f3.near -side top -fill x -padx 4

        #-----------------------------------------------------------
        # Sampling
        #-----------------------------------------------------------
        frame $w.sampling -relief groove -borderwidth 2
        pack $w.sampling -padx 2 -pady 2 -fill x
        label $w.sampling.l -text "Sampling"

	scale $w.sampling.srate_hi -variable $this-sampling_rate_hi \
            -from 0.5 -to 10.0 -label "Sampling Rate" \
            -showvalue true -resolution 0.1 \
            -orient horizontal \

	checkbutton $w.sampling.adaptive -text "Adaptive Sampling" -relief flat \
            -variable $this-adaptive -onvalue 1 -offvalue 0 \
            -anchor w -command "$s; $n"

	scale $w.sampling.srate_lo -variable $this-sampling_rate_lo \
            -from 0.1 -to 5.0 -label "Interactive Sampling Rate" \
            -showvalue true -resolution 0.1 \
            -orient horizontal \

	pack $w.sampling.l $w.sampling.srate_hi $w.sampling.adaptive \
            $w.sampling.srate_lo -side top -fill x -padx 4 -pady 2
        
        #-----------------------------------------------------------
        # Transfer Function
        #-----------------------------------------------------------
        frame $w.tf -relief groove -borderwidth 2
        pack $w.tf -padx 2 -pady 2 -fill x
        label $w.tf.l -text "Transfer Function"

	scale $w.tf.stransp -variable $this-alpha_scale \
		-from -1.0 -to 1.0 -label "Global Opacity" \
		-showvalue true -resolution 0.001 \
		-orient horizontal 

# 	scale $w.tf.cmap_size -variable $this-cmap_size \
# 		-from 4 -to 10 -label "Table Size (2^n)" \
# 		-showvalue true -resolution 1 \
# 		-orient horizontal \

	checkbutton $w.tf.sw -text "Software Rasterization" -relief flat \
            -variable $this-sw_raster -onvalue 1 -offvalue 0 \
            -anchor w -command "$n"

	pack $w.tf.l $w.tf.stransp $w.tf.sw \
            -side top -fill x -padx 4 -pady 2

        bind $w.f6.ambient <ButtonRelease> $n
        bind $w.f6.diffuse <ButtonRelease> $n
        bind $w.f6.specular <ButtonRelease> $n
        bind $w.f6.shine <ButtonRelease> $n

	bind $w.sampling.srate_hi <ButtonRelease> $n
	bind $w.sampling.srate_lo <ButtonRelease> $n

#	bind $w.tf.cmap_size <ButtonRelease> $n
	bind $w.tf.stransp <ButtonRelease> $n
	
	makeSciButtonPanel $w $w $this
        $this state
	moveToCursor $w
    }

    method state {} {
	set w .ui[modname]
	if {[set $this-shading] == 1} {
            $this activate $w.f6.ambient
            $this activate $w.f6.diffuse
            $this activate $w.f6.specular
            $this activate $w.f6.shine
            $this activate $w.f5.light
            $this activate $w.f5.light0
            $this activate $w.f5.light1
#             $this activate $w.f5.light2
#             $this activate $w.f5.light3
	} else {
            $this deactivate $w.f6.ambient
            $this deactivate $w.f6.diffuse
            $this deactivate $w.f6.specular
            $this deactivate $w.f6.shine
            $this deactivate $w.f5.light
            $this deactivate $w.f5.light0
            $this deactivate $w.f5.light1
#             $this deactivate $w.f5.light2
#             $this deactivate $w.f5.light3
	}
	if {[set $this-adaptive] == 1} {
            $this activate $w.sampling.srate_lo
        } else {
            $this deactivate $w.sampling.srate_lo
        }
    }
    method activate { w } {
	$w configure -state normal -foreground black
    }
    method deactivate { w } {
	$w configure -state disabled -foreground darkgrey
    }


}
