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
	global $this-shading_tab
	global $this-sampling_tab
	global $this-multires_tab

	# For backwards compatability
	global $this-contrast
	global $this-contrastfp
	global $this-draw_mode
	global $this-num_slices
	set $this-num_slices -1

        global $this-shading-button-state
        set $this-shading-button-state 1
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

	#frame for tabs
	frame $w.main.options
	pack $w.main.options -padx 2 -pady 2 -side top -fill x -expand 1
	#frame for display
	frame $w.main.options.disp -borderwidth 2
	pack $w.main.options.disp -padx 2 -pady 2 -side left \
	    -fill both -expand 1

	# Tabs
	iwidgets::labeledframe $w.main.options.disp.frame_title \
	    -labelpos nw -labeltext "Display Options"
	set dof [$w.main.options.disp.frame_title childsite]

	iwidgets::tabnotebook $dof.tabs -height 320  -width 300 \
	    -raiseselect true
	pack $dof.tabs -side top -fill x -expand yes

	add_default_tab $dof
	add_sampling_tab $dof
	add_shading_tab $dof
	$dof.tabs view 0
	$dof.tabs configure -tabpos "n"
	
	pack $w.main.options.disp.frame_title -side top -expand yes -fill x
	
	if { [set $this-multi_level] > 1 } {
	    add_multires_tab $dof
	}
 
	makeSciButtonPanel $w.main $w $this
        $this state
	moveToCursor $w

    }
    method add_default_tab { dof } {
        #-----------------------------------------------------------
        # Standard controls
        #-----------------------------------------------------------
	set tab [$dof.tabs add -label "Basic"]

	frame $tab.f -relief groove -borderwidth 2 
	pack $tab.f -padx 2 -pady 2 -fill x
	set n "$this-c needexecute"
        set s "$this state"

	global $this-render_style
	label $tab.f.l -text "Blending Mode"
	pack $tab.f.l -side top -fill x

        frame $tab.f.fmode
        pack $tab.f.fmode -padx 2 -pady 2 -fill x
	radiobutton $tab.f.fmode.modeo -text "Over Operator" -relief flat \
		-variable $this-render_style -value 0 \
		-anchor w -command $n
	radiobutton $tab.f.fmode.modem -text "MIP" -relief flat \
		-variable $this-render_style -value 1 \
		-anchor w -command $n
	pack $tab.f.fmode.modeo $tab.f.fmode.modem \
            -side left -fill x -padx 4 -pady 4
	pack $tab.f.fmode.modeo $tab.f.fmode.modem \
            -side left -fill x -padx 10 -pady 4 -expand y

        frame $tab.f.fres
        pack $tab.f.fres -padx 2 -pady 2 -fill x
        label $tab.f.fres.res -text "Resolution (bits)"
	radiobutton $tab.f.fres.b0 -text 8 -variable $this-blend_res -value 8 \
	    -command $n
	radiobutton $tab.f.fres.b1 -text 16 -variable $this-blend_res -value 16 \
	    -command $n
	radiobutton $tab.f.fres.b2 -text 32 -variable $this-blend_res -value 32 \
	    -command $n
	pack $tab.f.fres.res $tab.f.fres.b0 $tab.f.fres.b1 $tab.f.fres.b2 \
            -side left -fill x -padx 4 -pady 4

  	frame $tab.interp -relief groove -borderwidth 2
 	label $tab.interp.l -text "Interpolation Mode"
	frame $tab.interp.f
 	radiobutton $tab.interp.f.interp -text "Trilinear" -relief flat \
 		-variable $this-interp_mode -value 1 \
 		-anchor w -command $n
 	radiobutton $tab.interp.f.near -text "Nearest" -relief flat \
 		-variable $this-interp_mode -value 0 \
 		-anchor w -command $n
 	pack $tab.interp -padx 2 -pady 2 -fill x -side top
 	pack $tab.interp.l $tab.interp.f -side top -fill x -padx 4
	pack $tab.interp.f.interp $tab.interp.f.near -side left -fill x \
	    -padx 10 -expand y

        
        #-----------------------------------------------------------
        # Transfer Function
        #-----------------------------------------------------------
        frame $tab.tf -relief groove -borderwidth 2
        pack $tab.tf -padx 2 -pady 2 -fill x
        label $tab.tf.l -text "Transfer Function"

	scale $tab.tf.stransp -variable $this-alpha_scale \
		-from -1.0 -to 1.0 -label "Global Opacity" \
		-showvalue true -resolution 0.001 \
		-orient horizontal 

# 	scale $w.tf.cmap_size -variable $this-cmap_size \
# 		-from 4 -to 10 -label "Table Size (2^n)" \
# 		-showvalue true -resolution 1 \
# 		-orient horizontal \

	checkbutton $tab.tf.sw -text "Software ColorMap2 Rasterization" \
            -relief flat -variable $this-sw_raster -onvalue 1 -offvalue 0 \
            -anchor w -command "$n"

	pack $tab.tf.l $tab.tf.stransp $tab.tf.sw \
            -side top -fill x -padx 4 -pady 2


#	bind $tab.tf.cmap_size <ButtonRelease> $n
	bind $tab.tf.stransp <ButtonRelease> $n
   }	
    
    method add_sampling_tab { dof } {
	set n "$this-c needexecute"
        set s "$this state"
        #-----------------------------------------------------------
        # Sampling
        #-----------------------------------------------------------
	set $this-sampling_tab [$dof.tabs add -label "Sampling"]
	set tab [set $this-sampling_tab]
#          frame $w.main.sampling -relief groove -borderwidth 2
#          pack $w.main.sampling -padx 2 -pady 2 -fill x
          label $tab.l -text "Sampling"


	scale $tab.srate_hi -variable $this-sampling_rate_hi \
            -from 0.5 -to 10.0 -label "Sampling Rate" \
            -showvalue true -resolution 0.1 \
            -orient horizontal \

	checkbutton $tab.adaptive -text "Adaptive Sampling" -relief flat \
            -variable $this-adaptive -onvalue 1 -offvalue 0 \
            -anchor w -command "$s; $n"

	scale $tab.srate_lo -variable $this-sampling_rate_lo \
            -from 0.1 -to 5.0 -label "Interactive Sampling Rate" \
            -showvalue true -resolution 0.1 \
            -orient horizontal \

	pack $tab.l $tab.srate_hi $tab.adaptive \
            $tab.srate_lo -side top -fill x -padx 4 -pady 2
	
	bind $tab.srate_hi <ButtonRelease> $n
	bind $tab.srate_lo <ButtonRelease> $n
	
    }
    method add_shading_tab { dof } {
	set n "$this-c needexecute"
        set s "$this state"
	#-----------------------------------------------------------
	# Shading
	#-----------------------------------------------------------
	set $this-shading_tab [$dof.tabs add -label "Shading"]
	set tab [set $this-shading_tab]
	#	frame $w.main.f4 -relief groove -borderwidth 2
	#	pack $w.main.f4 -padx 2 -pady 2 -fill x
	checkbutton $tab.shading -text "Shading" -relief flat \
	    -variable $this-shading -onvalue 1 -offvalue 0 \
	    -anchor w -command "$s; $n"
	pack $tab.shading -side top -fill x -padx 4

	#-----------------------------------------------------------
	# Light
	#-----------------------------------------------------------
	frame $tab.f0
	pack $tab.f0 -padx 2 -pady 2 -fill x
	label $tab.f0.light -text "Attach Light to"
	radiobutton $tab.f0.light0 -text "Light 0" -relief flat \
	    -variable $this-light -value 0 \
	    -anchor w -command $n
	radiobutton $tab.f0.light1 -text "Light 1" -relief flat \
	    -variable $this-light -value 1 \
	    -anchor w -command $n
	#  	radiobutton $w.f5.light2 -text "Light 2" -relief flat \
	    #             -variable $this-light -value 2 \
	    #             -anchor w -command $n
	#  	radiobutton $w.f5.light3 -text "Light 3" -relief flat \
	    #             -variable $this-light -value 3 \
	    #             -anchor w -command $n
	pack $tab.f0.light $tab.f0.light0 $tab.f0.light1 \
	    -side left -fill x -padx 4

	#-----------------------------------------------------------
	# Material
	#-----------------------------------------------------------
	frame $tab.f1 -relief groove -borderwidth 2
	pack $tab.f1 -padx 2 -pady 2 -fill x
	label $tab.f1.material -text "Material"
	global $this-ambient
	scale $tab.f1.ambient -variable $this-ambient \
	    -from 0.0 -to 1.0 -label "Ambient" \
	    -showvalue true -resolution 0.001 \
	    -orient horizontal
	global $this-diffuse
	scale $tab.f1.diffuse -variable $this-diffuse \
	    -from 0.0 -to 1.0 -label "Diffuse" \
	    -showvalue true -resolution 0.001 \
	    -orient horizontal
	global $this-specular
	scale $tab.f1.specular -variable $this-specular \
	    -from 0.0 -to 1.0 -label "Specular" \
	    -showvalue true -resolution 0.001 \
	    -orient horizontal
	global $this-shine
	scale $tab.f1.shine -variable $this-shine \
	    -from 1.0 -to 128.0 -label "Shine" \
	    -showvalue true -resolution 1.0 \
	    -orient horizontal
	pack $tab.f1.material $tab.f1.ambient $tab.f1.diffuse \
	    $tab.f1.specular $tab.f1.shine \
	    -side top -fill x -padx 4

        bind $tab.f1.ambient <ButtonRelease> $n
        bind $tab.f1.diffuse <ButtonRelease> $n
        bind $tab.f1.specular <ButtonRelease> $n
        bind $tab.f1.shine <ButtonRelease> $n

        change_shading_state [set $this-shading-button-state]
    }


    method state {} {
	set w .ui[modname]
	if {[set $this-shading] == 1} {
	    set tab [set $this-shading_tab]
            $this activate $tab.f1.ambient
            $this activate $tab.f1.diffuse
            $this activate $tab.f1.specular
            $this activate $tab.f1.shine
            $this activate $tab.f0.light
            $this activate $tab.f0.light0
            $this activate $tab.f0.light1
#             $this activate $w.f5.light2
#             $this activate $w.f5.light3
	} else {
	    set tab [set $this-shading_tab]
            $this deactivate $tab.f1.ambient
            $this deactivate $tab.f1.diffuse
            $this deactivate $tab.f1.specular
            $this deactivate $tab.f1.shine
            $this deactivate $tab.f0.light
            $this deactivate $tab.f0.light0
            $this deactivate $tab.f0.light1
#             $this deactivate $w.f5.light2
#             $this deactivate $w.f5.light3
	}
	if {[set $this-adaptive] == 1} {
	    set tab [set $this-sampling_tab]
            $this activate $tab.srate_lo
        } else {
	    set tab [set $this-sampling_tab]
            $this deactivate $tab.srate_lo
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

    method add_multires_tab { dof } {
	set $this-multires_tab [$dof.tabs add -label "Multires"]
	set tab [set $this-multires_tab]

	frame $tab.f -relief groove -borderwidth 2
	pack $tab.f -padx 2 -pady 2 -fill x -expand yes	
	frame $tab.f.f1 -relief flat -borderwidth 2
	pack $tab.f.f1 -padx 2 -pady 2 -fill x -expand yes
	checkbutton $tab.f.f1.stencil -text "Use Stencil" \
	    -variable $this-use_stencil -command "$this-c needexecute"
	checkbutton $tab.f.f1.opacity -text "Highlight Levels" \
	    -variable $this-invert_opacity -command "$this-c needexecute"
	pack $tab.f.f1.stencil $tab.f.f1.opacity -side left
	
	frame $tab.f.f2 -relief flat -borderwidth 2
	pack $tab.f.f2 -padx 2 -pady 2 -fill x -expand yes
	label $tab.f.f2.l -text "Show level"
	pack $tab.f.f2.l -side left
	frame $tab.f.f2.f -relief flat -borderwidth 2
	pack $tab.f.f2.f -side right
	set selected 0
	for { set i 0 } { $i < [set $this-multi_level] } { incr i } {
	    frame $tab.f.f2.f.f$i -relief flat
	    pack $tab.f.f2.f.f$i -fill x -expand yes -side top
	    checkbutton $tab.f.f2.f.f$i.b -text $i  \
		-variable $this-l$i -command "$this-c needexecute" 
	    scale $tab.f.f2.f.f$i.s -variable $this-s$i \
		-from -1.0 -to 1.0 -orient horizontal -resolution 0.01

	    pack $tab.f.f2.f.f$i.b $tab.f.f2.f.f$i.s -side left
	    if { [isOn l$i] } {
		set selected 1
	    }
	    bind $tab.f.f2.f.f$i.s <ButtonRelease> "$this-c needexecute" 
	}
	if { !$selected && [winfo exists $tab.f.f2.f.f0.b] } {  
	    $tab.f.f2.f.f0.b select 
	}
    }

    method build_multi_level { } {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    set dof [$w.main.options.disp.frame_title childsite]
	    add_multires_tab $dof
	}

    }
    
    method destroy_multi_level { } {
	set w .ui[modname]
	if {[winfo exists $w.main]} {
	    set dof [$w.main.options.disp.frame_title childsite]
	    $dof.tabs delete 3
	}
#	build_ui
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

    method change_shading_state { val } {
        set $this-shading-button-state $val

        set w .ui[modname] 

        if {![winfo exists $w]} { 
            return
        }
        
	set dof [$w.main.options.disp.frame_title childsite]
        set tab [$dof.tabs childsite "Shading"]
        
        if { $val } {
            $tab.shading configure -fg "black"
        } else {
            $tab.shading configure -fg "darkgrey"
        }
    }
}
