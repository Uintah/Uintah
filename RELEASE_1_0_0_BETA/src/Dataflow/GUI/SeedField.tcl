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


catch {rename SeedField ""}

# this is just for building a simple frame...

proc MMakeFrame {base label} {

        frame $base -relief sunken -bd 3
        pack $base -anchor nw

        label $base.label -text $label
        pack $base.label  -anchor nw
}
proc MMakeFrame2 {base sd an} {

        frame $base -relief sunken -bd 3

        pack $base  -side $sd -anchor $an
}

itcl_class SCIRun_Fields_SeedField {
    inherit Module
    constructor {config} {
        set name SeedField
        set_defaults
    }
    method set_defaults {} {
	global $this-num_samps $this-samps_alpha $this-draw_pts $this-draw_vec
	global $this-hedge_scale $this-alphad $this-reghedge
        global $this-xstep $this-ystep

        set $this-xstep 1
        set $this-ystep 1

	set $this-reghedge 0

	set $this-alphad 1.0

	set $this-num_samps 1.0
	set $this-samps_alpha 0.5
	set $this-draw_pts 1
	set $this-draw_vec 0

	set $this-hedge_scale 0.05

	global $this-aug_amount

	set $this-aug_amount 1.0
    }
    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }
        toplevel $w

	frame $w.f

	pack $w.f 

	MMakeFrame2 $w.f.left left w
	MMakeFrame2 $w.f.right right e
	
	MMakeFrame $w.f.left.t "Main Control" 
	MMakeFrame $w.f.left.v "Visualization Control"

	MMakeFrame $w.f.right.i "Element Sample"
	MMakeFrame $w.f.right.w "Widget Control"
	
	#element sample
	set me $w.f.right.i

#	expscale $me.ns -label "Num 3D Samples" \
#		-orient horizontal \
#		-variable $this-num_samps

	scale $me.alpha -label "Volume/Gradient Alpha" -from 0.0 -to 1.0 \
		-length 202 -orient horizontal \
		-variable $this-samps_alpha -resolution 0.001

	button $me.apwt -text "Do Augment" -command "$this-c init grad_wt"

	button $me.qrt -text "Augment 3" -command "$this-c init grad_wt2"

     pack $me.alpha $me.apwt $me.qrt -anchor nw
#    pack $me.alpha $me.apwt $me.qrt $me.ns -anchor nw

	# widget stuff
	set me $w.f.right.w

	scale $me.aug -label "Element Augment" -from 0.1 -to 10.0 \
		-length 202 -orient horizontal \
		-variable $this-aug_amount -resolution 0.01

        scale $me.xstep -label "X Step Size" -from 1 -to 5 \
                -length 202 -orient horizontal \
                -variable $this-xstep

        scale $me.ystep -label "Y Step Size" -from 1 -to 5 \
                -length 202 -orient horizontal \
                -variable $this-ystep

	button $me.augelem -text "Augment Elements" -command "$this augelem"

	pack $me.aug $me.augelem $me.xstep $me.ystep -anchor nw
	

	set me $w.f.left.t

	# this is just the basic buttons

	button $me.dist -text "Init Distribution" \
		-command "$this-c init weights"

	button $me.push -text "Compute Points" \
		-command "$this-c init dist_samp"

	pack $me.dist $me.push -anchor nw

	#visualization checks...

	set me $w.f.left.v
	
	checkbutton $me.pts -text "Point Cloud" -relief flat \
		-variable $this-draw_pts \
		-command "$this-c draw points"

	checkbutton $me.vec -text "Hedgehog" -relief flat \
		-variable $this-draw_vec \
		-command "$this-c draw vectors"

	checkbutton $me.rh -text "Reg Hedgehog" -relief flat \
		-variable $this-reghedge 

#	expscale $me.hedge -label "Spine Scale" \
#		-orient horizontal \
#		-variable $this-hedge_scale 


	scale $me.alph -label "Alpha Scale" -from 0.0 -to 1.0 \
		-length 202 -orient horizontal \
		-variable $this-alphad -resolution 0.001 

	button $me.falpha -text "Flush Alpha" \
		-command "$this DoAlpha"

	pack $me.pts $me.vec $me.rh $me.alph $me.falpha -anchor nw
#	pack $me.pts $me.vec $me.rh $me.alph $me.falpha $me.hedge -anchor nw
    }
    method augelem {} {
	global $this-aug_amount
	$this-c init widget [set $this-aug_amount]
    }
    method DoAlpha {} {
	global $this-alphad
	$this-c init alpha [set $this-alphad]
    }
}



