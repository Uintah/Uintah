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

    catch {rename BioPSE_Inverse_Tikhonov ""}

    itcl_class BioPSE_Inverse_Tikhonov {
    inherit Module
    constructor {config} {
        set name Tikhonov
        set_defaults
    }
    method set_defaults {} {
	global $this-tex_var
	global $this-lambda_fix
	global $this-lambda_sld
	global $this-lambda_lc
	global $this-reg_method
	global $this-lambda_num
	global $this-haveUI
	global $this-lambda_min
	global $this-lambda_max
	set $this-tex_var 0.02
	set $this-lambda_fix 0.02
	set $this-lambda_sld 0.0
	set $this-lambda_lc 0.0
    	set $this-reg_method lcurve
	set $this-lambda_num 200
	set $this-haveUI 0
	set $this-lambda_min 1e-6
	set $this-lambda_max 10
    }
    method entervalue {} {
        set w .ui[modname]
	set color "#505050"
        #$this-c needexecute
	$w.ent.f2.l1 configure -foreground black
	$w.ent.f2.e1 configure -state normal -foreground black
	$w.sld.sld.f3.s configure -state disabled -foreground $color
	$w.lc.lam.l configure -foreground $color
	$w.lc.lam.e configure -state disabled -foreground $color
	clear_graph
	$w.sld.sld.f2.lvalue configure -state disabled -foreground $color
	$w.sld.sld.f2.l1 configure -foreground $color
	$w.lc.f2.e1 configure -state disabled -foreground $color
	$w.lc.f2.l1 configure -foreground $color
	}
    method useslider {} {
        set w .ui[modname]
	set color "#505050"
        $this calc_log [set $this-lambda_max] [set $this-lambda_min] [set $this-lambda_sld]
	$w.sld.sld.f3.s configure -state normal -foreground black
	$w.ent.f2.e1 configure -state disabled -foreground $color
	$w.ent.f2.l1 configure -foreground $color
	$w.lc.lam.l configure -foreground $color
	$w.lc.lam.e configure -state disabled -foreground $color
	clear_graph
	$w.sld.sld.f2.lvalue configure -state disabled -foreground black
	$w.sld.sld.f2.l1 configure -foreground black
	$w.lc.f2.e1 configure -state disabled -foreground $color
	$w.lc.f2.l1 configure -foreground $color
	}
    method uselcurve {} {
        set w .ui[modname]
	set color "#505050"
        #$this-c needexecute
	$w.sld.sld.f3.s configure -state disabled -foreground $color
	$w.ent.f2.l1 configure -foreground $color
	$w.ent.f2.e1 configure -state disabled -foreground $color
	$w.lc.lam.l configure -foreground black
	$w.lc.lam.e configure -state normal -foreground black
	$w.sld.sld.f2.lvalue configure -state disabled -foreground $color
	$w.sld.sld.f2.l1 configure -foreground $color
	$w.lc.f2.e1 configure -state normal -foreground black
	$w.lc.f2.l1 configure -foreground black
    }
    method execrunmode {} {
        set w .ui[modname]
        $this-c needexecute
    }
    ##############
    method plot_graph {lcurv lcor lam} {
        set w .ui[modname]
        if {![winfo exists $w]} {
            return
        }
	global $this-lambda_lc
	set $this-lambda_lc $lam

        $w.lc.data.g element configure LCurve -data $lcurv
        $w.lc.data.g element configure LCorner -data $lcor
    }
    ##############
    method clear_graph {} {
        set w .ui[modname]
        if {![winfo exists $w]} {
            return
        }
        $w.lc.data.g element configure LCurve -data "1 0"
        $w.lc.data.g element configure LCorner -data "1 0"
    }
    ##############
  	method calc_log {l_max l_min slider_value} {
	global $this-tex_var
	set w .ui[modname]
	set $this-tex_var [expr {[set $this-lambda_min]*pow(10,$slider_value/5*log10([set $this-lambda_max]/[set $this-lambda_min]))}] 
	}	

    ##############	
    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }

	global $this-haveUI
	set $this-haveUI 1

        ###################################
	#THIS PART IS FOR MAKING THE TITLE
	###################################
	toplevel $w
        wm minsize $w 150 20
	frame $w.titlefr -relief groove -border 3.5
	label $w.titlefr.l -text "Select Method for Lambda"
	pack $w.titlefr -side top -padx 2 -pady 2 -fill both
	pack $w.titlefr.l -side top -fill x

	global $this-lambda_fix
	global $this-lambda_sld
	global $this-reg_method
	
	#######################
	# Entry radio-button
	#######################
        frame $w.ent -relief groove -border 3
        pack $w.ent -side top -expand yes -fill x

        frame $w.ent.f1 -relief flat
        pack $w.ent.f1 -side top -expand yes -fill x

        radiobutton $w.ent.f1.b -text "Enter value" \
		-variable "$this-reg_method" -value single \
		-command "$this entervalue"

        pack $w.ent.f1.b -side left

        frame $w.ent.f2 -relief flat
        pack $w.ent.f2 -side top -expand yes -fill x
        label $w.ent.f2.l1 -text "Lambda: "
        entry $w.ent.f2.e1 -textvariable $this-lambda_fix
        pack $w.ent.f2.l1 $w.ent.f2.e1 -side left -expand yes \
		-fill x -padx 2 -pady 2
        #bind $w.ent.f2.e1 <Return> $n
	#########################
	# Slider radio-button
	#########################
        frame $w.sld -relief flat
        pack $w.sld -side top -expand yes -fill x
	
	frame $w.sld.sld -relief groove -border 3
        pack $w.sld.sld -side left -expand yes -fill x
        frame $w.sld.rng -relief groove -border 3
	pack $w.sld.rng -side right  -fill y
	
	frame $w.sld.rng.f1 -relief flat
	pack  $w.sld.rng.f1 -side top -fill x
	label $w.sld.rng.f1.l1 -text "Lambda Range"	
	pack  $w.sld.rng.f1.l1 -side top -fill x

	frame $w.sld.rng.f2 -relief flat
	pack  $w.sld.rng.f2 -side top -fill x
	entry $w.sld.rng.f2.e1 -textvariable $this-lambda_min
	label $w.sld.rng.f2.l1 -text "From "
	pack $w.sld.rng.f2.e1  $w.sld.rng.f2.l1 -side right
	
	frame $w.sld.rng.f3 -relief flat
	pack  $w.sld.rng.f3 -side top -fill x
	entry $w.sld.rng.f3.e1 -textvariable $this-lambda_max
	label $w.sld.rng.f3.l1 -text "  To  "
	pack $w.sld.rng.f3.e1  $w.sld.rng.f3.l1 -side right

	frame $w.sld.sld.f1 -relief flat
        pack $w.sld.sld.f1 -side top -expand yes -fill x
        radiobutton $w.sld.sld.f1.b -text "Choose using slider" \
		-variable "$this-reg_method" -value slider \
		-command "$this useslider"
        pack $w.sld.sld.f1.b -side left

	frame $w.sld.sld.f2 -relief flat
        pack $w.sld.sld.f2 -side top -expand yes -fill x
	
	entry $w.sld.sld.f2.lvalue -textvariable "$this-tex_var" -state disabled       
	label $w.sld.sld.f2.l1 -text "Lambda: "
	pack $w.sld.sld.f2.l1 $w.sld.sld.f2.lvalue -side left
   
	frame $w.sld.sld.f3 -relief flat
        pack $w.sld.sld.f3 -side top -expand yes -fill x
	scale $w.sld.sld.f3.s -from 0.0 -to 5.0 \
		-resolution 0.01 -orient horizontal \
		-variable "$this-lambda_sld" -showvalue false \
		-command "$this calc_log [set $this-lambda_max] [set $this-lambda_min] " 
	pack $w.sld.sld.f3.s -side left -expand yes -fill x -padx 2 -pady 2

	
	
	#######################
	# LCurve radio-button
	#######################

        frame $w.lc -relief groove -border 3
        pack $w.lc -side top -expand yes -fill x
	
        frame $w.lc.f1 -relief flat
        pack $w.lc.f1 -side top -expand yes -fill x
        radiobutton $w.lc.f1.b -text "L-curve             " \
		-variable "$this-reg_method" -value lcurve \
		-command "$this uselcurve"
	
        pack $w.lc.f1.b -side left
	
	frame $w.lc.f2 -relief flat
        pack $w.lc.f2 -side top -expand yes -fill x
	label $w.lc.f2.l1 -text "Number of Points: "
	entry $w.lc.f2.e1 -textvariable $this-lambda_num 
	pack $w.lc.f2.e1 $w.lc.f2.l1 -side right	

        frame $w.lc.data -relief groove -borderwidth 2
        blt::graph $w.lc.data.g -height 250 \
                -plotbackground #CCCCFF
        $w.lc.data.g element create LCurve -data "1 0" -color blue -symbol ""
        $w.lc.data.g element create LCorner -data "1 0" -color green -symbol ""
        $w.lc.data.g yaxis configure -logscale true -title "|| Rx ||"
        $w.lc.data.g xaxis configure -logscale true -title "|| Ax - y ||"
        pack $w.lc.data.g -side top -fill x
        pack $w.lc.data -side top -fill x
	
	global $this-lambda_lc
	frame $w.lc.lam -relief flat
	label $w.lc.lam.l -text "Lambda Corner: "
	entry $w.lc.lam.e -textvariable $this-lambda_lc -state disabled
	pack $w.lc.lam.l $w.lc.lam.e -side left -fill x -expand 1
	pack $w.lc.lam -side top -fill x

	######################
	#Execute Close Part
	######################

        frame $w.f6 -relief groove -border 3
        pack $w.f6 -side top -expand yes -fill x
	button $w.f6.ex -text "Execute" -command "$this execrunmode"
	button $w.f6.cl -text "Close" -command "destroy $w"
        pack $w.f6.ex $w.f6.cl -side left -expand yes -fill x -padx 2 -pady 2
	
	if {[set $this-reg_method] == "lcurve"} { $this uselcurve }
	if {[set $this-reg_method] == "single"} { $this entervalue }
	if {[set $this-reg_method] == "slider"} { $this useslider }
	}
}
