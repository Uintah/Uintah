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
	global $this-lambda_fix
	global $this-lambda_sld
	global $this-lambda_lc
	global $this-reg_method
	global $this-haveUI
	set $this-lambda_fix 0.02
	set $this-lambda_sld 0.02
	set $this-lambda_lc 0.0
    	set $this-reg_method lcurve
	set $this-haveUI 0
    }
    method entervalue {} {
        set w .ui[modname]
	set color "#505050"
        #$this-c needexecute
	$w.ent.f2.l1 configure -foreground black
	$w.ent.f2.e1 configure -state normal -foreground black
	$w.sld.f4.s configure -state disabled -foreground $color
	$w.lc.lam.l configure -foreground $color
	$w.lc.lam.e configure -state disabled -foreground $color
	clear_graph
    }
    method useslider {} {
        set w .ui[modname]
	set color "#505050"
        #$this-c needexecute
	$w.sld.f4.s configure -state normal -foreground black
	$w.ent.f2.e1 configure -state disabled -foreground $color
	$w.ent.f2.l1 configure -foreground $color
	$w.lc.lam.l configure -foreground $color
	$w.lc.lam.e configure -state disabled -foreground $color
	clear_graph
    }
    method uselcurve {} {
        set w .ui[modname]
	set color "#505050"
        #$this-c needexecute
	$w.sld.f4.s configure -state disabled -foreground $color
	$w.ent.f2.l1 configure -foreground $color
	$w.ent.f2.e1 configure -state disabled -foreground $color
	$w.lc.lam.l configure -foreground black
	$w.lc.lam.e configure -state normal -foreground black
    }
    method execrunmode {} {
        set w .ui[modname]
        $this-c needexecute
    }
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
    method clear_graph {} {
        set w .ui[modname]
        if {![winfo exists $w]} {
            return
        }
        $w.lc.data.g element configure LCurve -data "1 0"
        $w.lc.data.g element configure LCorner -data "1 0"
    }
    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }

	#set n "$this-c needexecute"

	global $this-haveUI
	set $this-haveUI 1

	toplevel $w
        wm minsize $w 150 20
	frame $w.titlefr -relief groove -border 3.5
	label $w.titlefr.l -text "Select Method for Lambda"
	pack $w.titlefr -side top -padx 2 -pady 2 -fill both
	pack $w.titlefr.l -side top -fill x

	global $this-lambda_fix
	global $this-lambda_sld
	global $this-reg_method
#    	set $this-reg_method lcurve

	# Entry radio-button
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

	# Slider radio-button
        frame $w.sld -relief groove -border 3
        pack $w.sld -side top -expand yes -fill x

	frame $w.sld.f3 -relief flat
        pack $w.sld.f3 -side top -expand yes -fill x
        radiobutton $w.sld.f3.b -text "Choose using slider" \
		-variable "$this-reg_method" -value slider \
		-command "$this useslider"
        pack $w.sld.f3.b -side left

        frame $w.sld.f4 -relief flat
        pack $w.sld.f4 -side top -expand yes -fill x
	scale $w.sld.f4.s -label "Lambda: " -from 0.0 -to 5.0 \
		-resolution 0.01 -orient horizontal \
		-variable $this-lambda_sld -showvalue true
        pack $w.sld.f4.s -side top -expand yes -fill x -padx 2 -pady 2

	# LCurve radio-button
        frame $w.lc -relief groove -border 3
        pack $w.lc -side top -expand yes -fill x

        frame $w.lc.f5 -relief flat
        pack $w.lc.f5 -side top -expand yes -fill x
        radiobutton $w.lc.f5.b -text "L-curve" \
		-variable "$this-reg_method" -value lcurve \
		-command "$this uselcurve"
        pack $w.lc.f5.b -side left

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
	label $w.lc.lam.l -text "Lambda: "
	entry $w.lc.lam.e -textvariable $this-lambda_lc -state disabled
	pack $w.lc.lam.l $w.lc.lam.e -side left -fill x -expand 1
	pack $w.lc.lam -side top -fill x

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
