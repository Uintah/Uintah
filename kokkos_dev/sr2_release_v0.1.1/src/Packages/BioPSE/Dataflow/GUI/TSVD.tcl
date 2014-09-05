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


    catch {rename BioPSE_Inverse_TSVD ""}

    itcl_class BioPSE_Inverse_TSVD {
    inherit Module
    constructor {config} {
        set name TSVD
        set_defaults
    }
    method set_defaults {} {
	
	global $this-lambda_fix
	global $this-lambda_sld
	global $this-lambda_max
	global $this-lambda_lc
	global $this-reg_method
	global $this-have_ui
	global $this-text_var	
	
	
	set $this-lambda_fix 12
	set $this-lambda_sld 1
	set $this-lambda_lc 0.0
    	set $this-reg_method lcurve
	set $this-have_ui 0
	set $this-lambda_max 100
	set $this-text_var   1
     }
    method entervalue {} {
        set w .ui[modname]
	set color "#505050"
        #$this-c needexecute
	$w.ent.f2.l1 configure -foreground black
	$w.ent.f2.e1 configure -state normal -foreground black

	$w.lc.lam.l configure -foreground $color
	$w.lc.lam.e configure -state disabled -foreground $color
	clear_graph


	}
     method uselcurve {} {
        set w .ui[modname]
	set color "#505050"
        #$this-c needexecute

	$w.ent.f2.l1 configure -foreground $color
	$w.ent.f2.e1 configure -state disabled -foreground $color
	$w.lc.lam.l configure -foreground black
	$w.lc.lam.e configure -state normal -foreground black
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
    ###############
    method calc_lambda {lmax sld_val} {
	#global $this-tex_var
	set w .ui[modname]
	set $this-text_var [expr {round($sld_val/10* $lmax-.499) }] 
    }
    ##############
    method ui {} {
	set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }

	global $this-have_ui
	set $this-have_ui 1

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

	}
}
