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

##
 #  ComputeCurrent.tcl: Set theta and phi for the dipole
 #
 #  Written by:
 #   David Weinstein
 #   Department of Computer Science
 #   University of Utah
 #   June 1999
 #
 #  Copyright (C) 1999 SCI Group
 # 
 #  Log Information:
 #
 ##

catch {rename BioPSE_LeadField_BuildMisfitField ""}

itcl_class BioPSE_LeadField_BuildMisfitField {
    inherit Module
    constructor {config} {
        set name BuildMisfitField
        set_defaults
    }
    method set_defaults {} {
        global $this-metric
        set $this-metric invCC
        global $this-pvalue
        set $this-pvalue 1
    }
    method make_entry {w text v c} {
        frame $w
        label $w.l -text "$text"
        pack $w.l -side left
	global $v
        entry $w.e -textvariable $v
        bind $w.e <Return> $c
        pack $w.e -side right
    }
    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }
        toplevel $w
        wm minsize $w 300 80
        frame $w.f
        pack $w.f -padx 2 -pady 2 -side top -expand yes
        
        global $this-metric
        frame $w.top -relief groove -borderwidth 2
        make_labeled_radio $w.top.metric "Error Metric" "" top \
                $this-metric \
                {{"Correlation Coefficient" CC } \
                {"Inverse Correlation Coefficient" invCC} \
                {"p Norm" RMS} \
                {"Relative RMS" relRMS}}
	make_entry $w.top.e "p value:" $this-pvalue "$this-c needexecute"
        pack $w.top.metric $w.top.e -side top
        pack $w.top -side top -fill x
    }   
}
