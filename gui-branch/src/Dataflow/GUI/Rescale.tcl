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

# an example field manipulation insertable UI

proc fm_ui_Rescale { p modname } {
    frame $p.l1
    frame $p.l2
    frame $p.l3
    frame $p.l4
    pack $p.l1 $p.l2 $p.l3 $p.l4 -padx 2 -pady 2
    
    label $p.l1.l -text "x factor: " -width 15
    entry $p.l1.factor -width 30
    label $p.l2.l -text "y factor: " -width 15
    entry $p.l2.factor -width 30
    label $p.l3.l -text "z factor: " -width 15
    entry $p.l3.factor -width 30
    pack $p.l1.l $p.l1.factor \
	 $p.l2.l $p.l2.factor \
	 $p.l3.l $p.l3.factor -side left -padx 2 -pady 2
}