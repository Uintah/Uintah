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

global port_spacing
set port_spacing 18

global port_width
set port_width 13

global port_height
set port_height 7 

global port_light_height
set port_light_height 4

global Color
set Color(Selected) LightSkyBlue2
set Color(Disabled) black
set Color(Compiling) "\#f0e68c"
set Color(Trace) red
set Color(ConnDisabled) gray
set Color(NetworkEditor) "\#036"
set Color(SubnetEditor) purple4 ;#$Color(NetworkEditor)
#set Color(NetworkEditor) "black"
set Color(ErrorFrameBG) $Color(NetworkEditor)
set Color(ErrorFrameFG) white
set Color(IconFadeStart) $Color(NetworkEditor)
set Color(Basecolor) gray
#set Color(Basecolor) darkred

global Subnet
set Subnet(Loading) 0

set basecolor $Color(Basecolor)

. configure -background $basecolor

option add *Frame*background black

option add *Button*padX 1
option add *Button*padY 1

option add *background $basecolor
option add *activeBackground $basecolor
option add *sliderForeground $basecolor
option add *troughColor $basecolor
option add *activeForeground white

option add *Scrollbar*activeBackground $basecolor
option add *Scrollbar*foreground $basecolor
option add *Scrollbar*width .35c
option add *Scale*width .35c

option add *selectBackground "white"
option add *selector red
option add *font "-Adobe-Helvetica-bold-R-Normal--*-120-75-*"
option add *highlightThickness 0

#set blt_library $env(TOP)/src/3rdParty/tcl/blt2.4h/library
source $DataflowTCL/platformSpecific.tcl



