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


set tcl_prompt1 "puts -nonewline \"scirun> \""
set tcl_prompt2 "puts -nonewline \"scirun>> \""

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

