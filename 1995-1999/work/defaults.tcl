#!/usr/local/bin/wish -f

. configure -background pink2
option add *background pink2
option add *sliderForeground pink2
option add *activeBackground pink2
option add *troughColor pink3
option add *activeForeground mediumblue
option add *Scrollbar*activeBackground SteelBlue2
option add *Scrollbar*foreground plum2
option add *Scrollbar*width .35c
option add *Scale*width .35c
option add *selectBackground "light blue"
option add *selector red
option add *font "-Adobe-Helvetica-bold-R-Normal--*-120-75-*"
option add *Button*padX 1
option add *Button*padY 1
option add *highlightThickness 0

set tcl_prompt1 "puts -nonewline \"scirun> \""
set tcl_prompt2 "puts -nonewline \"scirun>> \""
