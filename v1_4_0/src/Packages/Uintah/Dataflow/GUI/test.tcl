frame .f -borderwidth 5
pack .f -side top -expand yes -fill both

wm geometry . 200x200+0+0
scrollbar .f.vscroll -relief sunken \
	-command ".f.c yview"

canvas .f.c -scrollregion "0 0 1000 1000" \
	-yscrollcommand ".f.vscroll set" \
	-bg blue

pack .f.vscroll -side right -fill y
pack .f.c -expand yes -fill both

.f.c create line 20 20 180 990 

bind .f.c <ButtonPress-1> "updateCanvas .f.c %x %y"

proc updateCanvas { canv x y } {
    set xx [ $canv canvasx $x ]
    set yy [ $canv canvasy $y ]

    puts "$x $y -- $xx $yy"
}