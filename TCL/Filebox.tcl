proc makeFilebox {w var command cancel} {
    global filter,$w path,$w oldpath,$w var,$w oldsel,$w

    set filter,$w "*.*"

    set var,$w ""
    set oldsel,$w ""

    set path,$w [exec pwd]
    set oldpath,$w [set path,$w]

    frame $w.f

    frame $w.f.bro

    frame $w.f.bro.file
    label $w.f.bro.file.filesl -text Files
    listbox $w.f.bro.file.files -relief sunken \
	    -yscrollcommand "$w.f.bro.file.filess1 set" \
	    -xscrollcommand "$w.f.bro.file.filess2 set"
    set files $w.f.bro.file.files
    tk_listboxSingleSelect $files
    bind $w.f.bro.file.files <Button-1> \
	    "fbselect %y $w $files $var"
    bind $w.f.bro.file.files <Double-Button-1> \
	    "fbchoose %y $w $files $var \"$command\""
    scrollbar $w.f.bro.file.filess1 -relief sunken \
	    -foreground plum2 -activeforeground SteelBlue2 \
	    -command "$w.f.bro.file.files yview"
    scrollbar $w.f.bro.file.filess2 -relief sunken -orient horizontal \
	    -foreground plum2 -activeforeground SteelBlue2 \
	    -command "$w.f.bro.file.files xview"
    pack $w.f.bro.file.filesl -in $w.f.bro.file -side top -padx 2 \
	    -pady 2 -anchor w
    pack $w.f.bro.file.filess2 -in $w.f.bro.file -side bottom -padx 2 \
	    -pady 2 -anchor s -fill x
    pack $w.f.bro.file.files -in $w.f.bro.file -side left -padx 2 \
	    -pady 2 -anchor w
    pack $w.f.bro.file.filess1 -in $w.f.bro.file -side right -padx 2 \
	    -pady 2 -anchor e -fill y

    frame $w.f.bro.dir
    label $w.f.bro.dir.dirsl -text Directories
    listbox $w.f.bro.dir.dirs -relief sunken \
	    -yscrollcommand "$w.f.bro.dir.dirss1 set" \
	    -xscrollcommand "$w.f.bro.dir.dirss2 set"
    set dirs $w.f.bro.dir.dirs
    tk_listboxSingleSelect $dirs
    bind $w.f.bro.dir.dirs <Button-1> "fbdirs %y $w $dirs $files"
    bind $w.f.bro.dir.dirs <Double-Button-1> "fbdirs %y $w $dirs $files"
    scrollbar $w.f.bro.dir.dirss1 -relief sunken \
	    -foreground plum2 -activeforeground SteelBlue2 \
	    -command "$w.f.bro.dir.dirs yview"
    scrollbar $w.f.bro.dir.dirss2 -relief sunken -orient horizontal \
	    -foreground plum2 -activeforeground SteelBlue2 \
	    -command "$w.f.bro.dir.dirs xview"
    pack $w.f.bro.dir.dirsl -in $w.f.bro.dir -side top -padx 2 -pady 2 \
	    -anchor w
    pack $w.f.bro.dir.dirss2 -in $w.f.bro.dir -side bottom -padx 2 \
	    -pady 2 -anchor s -fill x
    pack $w.f.bro.dir.dirs -in $w.f.bro.dir -side left -padx 2 -pady 2 \
	    -anchor w
    pack $w.f.bro.dir.dirss1 -in $w.f.bro.dir -side right -padx 2 \
	    -pady 2 -anchor e -fill y

    pack $w.f.bro.dir $w.f.bro.file -in $w.f.bro -side left -padx 2 \
	    -pady 2 -expand 1 -fill x

    frame $w.f.filt
    label $w.f.filt.filtl -text Filter
    entry $w.f.filt.filt -relief sunken -width 40 -textvariable filter,$w
    bind $w.f.filt.filt <Return> "fbupdate $w $dirs $files"
    pack $w.f.filt.filtl -in $w.f.filt -side top -padx 2 -pady 2 \
	    -anchor w
    pack $w.f.filt.filt -in $w.f.filt -side bottom -padx 2 -pady 2 \
	    -anchor w -fill x

    frame $w.f.path
    label $w.f.path.pathl -text Path
    entry $w.f.path.path -relief sunken -width 40 -textvariable path,$w
    bind $w.f.path.path <Return> "fbpath $w $dirs $files"
    pack $w.f.path.pathl -in $w.f.path -side top -padx 2 -pady 2 -anchor w
    pack $w.f.path.path -in $w.f.path -side bottom -padx 2 -pady 2 \
	    -anchor w -fill x

    frame $w.f.sel
    label $w.f.sel.sell -text Selection
    entry $w.f.sel.sel -relief sunken -width 40 -textvariable var,$w
    bind $w.f.sel.sel <Return> "fbsel $w $dirs $files $var \"$command\""
    pack $w.f.sel.sell -in $w.f.sel -side top -padx 2 -pady 2 -anchor w
    pack $w.f.sel.sel -in $w.f.sel -side bottom -padx 2 -pady 2 \
	    -anchor w -fill x

    frame $w.f.but
    button $w.f.but.ok -text OK -command $command
    button $w.f.but.filt -text Filter -command "fbupdate $w $dirs $files"
    button $w.f.but.cancel -text Cancel -command $cancel
    pack $w.f.but.ok -in $w.f.but -side left -padx 2 -pady 2 -anchor w
    pack $w.f.but.cancel -in $w.f.but -side right -padx 2 -pady 2 \
	    -anchor e
    pack $w.f.but.filt -in $w.f.but -side left -padx 2 -pady 8 -expand 1

    pack $w.f.filt $w.f.path $w.f.bro $w.f.sel $w.f.but -in $w.f -side top \
	    -padx 2 -pady 2 -expand 1 -fill both
    pack $w.f

    fbupdate $w $dirs $files
}

proc fbsel {w dirs files var command} {
    global path,$w oldpath,$w oldsel,$w var,$w $var

    if [file isfile [set var,$w]] {
	set $var [set var,$w]
	eval $command
    } elseif [file isdirectory [set var,$w]] {
	fbcd $w [set var,$w] $dirs $files
    } else {
	set var,$w [set oldsel,$w]
    }
}

proc fbdirs {y w dirs files} {
    global path,$w

    set ind [$dirs nearest $y]
    $dirs select from $ind
    set dir [$dirs get $ind]

    if [expr [string compare "." $dir] == 0] {
	return
    } elseif [expr [string compare ".." $dir] == 0] {
	fbcd $w [file dirname [set path,$w]] $dirs $files
    } else {
	fbcd $w [set path,$w]/$dir $dirs $files
    }
}

proc fbpath {w dirs files} {
    global path,$w

    fbcd $w [set path,$w] $dirs $files
}

proc fbcd {w dir dirs files} {
    global path,$w oldpath,$w

    if [file isdirectory $dir] {
	set path,$w $dir
	set oldpath,$w [set path,$w]
	fbupdate $w $dirs $files
    } else {
	set path,$w [set oldpath,$w]
    }
}

proc fbupdate {w dirs files} {
    global filter,$w path,$w

    $dirs delete 0 end
    foreach i [lsort [glob -nocomplain [set path,$w]/.* [set path,$w]/*]] {
	if [file isdirectory $i] {
	    $dirs insert end [file tail $i]
	}
    }

    $files delete 0 end
    foreach i [lsort [glob -nocomplain [set path,$w]/{[set filter,$w]}]] {
	if [file isfile $i] {
	    $files insert end [file tail $i]
	}
    }

    update
}    

proc fbchoose {y w files var command} {
    fbselect $y $w $files $var
    global $var
    eval $command
}

proc fbselect {y w files var} {
    global path,$w var,$w oldsel,$w $var

    set ind [$files nearest $y]
    $files select from $ind
    set var,$w [set path,$w]/[$files get $ind]
    set oldsel,$w [set var,$w]
    set $var [set var,$w]
}
