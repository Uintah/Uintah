#############################################################################
# Visual Tcl v1.20 Project
#

#################################
# GLOBAL VARIABLES
#
global widget; 

#################################
# USER DEFINED PROCEDURES
#
proc init {argc argv} {

}

init $argc $argv


proc {main} {argc argv} {

}

proc {Window} {args} {
global vTcl
    set cmd [lindex $args 0]
    set name [lindex $args 1]
    set newname [lindex $args 2]
    set rest [lrange $args 3 end]
    if {$name == "" || $cmd == ""} {return}
    if {$newname == ""} {
        set newname $name
    }
    set exists [winfo exists $newname]
    switch $cmd {
        show {
            if {$exists == "1" && $name != "."} {wm deiconify $name; return}
            if {[info procs vTclWindow(pre)$name] != ""} {
                eval "vTclWindow(pre)$name $newname $rest"
            }
            if {[info procs vTclWindow$name] != ""} {
                eval "vTclWindow$name $newname $rest"
            }
            if {[info procs vTclWindow(post)$name] != ""} {
                eval "vTclWindow(post)$name $newname $rest"
            }
        }
        hide    { if $exists {wm withdraw $newname; return} }
        iconify { if $exists {wm iconify $newname; return} }
        destroy { if $exists {destroy $newname; return} }
    }
}

#################################
# VTCL GENERATED GUI PROCEDURES
#

proc vTclWindow. {base} {
    if {$base == ""} {
        set base .
    }
    ###################
    # CREATING WIDGETS
    ###################
    wm focusmodel $base passive
    wm geometry $base 1x1+0+0
    wm maxsize $base 1265 994
    wm minsize $base 1 1
    wm overrideredirect $base 0
    wm resizable $base 1 1
    wm withdraw $base
    wm title $base "vt.tcl"
    ###################
    # SETTING GEOMETRY
    ###################
}

proc vTclWindow.top17 {base} {
    if {$base == ""} {
        set base .top17
    }
    if {[winfo exists $base]} {
        wm deiconify $base; return
    }
    ###################
    # CREATING WIDGETS
    ###################
    toplevel $base -class Toplevel
    wm focusmodel $base passive
    wm geometry $base 856x455+3+155
    wm maxsize $base 1265 994
    wm minsize $base 1 1
    wm overrideredirect $base 0
    wm resizable $base 1 1
    wm deiconify $base
    wm title $base "Component Wizard"
    frame $base.fra18 \
        -borderwidth 2 -height 75 -relief groove -width 125 
    frame $base.fra18.fra26 \
        -borderwidth 2 -height 75 -relief groove -width 125 
    canvas $base.fra18.fra26.can32 \
        -background {#036} -borderwidth 2 -height 207 -relief sunken \
        -width 296 
    checkbutton $base.fra18.fra26.che34 \
        -anchor w -text {Has GUI} -variable che34 
    button $base.fra18.fra26.but36 \
        -padx 9 -pady 3 -text {Add port} 
    checkbutton $base.fra18.fra26.che47 \
        -text {Has dynamic port} -variable che47 
    button $base.fra18.fra26.but69 \
        -padx 9 -pady 3 -text {Del port} 
    frame $base.fra18.fra28 \
        -borderwidth 2 -height 75 -relief groove -width 125 
    frame $base.fra18.fra28.fra44 \
        -borderwidth 2 -height 75 -relief groove -width 125 
    entry $base.fra18.fra28.fra44.ent48 
    entry $base.fra18.fra28.fra44.ent49
    entry $base.fra18.fra28.fra44.ent50
    text $base.fra18.fra28.fra44.tex51
    label $base.fra18.fra28.fra44.lab52 \
        -anchor e -borderwidth 1 -text Author(s): 
    label $base.fra18.fra28.fra44.lab53 \
        -anchor e -borderwidth 1 -text Summary: 
    label $base.fra18.fra28.fra44.lab54 \
        -anchor e -borderwidth 1 -text {Example .sr file:} 
    label $base.fra18.fra28.fra44.lab55 \
        -anchor e -borderwidth 1 -text Description: 
    label $base.fra18.fra28.lab45 \
        -borderwidth 1 -text Overview 
    frame $base.fra18.fra28.fra56 \
        -borderwidth 2 -height 75 -relief groove -width 125 
    listbox $base.fra18.fra28.fra56.lis58 
    label $base.fra18.fra28.fra56.lab60 \
        -borderwidth 1 -text Plans 
    listbox $base.fra18.fra28.fra56.lis61 
    label $base.fra18.fra28.fra56.lab62 \
        -borderwidth 1 -text Steps 
    text $base.fra18.fra28.fra56.tex63
    label $base.fra18.fra28.fra56.lab64 \
        -borderwidth 1 -text {Step description} 
    button $base.fra18.fra28.fra56.but65 \
        -padx 9 -pady 3 -text Add 
    button $base.fra18.fra28.fra56.but66 \
        -padx 9 -pady 3 -text Del 
    button $base.fra18.fra28.fra56.but67 \
        -padx 9 -pady 3 -text Add 
    button $base.fra18.fra28.fra56.but68 \
        -padx 9 -pady 3 -text Del 
    label $base.fra18.fra28.lab57 \
        -borderwidth 1 -text {Testing procedure} 
    label $base.fra18.lab70 \
        -borderwidth 1 -text {Operational attributes} 
    label $base.fra18.lab71 \
        -borderwidth 1 -text Design 
    label $base.lab23 \
        -borderwidth 1 -text {Component Specification} 
    button $base.but38 \
        -padx 9 -pady 3 -text {Open specification} 
    button $base.but39 \
        -padx 9 -pady 3 -text {Save specification} 
    button $base.but40 \
        -padx 9 -pady 3 -text {Create component} 
    button $base.but41 \
        -padx 9 -pady 3 -text Cancel 
    ###################
    # SETTING GEOMETRY
    ###################
    place $base.fra18 \
        -x 5 -y 20 -width 845 -height 390 -anchor nw -bordermode ignore 
    place $base.fra18.fra26 \
        -x 5 -y 15 -width 315 -height 370 -anchor nw -bordermode ignore 
    place $base.fra18.fra26.can32 \
        -x 5 -y 10 -width 302 -height 213 -anchor nw -bordermode ignore 
    place $base.fra18.fra26.che34 \
        -x 5 -y 230 -width 86 -height 22 -anchor nw -bordermode ignore 
    place $base.fra18.fra26.but36 \
        -x 5 -y 255 -width 72 -height 26 -anchor nw -bordermode ignore 
    place $base.fra18.fra26.che47 \
        -x 90 -y 230 -width 136 -height 22 -anchor nw -bordermode ignore 
    place $base.fra18.fra26.but69 \
        -x 85 -y 255 -anchor nw -bordermode ignore 
    place $base.fra18.fra28 \
        -x 325 -y 15 -width 515 -height 370 -anchor nw -bordermode ignore 
    place $base.fra18.fra28.fra44 \
        -x 5 -y 15 -width 505 -height 170 -anchor nw -bordermode ignore 
    place $base.fra18.fra28.fra44.ent48 \
        -x 110 -y 10 -width 388 -height 22 -anchor nw -bordermode ignore 
    place $base.fra18.fra28.fra44.ent49 \
        -x 110 -y 35 -width 388 -height 22 -anchor nw -bordermode ignore 
    place $base.fra18.fra28.fra44.ent50 \
        -x 110 -y 60 -width 388 -height 22 -anchor nw -bordermode ignore 
    place $base.fra18.fra28.fra44.tex51 \
        -x 110 -y 85 -width 388 -height 80 -anchor nw -bordermode ignore 
    place $base.fra18.fra28.fra44.lab52 \
        -x 5 -y 15 -width 106 -height 18 -anchor nw -bordermode ignore 
    place $base.fra18.fra28.fra44.lab53 \
        -x 15 -y 35 -width 96 -height 18 -anchor nw -bordermode ignore 
    place $base.fra18.fra28.fra44.lab54 \
        -x 5 -y 60 -width 106 -height 18 -anchor nw -bordermode ignore 
    place $base.fra18.fra28.fra44.lab55 \
        -x 5 -y 85 -width 106 -height 18 -anchor nw -bordermode ignore 
    place $base.fra18.fra28.lab45 \
        -x 20 -y 5 -width 66 -height 18 -anchor nw -bordermode ignore 
    place $base.fra18.fra28.fra56 \
        -x 5 -y 200 -width 505 -height 165 -anchor nw -bordermode ignore 
    place $base.fra18.fra28.fra56.lis58 \
        -x 5 -y 25 -width 73 -height 101 -anchor nw -bordermode ignore 
    place $base.fra18.fra28.fra56.lab60 \
        -x 5 -y 5 -width 71 -height 18 -anchor nw -bordermode ignore 
    place $base.fra18.fra28.fra56.lis61 \
        -x 80 -y 25 -width 73 -height 101 -anchor nw -bordermode ignore 
    place $base.fra18.fra28.fra56.lab62 \
        -x 80 -y 5 -width 71 -height 18 -anchor nw -bordermode ignore 
    place $base.fra18.fra28.fra56.tex63 \
        -x 155 -y 25 -width 343 -height 135 -anchor nw -bordermode ignore 
    place $base.fra18.fra28.fra56.lab64 \
        -x 265 -y 5 -width 111 -height 18 -anchor nw -bordermode ignore 
    place $base.fra18.fra28.fra56.but65 \
        -x 5 -y 130 -width 37 -height 26 -anchor nw -bordermode ignore 
    place $base.fra18.fra28.fra56.but66 \
        -x 45 -y 130 -width 32 -height 26 -anchor nw -bordermode ignore 
    place $base.fra18.fra28.fra56.but67 \
        -x 80 -y 130 -width 37 -height 26 -anchor nw -bordermode ignore 
    place $base.fra18.fra28.fra56.but68 \
        -x 120 -y 130 -width 32 -height 26 -anchor nw -bordermode ignore 
    place $base.fra18.fra28.lab57 \
        -x 20 -y 190 -width 116 -height 18 -anchor nw -bordermode ignore 
    place $base.fra18.lab70 \
        -x 15 -y 5 -width 141 -height 18 -anchor nw -bordermode ignore 
    place $base.fra18.lab71 \
        -x 340 -y 5 -width 51 -height 18 -anchor nw -bordermode ignore 
    place $base.lab23 \
        -x 20 -y 10 -width 151 -height 18 -anchor nw -bordermode ignore 
    place $base.but38 \
        -x 5 -y 420 -width 125 -height 26 -anchor nw -bordermode ignore 
    place $base.but39 \
        -x 130 -y 420 -width 125 -height 26 -anchor nw -bordermode ignore 
    place $base.but40 \
        -x 265 -y 420 -width 128 -height 26 -anchor nw -bordermode ignore 
    place $base.but41 \
        -x 405 -y 420 -anchor nw -bordermode ignore 
}

Window show .
Window show .top17

main $argc $argv
