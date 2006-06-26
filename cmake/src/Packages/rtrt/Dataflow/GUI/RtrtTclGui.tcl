
# Override some scirun stuff
set tcl_prompt1 "puts -nonewline \"rtrt> \""
set tcl_prompt2 "puts -nonewline \"rtrt>> \""

itcl_class RTRTGui {

    constructor {config} {
        global $this-nprocs
        set $this-nprocs 1
    }
    
    method modname {} {
	return [string range $this [expr [string last :: $this] + 2] end]
    }

    method printthis {} {
        puts "$this"
    }

    method printnprocs {} {
        global $this-nprocs
        puts "--nprocs = [set $this-nprocs]"
    }

    method printmodname {} {
        puts [eval modname]
    }
}			

proc setrtrtctx {name} {
    puts "setrtrtctx called"
    global rtrtctx;
    set rtrtctx $name
    RTRTGui $rtrtctx
    $rtrtctx printthis
    $rtrtctx printnprocs
}

proc makeRtrtGui {} {
    puts "makeRtrtGui called"

    wm protocol . WM_DELETE_WINDOW { NiceQuit }
    wm minsize . 100 100
    wm title . "RTRT Gui"
#    wm geometry . 400x500

    frame .main

    buildUI .main

    pack .main
}

proc buildUI {w} {
    button $w.hello -text "hello" \
	-command "rtrtgui hello"
    pack $w.hello -anchor w -pady 2

    button $w.quit -text "Quit" \
	-command { NiceQuit }
    pack $w.quit -anchor w

#    buton $w.whatsthis -text "This" -command "puts $this"
#    pack $w.whatsthis -anchor w
}

proc NiceQuit {} {
    puts "NiceQuit called"
    rtrtgui quit
}