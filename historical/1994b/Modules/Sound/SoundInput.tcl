
proc uiSoundReader {modid} {
    set w .ui$modid
    if {[winfo exists $w]} {
        raise $w
        return;
    }
    toplevel $w
    frame $w.f
    pack $w.f
    checkbutton $w.f.onoff -text "Sound On/Off" -relief flat \
	    -variable onoff,$modid -command "doSoundReaderSwitch $modid"
}

proc doSoundReaderSwitch {modid} {
    if {[set onoff,$modid]} {
	# Turning it on...
	$modid needexecute
    } else {
	# The module will take care of the rest
    }
}
