
proc uiGenColormap {modid} {
    set w .ui$modid
    if {[winfo exists $w]} {
        raise $w
        return;
    }
    toplevel $w
    frame $w.f 
    pack $w.f -padx 2 -pady 2
    set n "$modid needexecute "

    global nlevels,$modid
    set nlevels,$modid 50

    global ambient_intens,$modid
    set ambient_intens,$modid 0

    global diffuse_intens,$modid
    set diffuse_intens,$modid 1

    global specular_intens,$modid
    set specular_intens,$modid 0

    global spec_percent,$modid
    set spec_percent,$modid 1

    global r,spec_color,$modid g,spec_color,$modid b,spec_color,$modid
    set r,spec_color,$modid .6
    set g,spec_color,$modid .6
    set b,spec_color,$modid .6

    global shininess,$modid
    set shininess,$modid 10

    global map_type,$modid
    set map_type,$modid rainbow

    global rainbow_hue_min,$modid
    set rainbow_hue_min,$modid 0
    global rainbow_hue_max,$modid
    set rainbow_hue_max,$modid 300
    global rainbow_sat,$modid
    set rainbow_sat,$modid 1
    global rainbow_val,$modid
    set rainbow_val,$modid 1
}

