#!./itcl_wish -f

source defaults.tcl
source sci_conflib.tcl

ConfigChoice .machine -choices {Linux SGI} -text "Machine: " \
	-name MACHINE
ConfigChoice .variant -choices {Debug Optimized} -text "Variant: " \
	-name VARIANT
ConfigChoice .compiler -choices {CFront Delta GNU} -text "Compiler: " \
	-name COMPILER -file "cm"
ConfigChoice .assertions -choices {0 1 2 3 4} -text "Assertion level: " \
	-name "ASSERTION_LEVEL" -file ch
ConfigBool .opengl -text "OpenGL? " -name OPENGL

frame .bottom -borderwidth 10
pack .bottom -side bottom

button .bottom.apply -text "Apply" -command "ConfigBase :: apply"
button .bottom.reset -text "Reset" -command "ConfigBase :: reset"
button .bottom.cancel -text "Exit" -command exit
pack .bottom.apply .bottom.reset .bottom.cancel -side left -padx 10 -pady 4 \
	-ipadx 5 -ipady 5

label .l -text "Directories: " -foreground red -relief groove -anchor nw
pack .l -side top -anchor nw -fill x

frame .left -relief groove -borderwidth 2
pack .left -side left -anchor nw -fill y -padx 2 -pady 2
set i 0
foreach t {Classlib Comm Constraints Dataflow Datatypes Devices Geom \
	   Geometry Malloc Math Multitask TCL Widgets} {
	ConfigDir .left.$i -text "$t: " -name DIR_$t -dir $t -modname $t
	incr i
}

frame .right -relief groove -borderwidth 2
pack .right -side right -anchor nw -fill y -padx 2 -pady 2
set i 0
foreach t {Contours FEM Fields Matrix Mesh Readers \
	   Salmon Sound Surface Visualization Writers} {
	ConfigDir .right.$i -text "Modules/$t: " -name DIR_Modules_$t \
		-dir Modules/$t -modname Module_$t
	incr i
}

foreach t {sci tcl tk itcl} {
	ConfigDir .right.$i -text "tcl/$t: " -name DIR_tcl_$t \
		-dir tcl/$t -modname tcl_$t
	incr i
}
