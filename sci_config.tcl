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
ConfigBool .quarks -text "Quarks? " -name QUARKS
ConfigBool .irix6 -text "Irix 6? " -name IRIX6

frame .bottom -borderwidth 10
pack .bottom -side bottom

button .bottom.apply -text "Apply" -command "ConfigBase :: apply"
button .bottom.reset -text "Reset" -command "ConfigBase :: reset"
button .bottom.cancel -text "Exit" -command exit
pack .bottom.apply .bottom.reset .bottom.cancel -side left -padx 10 -pady 4 \
	-ipadx 5 -ipady 5

