#!./itcl_wish -f

source ../defaults.tcl
source ../tcl/itcl/library/itcl_mkindex.tcl


itcl_class IndexDir {
    constructor {config} {
	lappend instances $this

	set class [$this info class]
	::rename $this $this-tmp-
	::frame $this -class $class
	::rename $this $this-win-
	::rename $this-tmp- $this

	label $this.label -text $text
	pack $this.label -side left
	frame $this.l
	pack $this.l -side right
	button $this.l.ind -command "$this apply" -text "Make Index"
	pack $this.l.ind -side left

	pack $this -side top -fill x
    }
    method apply {} {
	cd ..
	itcl_mkindex $dir *.tcl *.itcl
	cd scripts
	
	if {[catch {set f [open ../auto.tcl r+]}]} {
	    return ""
	}
	set result ""
	while {[gets $f line] >= 0} {
	    set x [split $line]
	    if {[llength $x] != 3} {
		puts stderr "Error in config.values, line:"
		puts stderr $line
		exit
	    }
	    lappend result [lindex $x 2]
	}
	seek $f 0 end
	if [expr [lsearch $result "\$sci_root/$dir"] > -1] {
	    puts "Updated $dir/tclIndex"
	} else {
	    puts $f "lappend auto_path \$sci_root/$dir"
	    puts "Added $dir/tclIndex"
	}
	close $f
	
    }

    public name ""
    public text
    public dir
}

frame .bottom -borderwidth 10
pack .bottom -side bottom

button .bottom.cancel -text "Exit" -command exit
pack .bottom.cancel -side left -padx 10 -pady 4 -ipadx 5 -ipady 5

label .l -text "Directories: " -foreground red -relief groove -anchor nw
pack .l -side top -anchor nw -fill x

frame .left -relief groove -borderwidth 2
pack .left -side left -anchor nw -fill y -padx 2 -pady 2
set i 0
foreach t {Classlib Comm Constraints Dataflow Datatypes Devices Geom \
	   Geometry Malloc Math Multitask TCL Widgets} {
	IndexDir .left.$i -text "$t: " -name DIR_$t -dir $t
	incr i
}

frame .right -relief groove -borderwidth 2
pack .right -side right -anchor nw -fill y -padx 2 -pady 2
set i 0
foreach t {Contours FEM Fields Matrix Mesh Readers \
	   Salmon Sound Surface Visualization Writers} {
	IndexDir .right.$i -text "Modules/$t: " -name DIR_Modules_$t \
		-dir Modules/$t
	incr i
}

foreach t {sci tcl tk itcl} {
	IndexDir .right.$i -text "tcl/$t: " -name DIR_tcl_$t -dir tcl/$t
	incr i
}
