
proc read_config {} {
    if {[catch {set f [open config.values r]}]} {
	return ""
    }
    set result ""
    while {[gets $f line] >= 0} {
	set x [split $line]
	if {[llength $x] != 2} {
	    puts stderr "Error in config.values, line:"
	    puts stderr $line
	    exit
	}
	lappend result [lindex $x 0]
	lappend result [lindex $x 1]
    }
    close $f
    return $result
}

set rebuild_all_symlinks 0

itcl_class ConfigBase {
    constructor {config} {
	lappend instances $this

	set value [ConfigBase :: lookup $name]
	global $this-value
	set $this-value $value
	set orig_value $value
    }
    common instances
    proc reset {} {
	foreach t $instances {
	    $t do_reset
	}
    }

    common all_values [read_config]
    proc lookup {id} {
	set idx [lsearch $all_values $id]
	if {$idx == -1} {
	    return "<none>";
	}
	incr idx
	return [lindex $all_values $idx]
    }

    protected orig_value
    method do_reset {} {
	global $this-value
	set $this-value $orig_value
    }

    proc apply {} {
	if {[.machine changed] || [.variant changed] || [.compiler changed]} {
	    set rebuild_all_symlinks 1
	} else {
	    set rebuild_all_symlinks 0
	}
	set cv [open config.values w]
	set ch [open config.h.tmp w]
	set cm [open config_imake.h.tmp w]
	foreach t $instances {
	    $t do_apply $cv $ch $cm
	}
	close $cv
	close $ch
	close $cm

	set failed [catch "exec diff config.h.tmp config.h > /dev/null"]
	set need_make 0
	if {$failed} {
	    exec mv config.h.tmp config.h
	    set need_make 1
	} else {
	    exec rm config.h.tmp
	}
	set failed [catch "exec diff config_imake.h.tmp config_imake.h > /dev/null"]
	if {$failed} {
	    exec mv config_imake.h.tmp config_imake.h
	    if {![file exists Makefile]} {
		exec xmkmf >@ stdout 2>@ stderr
	    }
	    if {$rebuild_all_symlinks} {		
		tk_dialog .makewin "Make Advice" "You need to do a \"make clean\" and a \"make World\"" \
			"" 0 Ok
	    } else {
		tk_dialog .makewin "Make Advice" "You need to do a \"make World\"" \
			"" 0 Ok
	    }
	} else {
	    exec rm config_imake.h.tmp
	    if {$need_make} {
		tk_dialog .makewin "Make Advice" "You need to do a \"make\"" \
			"" 0 Ok
	    }
	}
    }
    method do_apply {cv ch cm} {
	global $this-value
	puts $cv "$name [set $this-value]"
	$this do_apply_spec $ch $cm
    }

    public name ""
    public file "both" {
	if {$file != "both" && $file != "ch" && $file != "cm"} {
	    puts "bad value for -file: $file"
	    exit
        }
    }

    proc vname {opt} {
	set mach [.machine get]
	if {$opt} {
	    set var optimized
	} else {
	    set var [.variant get]
	}
	set compiler [.compiler get]
	return "[string tolower $mach]-[string tolower $var]-[string tolower $compiler]"
    }
    method changed {} {
	global $this-value
	if {[set $this-value] != $orig_value} {
	    return 1
	} else {
	    return 0
	}
    }
    method get {} {
	global $this-value
	return [set $this-value]
    }
}

itcl_class ConfigChoice {
    inherit ConfigBase
    constructor {config} {
	ConfigBase::constructor
	set class [$this info class]
	::rename $this $this-tmp-
	::frame $this -class $class
	::rename $this $this-win-
	::rename $this-tmp- $this

	label $this.label -text $text
	pack $this.label -side left

	set initted 1
	$this repack

	pack $this -side top -fill x
	global $this-value
	if {[set $this-value] == "<none>"} {
	    set $this-value [lindex $choices 0]
	}
    }
    destructor {
    }

    method repack {} {
	if {!$initted} {
	    return;
	}
	if {[winfo exists $this.l]} {
	    pack forget $this.l
	}
	frame $this.l
	pack $this.l -side left
	set i 0
	foreach n $choices {
	    radiobutton $this.l.$i -value "$n" -variable $this-value \
		    -text "$n"
	    pack $this.l.$i -side left
	    incr i
	}
    }

    protected initted 0
    public choices "" { $this repack }
    public text ""

    method do_apply_spec {ch cm} {
	global $this-value
	if {$file == "both" || $file == "ch"} {
	    puts $ch "\#define SCI_[set name]_[set $this-value]"
	}
	if {$file == "both" || $file == "cm"} {
	    puts $cm "\#define SCI_[set name]_[set $this-value]"
	}
    }

}

itcl_class ConfigBool {
    inherit ConfigBase
    constructor {config} {
	ConfigBase::constructor
	set class [$this info class]
	::rename $this $this-tmp-
	::frame $this -class $class
	::rename $this $this-win-
	::rename $this-tmp- $this

	label $this.label -text $text
	pack $this.label -side left
	frame $this.l
	pack $this.l -side left
	radiobutton $this.l.true -value true -variable $this-value -text True
	radiobutton $this.l.false -value false -variable $this-value -text False
	pack $this.l.true $this.l.false -side left

	pack $this -side top -fill x
	global $this-value
	if {[set $this-value] == "<none>"} {
	    set $this-value false
	}
    }
    destructor {
    }

    method do_apply_spec {ch cm} {
	global $this-value
	if {[set $this-value]} {
	    if {$file == "both" || $file == "ch"} {
		puts $ch "\#define SCI_$name"
	    }
	    if {$file == "both" || $file == "cm"} {
		puts $cm "\#define SCI_$name"
	    }
	}
    }
    public text
}

itcl_class ConfigDir {
    inherit ConfigBase
    constructor {config} {
	ConfigBase::constructor
	set class [$this info class]
	::rename $this $this-tmp-
	::frame $this -class $class
	::rename $this $this-win-
	::rename $this-tmp- $this


	label $this.label -text $text
	pack $this.label -side left
	frame $this.l
	pack $this.l -side right
	radiobutton $this.l.lib -value lib -variable $this-value \
		-text "Library copy"
	radiobutton $this.l.olib -value olib -variable $this-value \
		-text "Optimized library copy"
	radiobutton $this.l.co -value co -variable $this-value \
		-text "Checked out"
	pack $this.l.lib $this.l.olib $this.l.co -side left

	pack $this -side top -fill x

	global $this-value
	if {[set $this-value] == "<none>"} {
	    set $this-value co
	    set orig_value co
	}
    }

    method do_apply_spec {ch cm} {
	global $this-value
	set value [set $this-value]
	if {$value == "co"} {
	    puts $cm "\#define SCI_$name"
	}
	set rebuild_this_symlinks 0
	if {$value != $orig_value} {
	    if {$value == "co"} {
		# checkout the library
		exec rm -r $dir
		set p [pwd]
		cd [file dirname $dir]
		exec cvs checkout $modname >@ stdout 2>@ stderr
		cd $p
	    } else {
		if {$orig_value == "co"} {
		    # check the library back in
		    exec cvs update $dir >& cvs_output
		    set f [open cvs_output]
		    set l ""
		    while {[gets $f line] >= 0} {
			set c [string index $line 0]
			if {$c == "?" || $c == "M" || $c == "R" || $c == "A"} {
			    lappend l $line
			}
		    }
		    close $f
		    exec rm cvs_output
		    if {$l != ""} {
			set s "There was an error checking in module: $dir"
			foreach t $l {
			    append s "\n" $t
			}
			append s "\n\nFix this error and press Apply again"
			tk_dialog .error "Error on checkin" $s \
				"" 0 Ok
			return
		    }
		    exec rm -r $dir
		}
		set rebuild_this_symlinks 1
	    }
	}

	global rebuild_all_symlinks
	if {$rebuild_this_symlinks || ($value == "lib" && $rebuild_all_symlinks)} {
	    # Rebuild symbolic links
	    catch "exec rm -rf $dir"
	    exec mkdir $dir

	    # Get vname
	    set opt 0
	    if {$value == "olib"} {
		set opt 1
	    }
	    set vname [ConfigBase :: vname $opt]
	    
	    puts "vname is $vname"

	    global env
	    set root $env(SCI_DEVELOP)/work_lib/$vname/work/$dir
	    puts "root is $root"
	    if {[catch {set files [glob $root/Imakefile $root/*.h $root/*.tcl $root/lib*.o $root/lib*.a]}]} {
		set files ""
	    }
	    foreach t $files {
		exec ln -s $t $dir/[file tail $t]
	    }
	}
	set orig_value $value
    }
    public text
    public dir
    public modname
}
