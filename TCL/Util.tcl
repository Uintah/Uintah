
#
# Utility functions
#



#
# Convenience routine for making radiobuttons
#

proc make_labeled_radio {root labeltext command packside variable list} {
    frame $root
    label $root.label -text $labeltext
    pack $root.label -padx 2 -side $packside
    set i 0
    foreach t $list {
	set name [lindex $t 0]
	set value $name
	if {[llength $t] == 2} {
	    set value [lindex $t 1]
	}
	radiobutton $root.$i -text "$name" -variable $variable \
		-value $value -command $command
	pack $root.$i -side $packside -anchor nw
	incr i
    }
}
