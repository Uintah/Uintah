itcl_class SCIRun_Fields_ManageFieldSet {
    inherit Module
    constructor {config} {
        set name ManageFieldSet
        set_defaults
    }

    method set_defaults {} {
    }

    method execute {} {
	$this-c needexecute
    }

    method ui {} {

        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }
        toplevel $w

	frame $w.sel

	blt::hiertable $w.sel.h  -width 0 \
		-yscrollcommand "$w.sel.vs set" \
		-xscrollcommand "$w.sel.hs set" \
		-selectmode multiple \
		-hideroot yes 

	$w.sel.h column configure treeView -text "View"
	#$w.sel.h column insert 0 pre
	#$w.sel.h column insert end post
	$w.sel.h column configure treeView -hide no -edit no

	$w.sel.h text configure -selectborderwidth 0
	scrollbar $w.sel.vs -orient vertical   -command "$w.sel.h yview"
	scrollbar $w.sel.hs -orient horizontal -command "$w.sel.h xview"

	grid columnconfigure $w.sel 0 -weight 1
	grid rowconfigure    $w.sel 0 -weight 1

	grid $w.sel.h $w.sel.vs $w.sel.hs
	grid config $w.sel.h  -column 0 -row 0 -columnspan 1 -rowspan 1 -sticky "s\nsew"
	grid config $w.sel.hs -column 0 -row 1 -columnspan 1 -rowspan 1 -sticky "ew"
	grid config $w.sel.vs -column 1 -row 0 -columnspan 1 -rowspan 1 -sticky "ns"

	pack $w.sel -side top

	
	$w.sel.h configure -icons ""
	$w.sel.h configure -activeicons ""
	$w.sel.h insert end a b c "a d" "a e" "b f" "b f g"
	# $w.sel.h entry configure a -data { pre a-pre post a-post }
	# $w.sel.h entry configure b -data { pre b-pre post b-post }
	# $w.sel.h entry configure c -data { pre c-pre post c-post }
	# $w.sel.h entry configure "a d" -data { pre d-pre post d-post }
	# $w.sel.h entry configure "a e" -data { pre e-pre post e-post }
	# $w.sel.h entry configure "b f" -data { pre f-pre post f-post }
	# $w.sel.h entry configure "b f g" -data { pre g-pre post g-post }

        button $w.execute -text "Execute" -command "$this execute"
        pack $w.execute -side bottom
    }
}


