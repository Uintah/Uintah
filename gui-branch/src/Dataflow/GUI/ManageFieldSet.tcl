#
#  The contents of this file are subject to the University of Utah Public
#  License (the "License"); you may not use this file except in compliance
#  with the License.
#  
#  Software distributed under the License is distributed on an "AS IS"
#  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
#  License for the specific language governing rights and limitations under
#  the License.
#  
#  The Original Source Code is SCIRun, released March 12, 2001.
#  
#  The Original Source Code was developed by the University of Utah.
#  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
#  University of Utah. All Rights Reserved.
#

itcl_class SCIRun_Fields_ManageFieldSet {
    inherit ModuleGui
    constructor {config} {
        set name ManageFieldSet
        set_defaults
    }


    method set_defaults {} {
    }

    method execute {} {
	set $this-state output
	$this-c needexecute
    }

    method add_fitem {index name data} {
	set ht .ui[modname].sel.h
	
	set ent [$ht insert -at $index end $name]
	$ht entry configure $ent -data $data
	return $ent
    }

    method add_sitem {index name} {
	set ht .ui[modname].sel.h
	
	set ent [$ht insert -at $index end $name]
	$ht entry configure $ent -foreground blue4
	return $ent
    }

    method clear_all {} {
	set ht .ui[modname].sel.h
	$ht delete 0
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
		-allowduplicates yes \
		-icons "" -activeicons "" \
		-hideroot yes 

	$w.sel.h column configure treeView -text "Field Name"
	$w.sel.h column insert end Datatype Location
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

	add_sitem 0 "none"
	#clear_all

	#add_fitem a { Datatype double Location edge }
	#add_fitem a { Datatype double Location node }
	#add_fitem a { Datatype double Location face }
	#add_fitem b { Datatype int Location node }
	#add_sitem c
	#add_fitem "c d" { Datatype unknown Location none }
	#$w.sel.h insert end a b c "a d" "a e" "b f" "b f g"
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


