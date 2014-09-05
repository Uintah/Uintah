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

itcl_class BioPSE_Forward_ModifyConductivities {
    inherit Module
    constructor {config} {
        set name ModifyConductivities
        set_defaults
    }


    method set_defaults {} {
    }

    method set_item {name data} {
	set ht .ui[modname].f.h
	
	set ent [$ht insert end $name]
	$ht entry configure $ent -data $data
	return $ent
    }

    method get_item {name} {
	set ht .ui[modname].f.h
	set id [$ht find -name $name]
	return [$ht entry cget $id -data]
    }

    method clear_all {} {
	set ht .ui[modname].f.h
	$ht delete 0
    }

    method isopen {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    return "open"
	}
	return "closed"
    }

    method ui {} {

        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }
        toplevel $w

	frame $w.f

	blt::hiertable $w.f.h -width 0 \
		-yscrollcommand "$w.f.vs set" \
		-xscrollcommand "$w.f.hs set" \
		-allowduplicates yes \
		-selectmode single \
		-icons "" -activeicons "" \
		-hideroot yes -linewidth 0

	$w.f.h column configure treeView -hide yes
	$w.f.h column insert end \
		Material Scale C00 C01 C02 C10 C11 C12 C20 C21 C22
	$w.f.h column configure \
		Material Scale C00 C01 C02 C10 C11 C12 C20 C21 C22 -edit yes
	$w.f.h column configure Material -justify right

	$w.f.h text configure -selectborderwidth 0
	scrollbar $w.f.vs -orient vertical   -command "$w.f.h yview"
	scrollbar $w.f.hs -orient horizontal -command "$w.f.h xview"

	grid columnconfigure $w.f 0 -weight 1
	grid rowconfigure    $w.f 0 -weight 1

	grid $w.f.h $w.f.vs $w.f.hs
	grid config $w.f.h  -column 0 -row 0 \
		-columnspan 1 -rowspan 1 -sticky "s\nsew"
	grid config $w.f.hs -column 0 -row 1 \
		-columnspan 1 -rowspan 1 -sticky "ew"
	grid config $w.f.vs -column 1 -row 0 \
		-columnspan 1 -rowspan 1 -sticky "ns"


	focus $w.f.h
	bind $w.f.h <ButtonPress-1> "$w.f.h text get %x %y; focus $w.f.h.edit"

	frame $w.controls

        button $w.controls.execute -text "Execute" \
	    -command "$this-c needexecute"
        button $w.controls.addnew -text "Add New" -command "$this-c addnew"

	pack $w.controls.execute $w.controls.addnew -side left -fill x
	pack $w.f $w.controls -side top

	$this-c redo_gui
    }
}


