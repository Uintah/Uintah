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

proc combo_listbox {p {prompt ""} {command ""}} {
    global $p
    global $p.entry
    global $p.listbox
    global $p.buttons
    global $p.buttons.add
    global $p.buttons.del


    frame $p
    prompted_entry $p.entry $prompt
    iwidgets::scrolledlistbox $p.listbox -vscrollmode static \
        -hscrollmode dynamic -scrollmargin 3 -height 60
    pack $p.entry -side top -fill x -expand true -padx .1c -pady .1c -anchor s
    pack $p.listbox -side top -fill both -expand true -padx .1c -pady .1c

    frame $p.buttons
    button $p.buttons.add -text "Add" \
        -command "global $p.entry.real_text; \
            $p.listbox insert end \"\[set $p.entry.real_text\]\";
            if \[expr \[string compare \"$command\" {}\] != 0\] \{
                eval $command
            \}"
    button $p.buttons.del -text "Delete" \
        -command "while {\[llength \[$p.listbox curselection\]\]} { \
            set i \[lindex \[$p.listbox curselection\] 0\]; \
            $p.listbox delete \[set i\] \[set i\]; \"
            eval $command
        }"

    pack $p.buttons.add -side left -ipadx .1c -ipady .1c -padx .1c -pady .1c \
        -fill both -anchor nw
    pack $p.buttons.del -side left -ipadx .1c -ipady .1c -padx .1c -pady .1c \
        -fill both -anchor nw
    pack $p.buttons -side bottom
}


