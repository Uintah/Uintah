#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  License for the specific language governing rights and limitations under
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
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


