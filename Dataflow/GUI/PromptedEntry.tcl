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

proc prompted_entry {t {prompt ""} {command ""} args} {
    global $t.real_text
    global $t.prompt
    global $t.command
    set $t.prompt $prompt
    set $t.command $command
    eval {entry $t} $args
    set $t.real_text [$t get]
    prompted_entry_add_prompt $t
    bindtags $t [concat [bindtags $t] PromptedEntry$t] 
    bind PromptedEntry$t <FocusOut> {
        set %W.real_text [%W get]
        prompted_entry_add_prompt %W
    }
    bind PromptedEntry$t <FocusIn> {
        if {[string compare "" [set %W.real_text]] == 0} {
            %W config -foreground black
            %W delete 0 [expr "1 + [string len [%W get]]"]
        } else {
        }
    }
    bind PromptedEntry$t <Key> {
        set %W.real_text [%W get]
        eval [set %W.command]
    }
}

proc prompted_entry_add_prompt {t} {
    global $t.real_text
    global $t.prompt
    if {[string compare "" [set $t.real_text]] == 0} {
        $t insert 1 [set $t.prompt]
        $t config -foreground darkcyan
    }
}

proc set_prompted_entry {t {text ""}} {
    if [expr [string compare "" $text] == 0] {
        return;
    }
    global $t
    global $t.real_text
    set $t.real_text $text
    $t delete 0 end
    $t insert 0 $text
    $t config -foreground black
}

proc get_prompted_entry {t} {
    global $t.real_text
    return [set $t.real_text]
}
