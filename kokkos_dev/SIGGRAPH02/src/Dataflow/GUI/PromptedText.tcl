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

proc prompted_text {t {prompt ""} {command ""} args} {
    global $t.real_text
    global $t.prompt
    global $t.command
    set $t.command $command
    set $t.prompt $prompt
    eval {text $t} $args
    set $t.real_text [$t get 1.0 end]
    prompted_text_add_prompt $t
    bindtags $t [concat [bindtags $t] PromptedText$t] 
    bind PromptedText$t <FocusOut> {
        set %W.real_text [%W get 1.0 end]
        prompted_text_add_prompt %W
    }
    bind PromptedText$t <FocusIn> {
        if {[string compare "\n" [set %W.real_text]] == 0} {
            %W tag delete prompt_color
            %W delete 1.0 end
        } else {
        }
    }
    bind PromptedText$t <Key> {
        set %W.real_text [%W get 1.0 end]
        eval [set %W.command]
    }
}

proc prompted_text_add_prompt {t} {
    global $t.real_text
    global $t.prompt
    if {[string compare "\n" [set $t.real_text]] == 0} {
        $t insert 1.0 [set $t.prompt]
        $t tag add prompt_color 1.0 end
        $t tag configure prompt_color -foreground darkcyan
    }
}

proc set_prompted_text {t {text ""}} {
    if [expr [string compare "" $text] == 0] {
        return;
    }
    global $t
    global $t.real_text
    set $t.real_text $text
    $t delete 1.0 end
    $t insert end "$text"
    $t config -foreground black
}

proc get_prompted_text {t} {
    global $t.real_text
    return [set $t.real_text]
}
