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
