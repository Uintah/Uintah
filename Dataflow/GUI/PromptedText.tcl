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
