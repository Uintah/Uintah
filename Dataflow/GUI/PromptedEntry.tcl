proc prompted_entry {t {prompt ""} args} {
    global $t.real_text
    global $t.prompt
    set $t.prompt $prompt
    eval {entry $t} $args
    set $t.real_text [$t get]
    prompted_entry_add_prompt $t
    bindtags $t [concat [bindtags $t] PromptedEntry] 
    bind PromptedEntry <FocusOut> {
        set %W.real_text [%W get]
        prompted_entry_add_prompt %W
    }
    bind PromptedEntry <FocusIn> {
        if {[string compare "" [set %W.real_text]] == 0} {
            %W config -foreground black
            %W delete 0 [expr "1 + [string len [%W get]]"]
        } else {
        }
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
