proc prompted_text {t {prompt ""} args} {
    global $t.real_text
    global $t.prompt
    set $t.prompt $prompt
    eval {text $t} $args
    set $t.real_text [$t get 1.0 end]
    prompted_text_add_prompt $t
    bindtags $t [concat [bindtags $t] PromptedText] 
    bind PromptedText <FocusOut> {
        set %W.real_text [%W get 1.0 end]
        prompted_text_add_prompt %W
    }
    bind PromptedText <FocusIn> {
        if {[string compare "\n" [set %W.real_text]] == 0} {
            %W tag delete prompt_color
            %W delete 1.0 end
        } else {
        }
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
