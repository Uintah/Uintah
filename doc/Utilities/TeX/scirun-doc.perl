# -*-perl-*-
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

# Extension package for latex2html translator.  This package
# implements commands in scirun-doc.sty.

package main;

sub do_cmd_newmucmd {
    local($_) = @_;
    s/$next_pair_pr_rx//o;
    $& =~ />(.*)</;
    my $name = $1;
    eval "sub do_cmd_$name { \
local(\$_) = \@_; \
s/\$next_pair_pr_rx//o; \
join('', qq|<span class=\"$name\">\$&</span>|,\$_);}";
    s/$next_pair_pr_rx//o;
    $_;
}

sub do_cmd_textless {
    join('', qq|&lt;|, $_[0]);
}

sub do_cmd_textgreater {
    join('', qq|&gt;|, $_[0]);
}

sub do_cmd_xmlstarttag {
    local($_) = @_;
    s/$next_pair_pr_rx//o;
    my $out = '<SPAN class="xmltag">&lt;' . $& . '&gt;</SPAN>';
    join('', $out, $_);
}

sub do_cmd_xmlendtag {
    local($_) = @_;
    s/$next_pair_pr_rx//o;
    my $out = '<SPAN class="xmltag">&lt;/' . $& . '&gt;</SPAN>';
    join('', $out, $_);
}

sub do_cmd_velide {
    join('', qq/&\#x22EE;/, $_[0]);
}

sub do_cmd_acronym {
    local($_) = @_;
    s/$next_pair_pr_rx//o;
    join('',qq|<span class="acronym">$&</span>|, $_);
}

sub do_cmd_note {
    local($_) = @_;
    s/$next_pair_pr_rx//o;
    my($first) = $&;
    my($text, $title);
    if (s/$next_pair_pr_rx//o) {
	$text = $&;
	$title = $first;
    } else {
	$text = $first;
	$title = "Note";
    }
    join('', qq|<div class="note"><span class="notetitle">$title</span><br /><br />$text</div>|, $_);
}


1;
