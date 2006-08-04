# -*-perl-*-

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

sub do_cmd_textasciitilde {
    join('', qq|&tilde;|, $_[0]);
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
