

.SUFFIXES: .tex .ps .dvi
.tex.dvi:
	latex $<

.dvi.ps:
	dvips -o $@ $<

all: design.ps develop.ps user.ps

vdesign: design.dvi
	xdvi design

vdevelop: develop.dvi
	xdvi develop

vuser: user.dvi
	xdvi user
