# 
# 
# The MIT License
# 
# Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
# Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
# University of Utah.
# 
# License for the specific language governing rights and limitations under
# Permission is hereby granted, free of charge, to any person obtaining a 
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, 
# and/or sell copies of the Software, and to permit persons to whom the 
# Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included 
# in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
# DEALINGS IN THE SOFTWARE.
# 
# 
# 
# 

SRCDIR := tools

FSPEC := $(SRCDIR)/genfspec.pl
FSPEC_ORIG := $(SRCDIR)/fspec.pl

%_fort.h: %.fspec $(FSPEC)
ifeq ($(SCI_MAKE_BE_QUIET),true)
	@perl $(FSPEC) $< $@
	@echo "FSpec:     $@"
else
	perl $(FSPEC) $< $@
endif

$(FSPEC): $(SRCDIR)/stamp-fspec

$(SRCDIR)/stamp-fspec: $(SRCTOP)/$(SRCDIR)/fspec.pl.in ${OBJTOP}/config.status
	@( CONFIG_FILES="tools/fspec.pl" CONFIG_HEADERS="" ./config.status ) 1>&2
	@if cmp -s $(FSPEC) $(FSPEC_ORIG) 2>/dev/null; then echo "$(FSPEC) is unchanged"; else  mv $(FSPEC_ORIG) $(FSPEC); chmod +x $(FSPEC); echo "$(FSPEC) is changed"; fi
	echo timestamp > tools/stamp-fspec
