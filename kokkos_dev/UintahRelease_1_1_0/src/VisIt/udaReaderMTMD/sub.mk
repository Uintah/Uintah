# 
# 
# The MIT License
# 
# Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
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

# Makefile fragment for the VisIt directory

# 'SRCDIR' get's overwritten when the actual rules execute in make, so
# I use my VISIT_SRCDIR to avoid this from happening...

ifeq ($(BUILD_VISIT),yes)

VISIT_SRCDIR := VisIt/udaReaderMTMD

# Force the make system to do the visit_stuff:

ALLTARGETS := $(ALLTARGETS) visit_stuff

# 

#
# List of the .h, .C, and .xml files in the src side that need to be linked on the bin side.
#
links_to_create := $(addprefix $(OBJTOP_ABS)/$(VISIT_SRCDIR)/, $(notdir $(wildcard $(SRCTOP)/$(VISIT_SRCDIR)/*.? $(SRCTOP)/$(VISIT_SRCDIR)/*.xml)))

visit_stuff : $(links_to_create) ${VISIT_SRCDIR}/Makefile.visit ${VISIT_SRCDIR}/avtudaReaderMTMDFileFormat.C
	@cd ${VISIT_SRCDIR}; \
           make -f Makefile.visit 

#
#  The '-f' for the 'ln' command should not be necessary (as the 'ln'
#  should only be run once), but if for some reason make tried to
#  re-run this command and re-create the links, the '-f' will avoid an
#  error code.
#
$(links_to_create) :
	@echo "Creating symbolic link to $@... this occurs only one time."
	@ln -fs $(SRCTOP_ABS)/$(VISIT_SRCDIR)/`basename $@` $@

#
# This creates the VisIt Makefile.  Have to move your Makefile out of
# the way, rename VisIt's Makefile, and put ours back.
#
${VISIT_SRCDIR}/Makefile.visit : lib/libStandAlone_tools_uda2vis.${SO_OR_A_FILE}
	@echo create visit makefile
	@cd ${VISIT_SRCDIR}; \
	  rm -f Makefile.visit; \
	  mv Makefile Makefile.sci; \
	  ${VISIT_INSTALL_DIR}/src/bin/xml2makefile -private -clobber udaReaderMTMD.xml; \
	  sed -e "s,^CPPFLAGS=,CPPFLAGS=-I${OBJTOP_ABS} -I${SRCTOP_ABS} ," Makefile > Makefile.visit; \
	  mv Makefile.sci Makefile

#
# The following says that the .C file is dependent on the .C.in file.  If the .C file is out of date,
# then run the 'config.status' command with the argument "--file=...C:...test.C.in, which will
# use the src_side/.../test...C.in file to generate the bin_side/.../...C file.
#
${VISIT_SRCDIR}/avtudaReaderMTMDFileFormat.C: ${VISIT_SRCDIR}/testavtudaReaderMTMDFileFormat.C.in
	config.status --file=${OBJTOP_ABS}/$@:$<

endif