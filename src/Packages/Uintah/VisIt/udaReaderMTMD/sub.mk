
# Makefile fragment for the Packages/Uintah/VisIt directory

# 'SRCDIR' get's overwritten when the actual rules execute in make, so
# I use my VISIT_SRCDIR to avoid this from happening...

VISIT_SRCDIR := Packages/Uintah/VisIt/udaReaderMTMD

# Force the make system to do the visit_stuff:

ALLTARGETS := $(ALLTARGETS) visit_stuff

# 

links_to_create := $(addprefix $(OBJTOP_ABS)/$(VISIT_SRCDIR)/, $(notdir $(wildcard $(SRCTOP)/$(VISIT_SRCDIR)/*.? $(SRCTOP)/$(VISIT_SRCDIR)/*.xml)))

visit_stuff : $(links_to_create) ${VISIT_SRCDIR}/Makefile.visit ${VISIT_SRCDIR}/avtudaReaderMTMDFileFormat.C
	@cd ${VISIT_SRCDIR}; \
           make -f Makefile.visit 

$(links_to_create) :
	@echo "Creating symbolic link to $@... this occurs only one time."
	@ln -s $(SRCTOP)/$(VISIT_SRCDIR)/`basename $@` $@

${VISIT_SRCDIR}/Makefile.visit : lib/libPackages_Uintah_StandAlone_tools_uda2vis.so
	@echo create visit makefile
	@cd ${VISIT_SRCDIR}; \
	  rm -f Makefile.visit; \
	  mv Makefile Makefile.sci; \
	  ${VISIT_INSTALL_DIR}/src/bin/xml2makefile -private -clobber udaReaderMTMD.xml; \
	  sed -e "s,^CPPFLAGS=,CPPFLAGS=-I${OBJTOP_ABS} -I${SRCTOP} ," Makefile > Makefile.visit; \
	  mv Makefile.sci Makefile

${VISIT_SRCDIR}/avtudaReaderMTMDFileFormat.C: ${VISIT_SRCDIR}/testavtudaReaderMTMDFileFormat.C.in
	@( Here="`pwd`" ; cd ${OBJTOP} ; Top="`pwd`" ; CONFIG_FILES=`echo $${Here} | sed -e "s%^"$${Top}"/%%" -e "s%^"$${Top}"%%"`${OBJTOP_ABS}/${VISIT_SRCDIR}/avtudaReaderMTMDFileFormat.C CONFIG_HEADERS="" ./config.status )

