#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR   := Phil/Modules/Tbon

SRCS     += \
	$(SRCDIR)/TriGroup.cc\
	$(SRCDIR)/Clock.cc\
	$(SRCDIR)/Tbon.cc \
	$(SRCDIR)/Bono.cc\
	$(SRCDIR)/BonoP.cc\
	$(SRCDIR)/BonoCL.cc \
	$(SRCDIR)/ViewGrid.cc\
	$(SRCDIR)/ViewMesh.cc \
	$(SRCDIR)/ElapsedTime.cc\
	$(SRCDIR)/TbonP.cc\
	$(SRCDIR)/TbonCL.cc \
	$(SRCDIR)/TbonUG.cc\
	$(SRCDIR)/TbonOOC1.cc\
	$(SRCDIR)/TbonOOC2.cc \
#[INSERT NEW CODE FILE HERE]

#
# $Log$
# Revision 1.3  2000/10/24 05:57:44  moulding
# new module maker Phase 2: new module maker goes online
#
# These changes clean out the last remnants of the old module maker and
# bring the new module maker online.
#
# Revision 1.2  2000/03/20 19:37:30  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:28:12  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
