#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR   := Uintah/Components/MPM/ConstitutiveModel

SRCS     += $(SRCDIR)/CompMooneyRivlin.cc $(SRCDIR)/CompNeoHook.cc \
	$(SRCDIR)/CompNeoHookPlas.cc $(SRCDIR)/$(SRCDIR)Factory.cc \
	$(SRCDIR)/ElasticConstitutiveModel.cc \
	$(SRCDIR)/HyperElasticDamage.cc $(SRCDIR)/ViscoElasticDamage.cc \
	$(SRCDIR)/materials_dat.cc

#
# $Log$
# Revision 1.1  2000/03/17 09:29:34  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
