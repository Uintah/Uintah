#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR   := Uintah/Components/MPM/ConstitutiveModel

SRCS     += $(SRCDIR)/CompMooneyRivlin.cc $(SRCDIR)/CompNeoHook.cc \
	$(SRCDIR)/CompNeoHookPlas.cc $(SRCDIR)/ConstitutiveModelFactory.cc \
	$(SRCDIR)/ElasticConstitutiveModel.cc \
	$(SRCDIR)/HyperElasticDamage.cc $(SRCDIR)/ViscoElasticDamage.cc

#
# $Log$
# Revision 1.2  2000/03/20 17:17:10  sparker
# Made it compile.  There are now several #idef WONT_COMPILE_YET statements.
#
# Revision 1.1  2000/03/17 09:29:34  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
