#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR   := Uintah/Components/MPM/ConstitutiveModel

SRCS     += $(SRCDIR)/CompMooneyRivlin.cc $(SRCDIR)/CompNeoHook.cc \
	$(SRCDIR)/CompNeoHookPlas.cc $(SRCDIR)/ConstitutiveModelFactory.cc \
	$(SRCDIR)/ConstitutiveModel.cc $(SRCDIR)/ElasticConstitutiveModel.cc \
	$(SRCDIR)/HyperElasticDamage.cc $(SRCDIR)/ViscoElasticDamage.cc \
	$(SRCDIR)/MPMMaterial.cc 

PSELIBS := Uintah/Grid

#
# $Log$
# Revision 1.5  2000/04/19 05:26:05  sparker
# Implemented new problemSetup/initialization phases
# Simplified DataWarehouse interface (not finished yet)
# Made MPM get through problemSetup, but still not finished
#
# Revision 1.4  2000/03/30 18:31:22  guilkey
# Moved Material base class to Grid directory.  Modified MPMMaterial
# and sub.mk to coincide with these changes.
#
# Revision 1.3  2000/03/24 00:44:33  guilkey
# Added MPMMaterial class, as well as a skeleton Material class, from
# which MPMMaterial is inherited.  The Material class will be filled in
# as it's mission becomes better identified.
#
# Revision 1.2  2000/03/20 17:17:10  sparker
# Made it compile.  There are now several #idef WONT_COMPILE_YET statements.
#
# Revision 1.1  2000/03/17 09:29:34  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
