#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR   := Uintah/Components/MPM/ConstitutiveModel

SRCS     += $(SRCDIR)/CompMooneyRivlin.cc $(SRCDIR)/CompNeoHook.cc \
	$(SRCDIR)/CompNeoHookPlas.cc $(SRCDIR)/ConstitutiveModelFactory.cc \
	$(SRCDIR)/ConstitutiveModel.cc \
	$(SRCDIR)/MPMMaterial.cc $(SRCDIR)/ViscoScram.cc \
	$(SRCDIR)/HypoElastic.cc 

PSELIBS := Uintah/Grid

#
# $Log$
# Revision 1.8  2000/12/08 22:16:22  bard
# Added Hypo-Elastic constitutive model.
#
# Revision 1.7  2000/08/22 00:17:49  guilkey
# Took ElasticConstitutiveModel, HyperElasticDamage and ViscoElasticDamage
# out of sub.mk and ConstitutiveModelFactory, since they currently aren't
# up to speed with the UCF.
#
# Revision 1.6  2000/08/21 18:37:41  guilkey
# Initial commit of ViscoScram stuff.  Don't get too excited yet,
# currently these are just cosmetically modified copies of CompNeoHook.
#
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
