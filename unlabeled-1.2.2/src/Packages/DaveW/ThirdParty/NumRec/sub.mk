#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := DaveW/ThirdParty/NumRec

SRCS     += $(SRCDIR)/amoeba.cc $(SRCDIR)/amotry.cc $(SRCDIR)/banbks.cc \
	$(SRCDIR)/bandec.cc $(SRCDIR)/banmprv.cc $(SRCDIR)/banmul.cc \
	$(SRCDIR)/protozoa.cc \
	$(SRCDIR)/dpythag.cc $(SRCDIR)/dsvbksb.cc $(SRCDIR)/dsvdcmp.cc \
	$(SRCDIR)/linbcg.cc $(SRCDIR)/nrutil.cc $(SRCDIR)/plgndr.cc

PSELIBS := DaveW/ThirdParty/OldLinAlg

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.2.2.5  2000/11/01 23:02:30  mcole
# Fix for previous merge from trunk
#
# Revision 1.2.2.3  2000/10/26 14:02:25  moulding
# merge HEAD into FIELD_REDESIGN
#
# Revision 1.4  2000/10/29 04:05:05  dmw
# updates to make downhill simplex more interactive
#
# Revision 1.3  2000/09/20 21:27:27  zyp
# Added protozoa
#
# Revision 1.2  2000/03/20 19:36:31  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:26:14  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
