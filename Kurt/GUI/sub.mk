#
# Makefile fragment for this subdirectory
# $Id$
#

SRCDIR := Kurt/GUI

ALLTARGETS := $(ALLTARGETS) $(SRCDIR)/tclIndex

$(SRCDIR)/tclIndex: $(SRCDIR)/VolVis.tcl $(SRCDIR)/PadField.tcl \
	$(SRCDIR)/TextureVolVis.tcl $(SRCDIR)/GLTextureBuilder.tcl \
	$(SRCDIR)/TexCuttingPlanes.tcl \
	$(SRCDIR)/VisControl.tcl \
	$(SRCDIR)/RescaleColorMapForParticles.tcl \
	$(SRCDIR)/ParticleVis.tcl
	$(SRCTOP)/scripts/createTclIndex $(SRCTOP)/Kurt/GUI

CLEANPROGS := $(CLEANPROGS) $(SRCDIR)/tclIndex

#
# $Log$
# Revision 1.5  2000/05/20 02:31:50  kuzimmer
# Multiple changes for new vis tools
#
# Revision 1.4  2000/05/16 20:54:00  kuzimmer
# added new directory
#
# Revision 1.3  2000/03/21 17:33:25  kuzimmer
# updating volume renderer
#
# Revision 1.2  2000/03/20 19:36:37  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:26:29  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
