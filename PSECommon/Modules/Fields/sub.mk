#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := PSECommon/Modules/Fields

SRCS     += \
	$(SRCDIR)/ClipField.cc\
	$(SRCDIR)/Downsample.cc\
	$(SRCDIR)/ExtractSurfs.cc\
        $(SRCDIR)/FieldFilter.cc\
	$(SRCDIR)/FieldGainCorrect.cc\
	$(SRCDIR)/FieldMedianFilter.cc\
	$(SRCDIR)/FieldRGAug.cc\
	$(SRCDIR)/FieldSeed.cc\
	$(SRCDIR)/Gradient.cc\
	$(SRCDIR)/GradientMagnitude.cc\
	$(SRCDIR)/MergeTensor.cc\
	$(SRCDIR)/OpenGL_Ex.cc\
	$(SRCDIR)/SFRGfile.cc\
	$(SRCDIR)/TracePath.cc\
	$(SRCDIR)/TrainSeg2.cc\
	$(SRCDIR)/TrainSegment.cc\
	$(SRCDIR)/TransformField.cc\
#[INSERT NEW MODULE HERE]

PSELIBS := PSECore/Dataflow PSECore/Datatypes PSECore/Widgets \
	SCICore/Persistent SCICore/Exceptions SCICore/Thread \
	SCICore/Containers SCICore/TclInterface SCICore/Geom \
	SCICore/Datatypes SCICore/Geometry SCICore/TkExtensions
LIBS := $(TK_LIBRARY) $(GL_LIBS) -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.3  2000/06/07 00:11:36  moulding
# made some modifications that will allow the module make to edit and add
# to this file
#
# Revision 1.2  2000/03/20 19:36:57  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:27:01  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
