#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := SCIRun/Modules/Image

SRCS     += $(SRCDIR)/Binop.cc $(SRCDIR)/Edge.cc $(SRCDIR)/FFT.cc \
	$(SRCDIR)/FFTImage.cc $(SRCDIR)/FilterImage.cc \
	$(SRCDIR)/Gauss.cc $(SRCDIR)/Hist.cc $(SRCDIR)/HistEq.cc \
	$(SRCDIR)/IFFT.cc $(SRCDIR)/IFFTImage.cc \
	$(SRCDIR)/ImageConvolve.cc $(SRCDIR)/ImageGen.cc \
	$(SRCDIR)/ImageSel.cc $(SRCDIR)/ImageToGeom.cc \
	$(SRCDIR)/Noise.cc $(SRCDIR)/PMFilterImage.cc \
	$(SRCDIR)/Radon.cc $(SRCDIR)/Segment.cc $(SRCDIR)/Sharpen.cc \
	$(SRCDIR)/Snakes.cc $(SRCDIR)/Subsample.cc $(SRCDIR)/Ted.cc \
	$(SRCDIR)/Threshold.cc $(SRCDIR)/Transforms.cc $(SRCDIR)/Turk.cc \
	$(SRCDIR)/Unop.cc $(SRCDIR)/ViewHist.cc $(SRCDIR)/WhiteNoiseImage.cc

PSELIBS := SCIRun/Datatypes/Image PSECore/Dataflow PSECore/Datatypes \
	SCICore/Datatypes SCICore/Persistent SCICore/Exceptions \
	SCICore/TclInterface SCICore/Containers SCICore/Thread \
	SCICore/Math SCICore/TkExtensions
LIBS := $(TK_LIBRARY) $(GL_LIBS) -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.2  2000/03/20 19:38:12  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:29:08  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
#
