# Makefile fragment for this subdirectory

# *** NOTE ***
# Do not remove or modify the comment line:
# #[INSERT NEW ?????? HERE]
# It is required by the module maker to properly edit this file.
# if you want to edit this file by hand, see the "Create A New Module"
# documentation on how to do it correctly.

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := Dataflow/Modules/Image

SRCS     += \
	$(SRCDIR)/Binop.cc\
	$(SRCDIR)/Edge.cc\
	$(SRCDIR)/FFT.cc\
	$(SRCDIR)/FFTImage.cc\
	$(SRCDIR)/FilterImage.cc\
	$(SRCDIR)/Gauss.cc\
	$(SRCDIR)/Hist.cc\
	$(SRCDIR)/HistEq.cc\
	$(SRCDIR)/IFFT.cc\
	$(SRCDIR)/IFFTImage.cc\
	$(SRCDIR)/ImageConvolve.cc\
	$(SRCDIR)/ImageGen.cc\
	$(SRCDIR)/ImageSel.cc\
	$(SRCDIR)/ImageToGeom.cc\
	$(SRCDIR)/Noise.cc\
	$(SRCDIR)/PMFilterImage.cc\
	$(SRCDIR)/Radon.cc\
	$(SRCDIR)/Segment.cc\
	$(SRCDIR)/Sharpen.cc\
	$(SRCDIR)/Snakes.cc\
	$(SRCDIR)/Subsample.cc\
	$(SRCDIR)/Ted.cc\
	$(SRCDIR)/Threshold.cc\
	$(SRCDIR)/Transforms.cc\
	$(SRCDIR)/Turk.cc\
	$(SRCDIR)/Unop.cc\
	$(SRCDIR)/ViewHist.cc\
	$(SRCDIR)/WhiteNoiseImage.cc\
#[INSERT NEW CODE FILE HERE]

PSELIBS := Dataflow/Network Dataflow/Ports Core/Datatypes \
	Core/Datatypes Core/Persistent Core/Exceptions \
	Core/TclInterface Core/Containers Core/Thread \
	Core/Math Core/TkExtensions
LIBS := $(TK_LIBRARY) $(GL_LIBS) -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

