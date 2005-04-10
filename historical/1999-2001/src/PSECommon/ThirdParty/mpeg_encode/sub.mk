#
# Makefile fragment for this subdirectory
# $Id$
#

include $(SRCTOP)/scripts/smallso_prologue.mk

SRCDIR   := PSECommon/ThirdParty/mpeg_encode

ARKSRC = $(SRCDIR)/mpegelib.c $(SRCDIR)/ARKmpeg.c

MP_JPEG_SRCS = $(SRCDIR)/nojpeg.c

MP_PARALLEL_SRCS = $(SRCDIR)/noparallel.fix.c

MP_BASE_SRCS = $(SRCDIR)/mfwddct.c $(SRCDIR)/postdct.c \
	$(SRCDIR)/huff.c $(SRCDIR)/bitio.c $(SRCDIR)/mheaders.c

MP_ENCODE_SRCS = $(SRCDIR)/iframe.c $(SRCDIR)/pframe.c \
	$(SRCDIR)/bframe.c $(SRCDIR)/psearch.c $(SRCDIR)/bsearch.c \
	$(SRCDIR)/block.c

MP_OTHER_SRCS = $(SRCDIR)/subsample.c $(SRCDIR)/param.c \
	$(SRCDIR)/rgbtoycc.c $(SRCDIR)/readframe.c $(SRCDIR)/combine.c \
	$(SRCDIR)/jrevdct.c $(SRCDIR)/frame.c $(SRCDIR)/fsize.c \
	$(SRCDIR)/frametype.c $(SRCDIR)/libpnmrw.c $(SRCDIR)/specifics.c \
	$(SRCDIR)/rate.c $(SRCDIR)/opts.c

SRCS     += $(MP_BASE_SRCS) $(MP_OTHER_SRCS) $(MP_ENCODE_SRCS) \
	$(MP_PARALLEL_SRCS) $(MP_JPEG_SRCS) $(ARKSRC)

PSELIBS :=
LIBS := -lc -lm

include $(SRCTOP)/scripts/smallso_epilogue.mk

#
# $Log$
# Revision 1.1  2001/01/17 20:09:29  kuzimmer
# added Berkely mpeg_encode library
#
# Revision 1.2  2000/03/20 19:36:29  sparker
# Added VPATH support
#
# Revision 1.1  2000/03/17 09:26:11  sparker
# New makefile scheme: sub.mk instead of Makefile.in
# Use XML-based files for module repository
# Plus many other changes to make these two things work
#
