# Makefile fragment for this subdirectory

SRCDIR   := Dataflow/ThirdParty/mpeg

SRCS     += $(SRCDIR)/mpeg.o $(SRCDIR)/codec.o $(SRCDIR)/huffman.o \
	$(SRCDIR)/io.o $(SRCDIR)/chendct.o $(SRCDIR)/lexer.o \
	$(SRCDIR)/marker.o $(SRCDIR)/me.o $(SRCDIR)/mem.o \
	$(SRCDIR)/stat.o $(SRCDIR)/stream.o $(SRCDIR)/transform.o

