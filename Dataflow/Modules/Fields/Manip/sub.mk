# Makefile for field manipulation code that is compiled and loaded on the fly

SRCS = 		test.cc \
		test2.cc \
		foo.cc
#[ ADD NEW SRC HERE ]

SRLIBS = 	-lCore_Thread \
		-lCore_Malloc \
		-lCore_Exceptions \
#[ ADD NEW LIBS HERE ]

LIBS =
#[ ADD NEW LIBS HERE ]

testLIBS := $(SRLIBS)
test2LIBS := $(SRLIBS) -lCore_Datatypes
fooLIBS := $(SRLIBS)