TARGET = mp3d
CCFLAGS = -O2
LDFLAGS = ../macros/sgi/libsgi.a -lm -lmpc
MACROS = ../macros/sgi/c.m4.sgi 

OBJS = adv.o setup.o mp3d.o

.SUFFIXES:
.SUFFIXES: .o .c .U .h .H
.c.o: ; cc -c $(CCFLAGS) $*.c -o $*.o
.U.c: ; m4 $(MACROS) $*.U >$*.c
.H.h: ; m4 $(MACROS) $*.H >$*.h

$(TARGET): $(OBJS)
	cc $(OBJS) -o $(TARGET) $(LDFLAGS)

adv.c mp3d.c setup.c: common.h

clean:
	-rm $(TARGET) *.[coh]
