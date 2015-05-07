
CC=g++
CFLAGS=-g

all : StackTrace StackTrace.symbols.sorted

StackTrace.o : StackTrace.cc
	$(CC) $(CFLAGS) -c $<

StackTrace : StackTrace.o
	$(CC) $(CFLAGS) -o StackTrace $<

StackTrace.symbols.sorted : StackTrace
	nm StackTrace | grep -v "      U " | grep -v "       w " | c++filt | sort > StackTrace.symbols.sorted

clean:
	rm -f StackTrace *.o *~ StackTrace.symbols.sorted
