
OBJS = readit.o scanner.o parser.o
CCC = CC -LANG:std -woff 1167 -MDupdate Makedepend
LEX = lex -t
YACC = yacc -d -v

.SUFFIXES: .cc 
.cc.o:
	$(CCC) -c $(CCFLAGS) $<

PROGRAM = readit

readit: $(OBJS)
	$(CCC) -o $@ $(CCFLAGS) $(OBJS) $(LIBS)

scanner.cc: scanner.l
	$(LEX) scanner.l > scanner.cc

scanner.o: parser.h

parser.cc parser.h: parser.y
	$(YACC) -d parser.y
	sed 's/getenv()/xxgetenv()/g' < y.tab.c > parser.cc
	rm y.tab.c
	mv y.tab.h parser.h

include Makedepend
