CC=g++

XALAN = -L/home/sci/darbyb/work/itk/Xalan/xml-xalan/c/lib
XERCES = -L/home/sci/darbyb/work/SCIRun_Thirdparty/1.7.2/1.7/Linux/gcc-3.2-32bit/lib
LLDLIBS = $(XERCES) -lxerces-c.21 $(XALAN) -lxalan-c1_4_0 -lm
INCLUDE = -I/usr/local/include/ -I/home/sci/darbyb/work/itk/Xalan/xml-xalan/c/src -I/home/sci/darbyb/work/SCIRun_Thirdparty/1.7.2/1.7/Linux/gcc-3.2-32bit/include

SOURCE = main.cc Generator.cc Parser.cc Translator.cc
EXEC = generate

${EXEC}:	${SOURCE}
		${CC} ${SOURCE} ${INCLUDE} ${LLDLIBS} -o ${EXEC}

clean:
	-rm -f *.o $(TARGETS) core *~

