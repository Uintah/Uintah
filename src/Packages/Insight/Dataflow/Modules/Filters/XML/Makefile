#####################################
#
# Makefile for generating Insight Code
#
#####################################

# Need to have full path as variable
SRCDIR := /home/sci/darbyb/work/Yarden/SCIRun/src/Packages/Insight
CATEGORY := Filters

OBJECTS := $(patsubst sci_%.xml, ../%.cc, $(wildcard sci_*.xml))
FILES := $(patsubst sci_%.xml, %, $(wildcard sci_*.xml))

all:	${OBJECTS}


# generate program must already be compiled
../%.cc: 	sci_%.xml
	/home/sci/darbyb/work/Yarden/SCIRun/src/Packages/Insight/Core/CodeGenerator/generate ${SRCDIR} ${CATEGORY} $(patsubst ../%.cc, %, $@)