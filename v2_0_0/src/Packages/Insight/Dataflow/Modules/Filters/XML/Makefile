#####################################
#
# Makefile for generating Insight Code
#
#####################################

# Need to have full path as variable
SRCDIR := /Users/yarden/projects/SCIRun/src/Packages/Insight
CATEGORY := Filters

OBJECTS := $(patsubst sci_%.xml, ../%.cc, $(wildcard sci_*.xml))
FILES := $(patsubst sci_%.xml, %, $(wildcard sci_*.xml))

all:	${OBJECTS}


# generate program must already be compiled
../%.cc: 	sci_%.xml
	/Users/yarden/projects/SCIRun/osx/Packages/Insight/Core/CodeGenerator/generate ${SRCDIR} ${CATEGORY} $(patsubst ../%.cc, %, $@)
