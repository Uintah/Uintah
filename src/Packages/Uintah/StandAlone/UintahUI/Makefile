# Makefile for GMC User Interface
# 
#------------------------------------------------------------------------------
#
SHELL=/bin/sh
srcdir = .
JAVAC = javac -g 
SOURCE_FILES = \
	ComputeParticleLocPanel.java\
	DecimalField.java \
	DisplayPartDistPanel.java \
	DisplayParticleLocPanel.java \
	GeneralInputsPanel.java \
	HelpAboutFrame.java \
	InputPartDistPanel.java \
	LightWeightCanvas.java \
	MPMInputsPanel.java \
	Packing.java \
	ParticleGeneratePanel.java \
	Particle.java \
	ParticleList.java \
	ParticleLocGeneratePanel.java \
	ParticleSizeDistInputPanel.java \
	ParticleSize.java \
	Point.java \
	UintahGui.java \
	PolygonDouble.java \
	UintahInputPanel.java \
	Voronoi.java \
	WholeNumberField.java

#------------------------------------------------------------------------------

all: UintahGui \
	ComputeParticleLocPanel\
	DecimalField\
	DisplayPartDistPanel\
	DisplayParticleLocPanel\
	GeneralInputsPanel\
	HelpAboutFrame\
	InputPartDistPanel\
	LightWeightCanvas\
	MPMInputsPanel\
	Packing\
	ParticleGeneratePanel\
	Particle\
	ParticleList\
	ParticleLocGeneratePanel\
	ParticleSizeDistInputPanel\
	ParticleSize\
	Point\
	PolygonDouble\
	UintahInputPanel\
	Voronoi\
	WholeNumberField

UI: UintahGui

ComputeParticleLocPanel:
	${JAVAC} ${srcdir}/ComputeParticleLocPanel.java

DecimalField:
	${JAVAC} ${srcdir}/DecimalField.java 

DisplayPartDistPanel:
	${JAVAC} ${srcdir}/DisplayPartDistPanel.java 

DisplayParticleLocPanel:
	${JAVAC} ${srcdir}/DisplayParticleLocPanel.java

GeneralInputsPanel:
	${JAVAC} ${srcdir}/GeneralInputsPanel.java 

HelpAboutFrame:
	${JAVAC} ${srcdir}/HelpAboutFrame.java 

InputPartDistPanel:
	${JAVAC} ${srcdir}/InputPartDistPanel.java 

LightWeightCanvas:
	${JAVAC} ${srcdir}/LightWeightCanvas.java 

MPMInputsPanel:
	${JAVAC} ${srcdir}/MPMInputsPanel.java 

Packing:
	${JAVAC} ${srcdir}/Packing.java 

ParticleGeneratePanel:
	${JAVAC} ${srcdir}/ParticleGeneratePanel.java 

Particle:
	${JAVAC} ${srcdir}/Particle.java 

ParticleList:
	${JAVAC} ${srcdir}/ParticleList.java 

ParticleLocGeneratePanel:
	${JAVAC} ${srcdir}/ParticleLocGeneratePanel.java 

ParticleSizeDistInputPanel:
	${JAVAC} ${srcdir}/ParticleSizeDistInputPanel.java 

ParticleSize:
	${JAVAC} ${srcdir}/ParticleSize.java 

Point:
	${JAVAC} ${srcdir}/Point.java 

UintahGui:
	${JAVAC} ${srcdir}/UintahGui.java 

PolygonDouble:
	${JAVAC} ${srcdir}/PolygonDouble.java 

UintahInputPanel:
	${JAVAC} ${srcdir}/UintahInputPanel.java 

Voronoi:
	${JAVAC} ${srcdir}/Voronoi.java 

WholeNumberField:
	${JAVAC} ${srcdir}/WholeNumberField.java

clean: 
	rm -f *.class

