# Makefile fragment for this subdirectory

SRCDIR := Packages/rtrt/StandAlone/scenes

SCENES := $(SRCDIR)/0.mo \
	$(SRCDIR)/VolumeVisMod.mo \
	$(SRCDIR)/gl.mo \
	$(SRCDIR)/1.mo \
	$(SRCDIR)/2.mo \
	$(SRCDIR)/3.mo \
	$(SRCDIR)/4.mo \
	$(SRCDIR)/5.mo \
	$(SRCDIR)/6.mo \
	$(SRCDIR)/7.mo \
	$(SRCDIR)/3ds.mo \
	$(SRCDIR)/bunny.mo \
	$(SRCDIR)/bunnyteapot.mo \
	$(SRCDIR)/galleon.mo \
	$(SRCDIR)/htvolumebrick.mo \
	$(SRCDIR)/hvolumebrick16.mo \
	$(SRCDIR)/hvolumebrickfloat.mo \
	$(SRCDIR)/flux.mo \
	$(SRCDIR)/i3d.mo \
	$(SRCDIR)/miphvb16.mo \
	$(SRCDIR)/mipvfem.mo \
	$(SRCDIR)/multihvb.mo \
	$(SRCDIR)/oldvfem.mo \
	$(SRCDIR)/spherefile.mo \
	$(SRCDIR)/teapot.mo \
	$(SRCDIR)/teapot.rational.mo \
	$(SRCDIR)/teapot.scene.mo \
	$(SRCDIR)/vfem.mo \
	$(SRCDIR)/uintahdata.mo \
	$(SRCDIR)/uintahisosurface.mo \
	$(SRCDIR)/uintahparticle2.mo \
	$(SRCDIR)/t0.mo \
	$(SRCDIR)/uintahparticle.mo

#add the scenes to the targets
ALLTARGETS := $(ALLTARGETS) $(SCENES)

RTRT_ULIBS = -lPackages_rtrt_Core -lPackages_Uintah_CCA_Components_DataArchiver -lPackages_Uintah_CCA_Components_MPM -lPackages_Uintah_Core_Grid -lCore_Geometry -lCore_Containers -lCore_Exceptions -lDataflow_Comm -lDataflow_XMLUtil $(XML_LIBRARY) $(MPI_LIBRARY) -lCore_Malloc

$(SRCDIR)/VolumeVisMod.mo: $(SRCDIR)/VolumeVisMod.o
	$(CXX) -o $@ $(LDFLAGS) -shared -Wl,-no_unresolved $(patsubst %.mo,%.o,$(filter %.mo,$@)) $(RTRT_ULIBS) -lnrrd -lair -lbiff -lnrrd -lm

$(SRCDIR)/uintahparticle2.mo: $(SRCDIR)/uintahparticle2.o
	$(CXX) -o $@ $(LDFLAGS) -shared -Wl,-no_unresolved $(patsubst %.mo,%.o,$(filter %.mo,$@)) $(RTRT_ULIBS)

$(SRCDIR)/uintahparticle.mo: $(SRCDIR)/uintahparticle.o
	$(CXX) -o $@ $(LDFLAGS) -shared -Wl,-no_unresolved $(patsubst %.mo,%.o,$(filter %.mo,$@)) $(RTRT_ULIBS)

$(SRCDIR)/uintahisosurface.mo: $(SRCDIR)/uintahisosurface.o
	$(CXX) -o $@ $(LDFLAGS) -shared -Wl,-no_unresolved $(patsubst %.mo,%.o,$(filter %.mo,$@)) $(RTRT_ULIBS)

#$(SRCDIR)/uintahparticle.o: $(SRCDIR)/uintahparticle.cc
#	$(CXX) -c $(CCFLAGS) $<

.SUFFIXES: .mo
.o.mo:
	$(CXX) -o $@ $(LDFLAGS) -shared -Wl,-no_unresolved $(patsubst %.mo,%.o,$(filter %.mo,$@)) -lPackages_rtrt_Core -lCore_Malloc

CLEANPROGS := $(CLEANPROGS) $(SCENES) 

#
#MO_GROUP := Packages/rtrt/StandAlone/rtrt/scenes/general
#ifeq ($(LARGESOS),yes)
#  PSELIBS := Packages/rtrt
#else

#  PSELIBS := \
#	Packages/rtrt/Core \
#	Packages/rtrt/visinfo \
#	Core/Thread \
#	Core/Exceptions

#endif
#LIBS :=

#include $(SCIRUN_SCRIPTS)/rtrt_module.mk


#SRCDIR := Packages/rtrt/StandAlone/rtrt/scenes

#SRCS := $(SRCDIR)/uintahparticle.cc

#MO_GROUP := Packages/rtrt/StandAlone/rtrt/scenes/uintah
#ifeq ($(LARGESOS),yes)
#  PSELIBS := Packages/rtrt
#else

#  PSELIBS := \
#	Packages/rtrt/Core \
#	Packages/Uintah/CCA/Components/DataArchiver \
#	Packages_Uintah_CCA_Components_MPM \
#	Packages_Uintah_Core_Grid \
#	Core_Geometry \
#	Core_Containers \
#	Core/Thread \
#	Core_Exceptions \
#	Dataflow_Comm \
#	Dataflow_XMLUtil

#endif
#LIBS := $(XML_LIBRARY) $(MPI_LIBRARY)

#include $(SCIRUN_SCRIPTS)/rtrt_module.mk


