# Makefile fragment for this subdirectory

SRCDIR := Packages/rtrt/StandAlone/scenes

INCLUDES += $(TEEM_INCLUDE)

SCENES := $(SRCDIR)/0.mo \
	$(SRCDIR)/obj_reader.mo \
	$(SRCDIR)/gl.mo \
	$(SRCDIR)/simple.mo \
	$(SRCDIR)/simple_spheres.mo \
	$(SRCDIR)/simple_disc.mo \
	$(SRCDIR)/instance_spheres.mo \
	$(SRCDIR)/1.mo \
	$(SRCDIR)/2.mo \
	$(SRCDIR)/3.mo \
	$(SRCDIR)/4.mo \
	$(SRCDIR)/5.mo \
	$(SRCDIR)/6.mo \
	$(SRCDIR)/7.mo \
	$(SRCDIR)/science-room-min.mo \
	$(SRCDIR)/3ds.mo \
	$(SRCDIR)/3dsm_ase.mo \
	$(SRCDIR)/bunny.mo \
	$(SRCDIR)/bunnyteapot.mo \
	$(SRCDIR)/galleon.mo \
	$(SRCDIR)/htvolumebrick.mo \
	$(SRCDIR)/hvolumebrick16.mo \
	$(SRCDIR)/hvolumebrickfloat.mo \
	$(SRCDIR)/hvolume_uchar.mo \
	$(SRCDIR)/flux.mo \
	$(SRCDIR)/i3d.mo \
	$(SRCDIR)/miphvb16.mo \
	$(SRCDIR)/mipvfem.mo \
	$(SRCDIR)/oldvfem.mo \
	$(SRCDIR)/original.mo \
	$(SRCDIR)/spherefile.mo \
	$(SRCDIR)/teapot.mo \
	$(SRCDIR)/teapot.rational.mo \
	$(SRCDIR)/teapot.scene.mo \
	$(SRCDIR)/t0.mo \
	$(SRCDIR)/uintahdata.mo \
	$(SRCDIR)/vfem.mo \
	$(SRCDIR)/volume_color.mo \
	$(SRCDIR)/ASE-RTRT.mo\
	$(SRCDIR)/simple_tri.mo \
	$(SRCDIR)/terrain.mo \
	$(SRCDIR)/heightfield.mo \
	$(SRCDIR)/graphics-museum.mo \
	$(SRCDIR)/min-museum.mo \
	$(SRCDIR)/3min-museum.mo \
	$(SRCDIR)/single-sphere.mo  \
	$(SRCDIR)/envmap-sphere.mo  \
	$(SRCDIR)/box3e.mo  \
	$(SRCDIR)/molecule.mo  \
	$(SRCDIR)/wine.mo \
	$(SRCDIR)/FordField.mo \
	$(SRCDIR)/david.mo \
	$(SRCDIR)/davidhead.mo  \
	$(SRCDIR)/pttest.mo \

#	$(SRCDIR)/vthorax.mo \
#	$(SRCDIR)/multihvb.mo \
#	$(SRCDIR)/hvolumevg.mo \

#	$(SRCDIR)/figure1.mo \
#	$(SRCDIR)/david_old.mo \
#	$(SRCDIR)/ramsey.mo \
# 	$(SRCDIR)/miptest.mo \
# 	$(SRCDIR)/venus.mo \
# 	$(SRCDIR)/crank.mo 
# 	$(SRCDIR)/cbox.mo \
#	$(SRCDIR)/graphics-museum-works.mo \
# 	$(SRCDIR)/buddha.mo \
#	$(SRCDIR)/living-room.mo \
#	$(SRCDIR)/seaworld-tubes2.mo \
#	$(SRCDIR)/seaworld-tubes3.mo \
#	$(SRCDIR)/sphere-room2.mo \
#	$(SRCDIR)/seaworld-tubes.mo \
#	$(SRCDIR)/cutdemo.mo  \
#	$(SRCDIR)/basic-sea.mo  \
#	$(SRCDIR)/spinning_instance_demo.mo \
#	$(SRCDIR)/stadium.mo \
#	$(SRCDIR)/multi-scene.mo \


ifeq ($(findstring Uintah, $(LOAD_PACKAGE)),Uintah)
SCENES += \
	$(SRCDIR)/uintahisosurface.mo \
	$(SRCDIR)/uintahparticle2.mo
endif

ifeq ($(findstring Teem, $(LOAD_PACKAGE)),Teem)
SCENES += \
	$(SRCDIR)/VolumeVisMod.mo \
	$(SRCDIR)/VolumeVis2DMod.mo \
	$(SRCDIR)/VolumeVisRGBAMod.mo \
	$(SRCDIR)/sketch.mo \
	$(SRCDIR)/pt_particle.mo \
	$(SRCDIR)/tstdemo.mo \
#	$(SRCDIR)/dtiglyph.mo \
#	$(SRCDIR)/science-room.mo \
#	$(SRCDIR)/science-room-full.mo \

endif


RTRT_DATA_DIR_DEST := $(OBJTOP)/$(SRCDIR)/data
RTRT_DATA_DIR_SRC := $(SRCTOP)/$(SRCDIR)/data

#add the scenes to the targets
#ALLTARGETS := $(ALLTARGETS) $(SCENES) $(RTRT_DATA_DIR_DEST) $(SRCTOP)/blah.data
ALLTARGETS := $(ALLTARGETS) $(SCENES)

#Now we need to create a link to the data directory so that we don't have 
#to copy the data.

#james:
#	@echo "SRCTOP = " $(SRCTOP)
#	@echo "SRCDIR = " $(SRCDIR)
#	@echo "OBJTOP = " $(OBJTOP)
#	@echo "RTRT_DATA_DIR_DEST = " $(RTRT_DATA_DIR_DEST)
#	@echo "RTRT_DATA_DIR_SRC = " $(RTRT_DATA_DIR_SRC)

# $(RTRT_DATA_DIR_DEST): $(RTRT_DATA_DIR_SRC)
# 	ln -s $(shell echo $(foreach t,$(subst /," ",$(SRCDIR))) | sed -e 's,../ ,../,g')$(SRCDIR)/data $@

# $(SRCDIR)/blah.data: $(SRCTOP)/$(SRCDIR)/blah.data
# 	@echo $@

LIBS := $(GL_LIBRARY) $(FASTM_LIBRARY) $(M_LIBRARY) $(THREAD_LIBRARY) $(PERFEX_LIBRARY) $(X_LIBRARY)

RTRT_ULIBS = -lPackages_rtrt_Core -lPackages_Uintah_Core_DataArchive -lPackages_Uintah_Core_Grid -lCore_Persistent -lCore_Geometry -lCore_Containers -lCore_Exceptions -lDataflow_Comm -lDataflow_XMLUtil $(SCI_THIRDPARTY_LIBRARY) $(XML_LIBRARY) $(MPI_LIBRARY) -lCore_Malloc -lCore_Thread $(LAPACKMP_LIBRARY)


$(SRCDIR)/VolumeVisMod.mo: $(SRCDIR)/VolumeVisMod.o
	$(CXX) -o $@ $(LDFLAGS) $(SOFLAGS) $(patsubst %.mo,%.o,$(filter %.mo,$@)) -lPackages_rtrt_Core -lCore_Exceptions -lCore_Geometry -lCore_Persistent -lCore_Malloc -lCore_Thread $(SCI_THIRDPARTY_LIBRARY) $(TEEM_LIBRARY) $(M_LIBRARY) $(GLUI_LIBRARY) $(GLUT_LIBRARY)

$(SRCDIR)/VolumeVis2DMod.mo: $(SRCDIR)/VolumeVis2DMod.o
	$(CXX) -o $@ $(LDFLAGS) $(SOFLAGS) $(patsubst %.mo,%.o,$(filter %.mo,$@)) -lPackages_rtrt_Core -lCore_Exceptions -lCore_Geometry -lCore_Persistent -lCore_Malloc -lCore_Thread $(SCI_THIRDPARTY_LIBRARY) $(TEEM_LIBRARY) $(M_LIBRARY) $(GLUI_LIBRARY) $(GLUT_LIBRARY)

$(SRCDIR)/VolumeVisRGBAMod.mo: $(SRCDIR)/VolumeVisRGBAMod.o
	$(CXX) -o $@ $(LDFLAGS) $(SOFLAGS) $(patsubst %.mo,%.o,$(filter %.mo,$@)) -lPackages_rtrt_Core -lCore_Exceptions -lCore_Geometry -lCore_Persistent -lCore_Malloc -lCore_Thread $(SCI_THIRDPARTY_LIBRARY) $(TEEM_LIBRARY) $(M_LIBRARY) $(GLUI_LIBRARY) $(GLUT_LIBRARY)

$(SRCDIR)/sketch.mo: $(SRCDIR)/sketch.o lib/libPackages_rtrt_Core.$(SO_OR_A_FILE)
	$(CXX) -o $@ $(LDFLAGS) $(SOFLAGS) $(patsubst %.mo,%.o,$(filter %.mo,$@)) -lPackages_rtrt_Core -lCore_Exceptions -lCore_Geometry -lCore_Persistent -lCore_Malloc -lCore_Thread $(SCI_THIRDPARTY_LIBRARY) $(TEEM_LIBRARY) $(M_LIBRARY) $(GLUI_LIBRARY) $(GLUT_LIBRARY)

$(SRCDIR)/pt_particle.mo: $(SRCDIR)/pt_particle.o
	$(CXX) -o $@ $(LDFLAGS) $(SOFLAGS) $(patsubst %.mo,%.o,$(filter %.mo,$@)) -lPackages_rtrt_Core -lCore_Exceptions -lCore_Geometry -lCore_Persistent -lCore_Malloc -lCore_Thread $(SCI_THIRDPARTY_LIBRARY) $(TEEM_LIBRARY) $(M_LIBRARY) $(GLUI_LIBRARY) $(GLUT_LIBRARY) -lm

$(SRCDIR)/dtiglyph.mo: $(SRCDIR)/dtiglyph.o
	$(CXX) -o $@ $(LDFLAGS) $(SOFLAGS) $(patsubst %.mo,%.o,$(filter %.mo,$@)) -lPackages_rtrt_Core -lCore_Exceptions -lCore_Geometry -lCore_Persistent -lCore_Malloc -lCore_Thread -lten -ldye $(SCI_THIRDPARTY_LIBRARY) $(TEEM_LIBRARY) -lhest $(M_LIBRARY) $(GLUI_LIBRARY) $(GLUT_LIBRARY) -lm

$(SRCDIR)/science-room.mo: $(SRCDIR)/science-room.o
	$(CXX) -o $@ $(LDFLAGS) $(SOFLAGS) $(patsubst %.mo,%.o,$(filter %.mo,$@)) -lPackages_rtrt_Core -lCore_Exceptions -lCore_Geometry -lCore_Persistent -lCore_Malloc -lCore_Thread -lten -ldye $(SCI_THIRDPARTY_LIBRARY) $(TEEM_LIBRARY) -lhest $(M_LIBRARY) $(GLUI_LIBRARY) $(GLUT_LIBRARY) -lm

$(SRCDIR)/uintahparticle2.mo: $(SRCDIR)/uintahparticle2.o
	$(CXX) -o $@ $(LDFLAGS) $(SOFLAGS) $(patsubst %.mo,%.o,$(filter %.mo,$@)) $(RTRT_ULIBS) $(GLUI_LIBRARY) $(GLUT_LIBRARY)

$(SRCDIR)/uintahisosurface.mo: $(SRCDIR)/uintahisosurface.o
	$(CXX) -o $@ $(LDFLAGS) $(SOFLAGS) $(patsubst %.mo,%.o,$(filter %.mo,$@)) $(RTRT_ULIBS) $(GLUI_LIBRARY) $(GLUT_LIBRARY)

#$(SRCDIR)/uintahparticle.o: $(SRCDIR)/uintahparticle.cc
#	$(CXX) -c $(CCFLAGS) $<

$(SCENES): lib/libPackages_rtrt_Core.$(SO_OR_A_FILE) lib/libPackages_rtrt_Core_PathTracer.$(SO_OR_A_FILE) lib/libCore_Persistent.$(SO_OR_A_FILE) lib/libCore_Geometry.$(SO_OR_A_FILE) lib/libCore_Malloc.$(SO_OR_A_FILE) lib/libCore_Thread.$(SO_OR_A_FILE) lib/libCore_Exceptions.$(SO_OR_A_FILE) lib/libCore_Math.$(SO_OR_A_FILE)
%.mo: %.o
	rm -f $@
	$(CXX) -o $@ $(LDFLAGS) $(SOFLAGS) $(patsubst %.mo,%.o,$(filter %.mo,$@)) -lPackages_rtrt_Core -lPackages_rtrt_Core_PathTracer -lCore_Exceptions -lCore_Geometry -lCore_Persistent -lCore_Malloc -lCore_Thread -lCore_Exceptions -lCore_Math $(SCI_THIRDPARTY_LIBRARY) $(XML_LIBRARY) $(M_LIBRARY) $(OOGL_LIBRARY) $(GLUI_LIBRARY) $(GLUT_LIBRARY) $(LAPACKMP_LIBRARY)

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


#SUBDIRS := \
#	Packages/rtrt/StandAlone/scenes/data
#include $(SCIRUN_SCRIPTS)/recurse.mk
