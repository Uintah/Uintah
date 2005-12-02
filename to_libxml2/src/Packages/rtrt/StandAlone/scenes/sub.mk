# Makefile fragment for this subdirectory

SRCDIR := Packages/rtrt/StandAlone/scenes

INCLUDES += $(TEEM_INCLUDE)

#RSE stands for RTRT_SCENE_EXTENSION
RSE=mo

SCENES := $(SRCDIR)/0.$(RSE) \
	$(SRCDIR)/obj_reader.$(RSE) \
	$(SRCDIR)/gl.$(RSE) \
	$(SRCDIR)/simple.$(RSE) \
	$(SRCDIR)/simple_spheres.$(RSE) \
	$(SRCDIR)/simple_disc.$(RSE) \
	$(SRCDIR)/instance_spheres.$(RSE) \
	$(SRCDIR)/1.$(RSE) \
	$(SRCDIR)/2.$(RSE) \
	$(SRCDIR)/3.$(RSE) \
	$(SRCDIR)/4.$(RSE) \
	$(SRCDIR)/5.$(RSE) \
	$(SRCDIR)/6.$(RSE) \
	$(SRCDIR)/7.$(RSE) \
	$(SRCDIR)/science-room-min.$(RSE) \
	$(SRCDIR)/3ds.$(RSE) \
	$(SRCDIR)/3dsm_ase.$(RSE) \
	$(SRCDIR)/bunny.$(RSE) \
	$(SRCDIR)/bunnyteapot.$(RSE) \
	$(SRCDIR)/galleon.$(RSE) \
	$(SRCDIR)/htvolumebrick.$(RSE) \
	$(SRCDIR)/hvolumebrick16.$(RSE) \
	$(SRCDIR)/hvolumebrickfloat.$(RSE) \
	$(SRCDIR)/hvolume_uchar.$(RSE) \
	$(SRCDIR)/hvolume_ushort.$(RSE) \
	$(SRCDIR)/flux.$(RSE) \
	$(SRCDIR)/i3d.$(RSE) \
	$(SRCDIR)/miphvb16.$(RSE) \
	$(SRCDIR)/mipvfem.$(RSE) \
	$(SRCDIR)/oldvfem.$(RSE) \
	$(SRCDIR)/original.$(RSE) \
	$(SRCDIR)/spherefile.$(RSE) \
	$(SRCDIR)/teapot.$(RSE) \
	$(SRCDIR)/teapot.rational.$(RSE) \
	$(SRCDIR)/teapot.scene.$(RSE) \
	$(SRCDIR)/t0.$(RSE) \
	$(SRCDIR)/uintahdata.$(RSE) \
	$(SRCDIR)/vfem.$(RSE) \
	$(SRCDIR)/volume_color.$(RSE) \
	$(SRCDIR)/ASE-RTRT.$(RSE)\
	$(SRCDIR)/simple_tri.$(RSE) \
	$(SRCDIR)/terrain.$(RSE) \
	$(SRCDIR)/heightfield.$(RSE) \
	$(SRCDIR)/graphics-museum.$(RSE) \
	$(SRCDIR)/min-museum.$(RSE) \
	$(SRCDIR)/3min-museum.$(RSE) \
	$(SRCDIR)/single-sphere.$(RSE)  \
	$(SRCDIR)/envmap-sphere.$(RSE)  \
	$(SRCDIR)/box3e.$(RSE)  \
	$(SRCDIR)/molecule.$(RSE)  \
	$(SRCDIR)/wine.$(RSE) \
	$(SRCDIR)/FordField.$(RSE) \
	$(SRCDIR)/david.$(RSE) \
	$(SRCDIR)/davidhead.$(RSE)  \
	$(SRCDIR)/pttest.$(RSE) \
	$(SRCDIR)/VolumeVisMod.$(RSE) \
	$(SRCDIR)/VolumeVis2DMod.$(RSE) \
	$(SRCDIR)/VolumeVisRGBAMod.$(RSE) \
	$(SRCDIR)/sketch.$(RSE) \
	$(SRCDIR)/pt_particle.$(RSE) \
	$(SRCDIR)/tstdemo.$(RSE) \
	$(SRCDIR)/living-room2.$(RSE) \

#	$(SRCDIR)/vthorax.$(RSE) \
#	$(SRCDIR)/multihvb.$(RSE) \
#	$(SRCDIR)/hvolumevg.$(RSE) \

#	$(SRCDIR)/figure1.$(RSE) \
#	$(SRCDIR)/david_old.$(RSE) \
#	$(SRCDIR)/ramsey.$(RSE) \
# 	$(SRCDIR)/miptest.$(RSE) \
# 	$(SRCDIR)/venus.$(RSE) \
# 	$(SRCDIR)/crank.$(RSE) 
# 	$(SRCDIR)/cbox.$(RSE) \
#	$(SRCDIR)/graphics-museum-works.$(RSE) \
# 	$(SRCDIR)/buddha.$(RSE) \
#	$(SRCDIR)/seaworld-tubes2.$(RSE) \
#	$(SRCDIR)/seaworld-tubes3.$(RSE) \
#	$(SRCDIR)/sphere-room2.$(RSE) \
#	$(SRCDIR)/seaworld-tubes.$(RSE) \
#	$(SRCDIR)/cutdemo.$(RSE)  \
#	$(SRCDIR)/basic-sea.$(RSE)  \
#	$(SRCDIR)/spinning_instance_demo.$(RSE) \
#	$(SRCDIR)/stadium.$(RSE) \
#	$(SRCDIR)/multi-scene.$(RSE) \
#	$(SRCDIR)/dtiglyph.$(RSE) \
#	$(SRCDIR)/science-room.$(RSE) \
#	$(SRCDIR)/science-room-full.$(RSE) \

ifeq ($(findstring Uintah, $(LOAD_PACKAGE)),Uintah)
SCENES += \
	$(SRCDIR)/uintahisosurface.$(RSE) \
	$(SRCDIR)/uintahparticle2.$(RSE)
endif

####################################################
## Add all the scenes to be built
ALLTARGETS := $(ALLTARGETS) $(SCENES)

LIBS := $(GL_LIBRARY) $(FASTM_LIBRARY) $(M_LIBRARY) $(THREAD_LIBRARY) $(PERFEX_LIBRARY) $(X_LIBRARY)

###################################################
# Specific targets for scenes that need something other than default
# build rules.

RTRT_PSELIBS := $(OOGL_LIBRARY) $(GLUI_LIBRARY) $(GLUT_LIBRARY) $(GL_LIBRARY) $(X_LIBRARY) $(FASTM_LIBRARY) $(M_LIBRARY) $(THREAD_LIBRARY) $(PERFEX_LIBRARY) $(SOUND_LIBRARY) $(LAPACKMP_LIBRARY)

RTRT_ULIBS = -lPackages_rtrt_Core -lPackages_Uintah_Core_DataArchive -lPackages_Uintah_Core_Grid -lCore_Persistent -lCore_Geometry -lCore_Containers -lCore_Exceptions -lDataflow_Comm -lCore_XMLUtil $(SCI_THIRDPARTY_LIBRARY) $(XML_LIBRARY) $(MPI_LIBRARY) -lCore_Malloc -lCore_Thread  $(GLUI_LIBRARY) $(GLUT_LIBRARY) $(LAPACKMP_LIBRARY) $(M_LIBRARY)

$(SRCDIR)/uintahparticle2.$(RSE): $(SRCDIR)/uintahparticle2.o
	$(CXX) -o $@ $(LDFLAGS) $(SOFLAGS) $(patsubst %.$(RSE),%.o,$(filter %.$(RSE),$@)) $(RTRT_ULIBS)

$(SRCDIR)/uintahisosurface.$(RSE): $(SRCDIR)/uintahisosurface.o
	$(CXX) -o $@ $(LDFLAGS) $(SOFLAGS) $(patsubst %.$(RSE),%.o,$(filter %.$(RSE),$@)) $(RTRT_ULIBS)

#######################################################################

$(SCENES): lib/libPackages_rtrt_Core.$(SO_OR_A_FILE) lib/libPackages_rtrt_Core_PathTracer.$(SO_OR_A_FILE) lib/libCore_Persistent.$(SO_OR_A_FILE) lib/libCore_Geometry.$(SO_OR_A_FILE) lib/libCore_Malloc.$(SO_OR_A_FILE) lib/libCore_Thread.$(SO_OR_A_FILE) lib/libCore_Exceptions.$(SO_OR_A_FILE) lib/libCore_Math.$(SO_OR_A_FILE)
%.$(RSE): %.o
	echo "Building rtrt scenefile $@"
	rm -f $@
	$(CXX) -o $@ $(LDFLAGS) $(SOFLAGS) $(patsubst %.$(RSE),%.o,$(filter %.$(RSE),$@)) -lPackages_rtrt_Core -lPackages_rtrt_Core_PathTracer -lCore_Exceptions -lCore_Geometry -lCore_Persistent -lCore_Malloc -lCore_Thread -lCore_Exceptions -lCore_Math $(SCI_THIRDPARTY_LIBRARY) $(TEEM_LIBRARY) $(XML_LIBRARY) $(M_LIBRARY) $(OOGL_LIBRARY) $(GLUI_LIBRARY) $(GLUT_LIBRARY) $(LAPACKMP_LIBRARY)
#	ln -fs `basename $@` $(patsubst %.$(RSE),%.mo,$(filter %.$(RSE),$@))

CLEANPROGS := $(CLEANPROGS) $(SCENES) 

