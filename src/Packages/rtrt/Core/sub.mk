# Makefile fragment for this subdirectory
include $(SCIRUN_SCRIPTS)/smallso_prologue.mk


SRCDIR   := Packages/rtrt/Core


SRCS += $(SRCDIR)/Worker.cc \
	$(SRCDIR)/gl_test.cc \
	$(SRCDIR)/VolumeVis.cc \
	$(SRCDIR)/Dpy.cc \
	$(SRCDIR)/Scene.cc \
	$(SRCDIR)/Image.cc \
	$(SRCDIR)/Camera.cc \
	$(SRCDIR)/Sphere.cc \
	$(SRCDIR)/Background.cc \
	$(SRCDIR)/Object.cc \
	$(SRCDIR)/Phong.cc \
	$(SRCDIR)/MetalMaterial.cc \
	$(SRCDIR)/PhongMaterial.cc \
	$(SRCDIR)/Material.cc \
	$(SRCDIR)/Point.cc \
	$(SRCDIR)/Transform.cc \
	$(SRCDIR)/LambertianMaterial.cc \
	$(SRCDIR)/CoupledMaterial.cc \
	$(SRCDIR)/DielectricMaterial.cc \
	$(SRCDIR)/Ball.cc \
	$(SRCDIR)/BallAux.cc \
	$(SRCDIR)/BallMath.cc \
	$(SRCDIR)/Light.cc \
	$(SRCDIR)/Group.cc \
	$(SRCDIR)/Rect.cc \
	$(SRCDIR)/Checker.cc \
	$(SRCDIR)/Vector.cc \
	$(SRCDIR)/BBox.cc \
	$(SRCDIR)/Exceptions.cc \
	$(SRCDIR)/Stats.cc \
	$(SRCDIR)/CrowMarble.cc \
	$(SRCDIR)/Noise.cc \
	$(SRCDIR)/Turbulence.cc \
	$(SRCDIR)/Random.cc \
	$(SRCDIR)/MusilRNG.cc \
	$(SRCDIR)/FastNoise.cc \
	$(SRCDIR)/FastTurbulence.cc \
	$(SRCDIR)/Tri.cc \
	$(SRCDIR)/clString.cc \
	$(SRCDIR)/Color.cc \
	$(SRCDIR)/BouncingSphere.cc  \
	$(SRCDIR)/BV1.cc \
	$(SRCDIR)/PerProcessorContext.cc \
	$(SRCDIR)/BV2.cc \
	$(SRCDIR)/Grid.cc \
	$(SRCDIR)/TrivialAllocator.cc \
	$(SRCDIR)/GridSpheres.cc \
	$(SRCDIR)/GridSpheresDpy.cc \
	$(SRCDIR)/Volume.cc \
	$(SRCDIR)/HitCell.cc \
	$(SRCDIR)/CubeRoot.cc \
	$(SRCDIR)/VolumeBrick.cc \
	$(SRCDIR)/GradientCell.cc \
	$(SRCDIR)/HVolumeBrick.cc \
	$(SRCDIR)/HVolumeBrick16.cc \
	$(SRCDIR)/HVolumeBrickColor.cc \
	$(SRCDIR)/HVolumeMaterial.cc \
	$(SRCDIR)/VolumeBase.cc \
	$(SRCDIR)/VolumeDpy.cc \
	$(SRCDIR)/VolumeBrick16.cc \
	$(SRCDIR)/Volume16.cc \
	$(SRCDIR)/MIPHVB16.cc \
	$(SRCDIR)/MIPGroup.cc \
	$(SRCDIR)/TimeObj.cc \
	$(SRCDIR)/CutPlane.cc \
	$(SRCDIR)/PlaneDpy.cc \
	$(SRCDIR)/Context.cc \
	$(SRCDIR)/UVMapping.cc \
	$(SRCDIR)/UVPlane.cc \
	$(SRCDIR)/ImageMaterial.cc \
	$(SRCDIR)/RationalBezier.cc \
	$(SRCDIR)/RationalMesh.cc \
	$(SRCDIR)/Bezier.cc \
	$(SRCDIR)/Mesh.cc \
	$(SRCDIR)/Util.cc \
	$(SRCDIR)/Box.cc \
	$(SRCDIR)/Parallelogram.cc \
	$(SRCDIR)/Speckle.cc \
	$(SRCDIR)/Cylinder.cc \
	$(SRCDIR)/hilbert.cc \
	$(SRCDIR)/Wood.cc \
	$(SRCDIR)/HTVolumeBrick.cc  \
	$(SRCDIR)/Disc.cc

PSELIBS :=  \
	Core/Thread Core/Exceptions Packages/rtrt/visinfo

LIBS := $(GL_LIBS) -lfastm -lm -lelf -lfetchop -lperfex

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk


