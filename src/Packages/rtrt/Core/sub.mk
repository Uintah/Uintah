# Makefile fragment for this subdirectory
include $(SCIRUN_SCRIPTS)/smallso_prologue.mk


SRCDIR   := Packages/rtrt/Core


SRCS += $(SRCDIR)/Worker.cc \
	$(SRCDIR)/gl_test.cc \
	$(SRCDIR)/Trigger.cc \
	$(SRCDIR)/VolumeVis.cc \
	$(SRCDIR)/VolumeVis2D.cc \
	$(SRCDIR)/MouseCallBack.cc \
	$(SRCDIR)/GridTris.cc \
	$(SRCDIR)/VolumeVisDpy.cc \
	$(SRCDIR)/Volvis2DDpy.cc \
	$(SRCDIR)/shape.cc \
	$(SRCDIR)/widget.cc \
	$(SRCDIR)/DpyBase.cc \
	$(SRCDIR)/Dpy.cc \
	$(SRCDIR)/Scene.cc \
	$(SRCDIR)/Image.cc \
	$(SRCDIR)/Camera.cc \
	$(SRCDIR)/Stealth.cc \
	$(SRCDIR)/Sphere.cc \
	$(SRCDIR)/Hemisphere.cc \
	$(SRCDIR)/Satellite.cc \
	$(SRCDIR)/SubMaterial.cc \
	$(SRCDIR)/Background.cc \
	$(SRCDIR)/Object.cc \
	$(SRCDIR)/Phong.cc \
	$(SRCDIR)/ASEReader.cc \
	$(SRCDIR)/ObjReader.cc \
	$(SRCDIR)/PLYReader.cc \
	$(SRCDIR)/InvisibleMaterial.cc \
	$(SRCDIR)/CycleMaterial.cc \
	$(SRCDIR)/MetalMaterial.cc \
	$(SRCDIR)/MultiMaterial.cc \
	$(SRCDIR)/MapBlendMaterial.cc \
	$(SRCDIR)/PhongMaterial.cc \
	$(SRCDIR)/Material.cc \
	$(SRCDIR)/LambertianMaterial.cc \
	$(SRCDIR)/LightMaterial.cc \
	$(SRCDIR)/CoupledMaterial.cc \
	$(SRCDIR)/DielectricMaterial.cc \
	$(SRCDIR)/Ball.cc \
	$(SRCDIR)/BallAux.cc \
	$(SRCDIR)/BallMath.cc \
	$(SRCDIR)/Light.cc \
	$(SRCDIR)/Group.cc \
	$(SRCDIR)/Gui.cc \
	$(SRCDIR)/Glyph.cc \
	$(SRCDIR)/Rect.cc \
	$(SRCDIR)/Checker.cc \
	$(SRCDIR)/BBox.cc \
	$(SRCDIR)/Exceptions.cc \
	$(SRCDIR)/EMBMaterial.cc \
	$(SRCDIR)/PortalMaterial.cc \
	$(SRCDIR)/Names.cc \
	$(SRCDIR)/Stats.cc \
	$(SRCDIR)/CrowMarble.cc \
	$(SRCDIR)/Noise.cc \
	$(SRCDIR)/Turbulence.cc \
	$(SRCDIR)/Random.cc \
	$(SRCDIR)/MusilRNG.cc \
	$(SRCDIR)/FastNoise.cc \
	$(SRCDIR)/FastTurbulence.cc \
	$(SRCDIR)/Tri.cc \
	$(SRCDIR)/SmallTri.cc \
	$(SRCDIR)/MeshedTri.cc \
	$(SRCDIR)/Color.cc \
	$(SRCDIR)/BouncingSphere.cc \
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
	$(SRCDIR)/HaloMaterial.cc \
	$(SRCDIR)/HVolumeBrick.cc \
	$(SRCDIR)/HVolumeBrick16.cc \
	$(SRCDIR)/HVolumeBrickColor.cc \
	$(SRCDIR)/HVolumeMaterial.cc \
	$(SRCDIR)/VolumeBase.cc \
	$(SRCDIR)/VolumeDpy.cc \
	$(SRCDIR)/VolumeBrick16.cc \
	$(SRCDIR)/Volume16.cc \
	$(SRCDIR)/Hist2DDpy.cc \
	$(SRCDIR)/VolumeVGBase.cc \
	$(SRCDIR)/HVolumeVG.cc \
	$(SRCDIR)/HVolumeVGspecific.cc \
	$(SRCDIR)/MIPHVB16.cc \
	$(SRCDIR)/TimeObj.cc \
	$(SRCDIR)/Plane.cc \
	$(SRCDIR)/CutPlane.cc \
	$(SRCDIR)/PlaneDpy.cc \
	$(SRCDIR)/MIPGroup.cc \
	$(SRCDIR)/Context.cc  \
	$(SRCDIR)/UVMapping.cc \
	$(SRCDIR)/PPMImage.cc \
	$(SRCDIR)/PNGImage.cc \
	$(SRCDIR)/UVPlane.cc \
	$(SRCDIR)/ImageMaterial.cc \
	$(SRCDIR)/TileImageMaterial.cc \
	$(SRCDIR)/RationalBezier.cc \
	$(SRCDIR)/RationalMesh.cc \
	$(SRCDIR)/Bezier.cc \
	$(SRCDIR)/Mesh.cc \
	$(SRCDIR)/Util.cc \
	$(SRCDIR)/Box.cc \
	$(SRCDIR)/Parallelogram.cc \
	$(SRCDIR)/Speckle.cc \
	$(SRCDIR)/Cylinder.cc \
	$(SRCDIR)/UVCylinder.cc \
	$(SRCDIR)/UVCylinderArc.cc \
	$(SRCDIR)/hilbert.cc \
	$(SRCDIR)/Wood.cc \
	$(SRCDIR)/HTVolumeBrick.cc  \
	$(SRCDIR)/Disc.cc \
	$(SRCDIR)/DiscArc.cc \
	$(SRCDIR)/Ring.cc \
	$(SRCDIR)/imageutils.c \
	$(SRCDIR)/input_ppm.c \
	$(SRCDIR)/write_ppm.c \
	$(SRCDIR)/rgbe.c \
	$(SRCDIR)/SunSky.cc \
	$(SRCDIR)/SHLambertianMaterial.cc \
	$(SRCDIR)/prefilter.cc \
	$(SRCDIR)/Token.cc  \
	$(SRCDIR)/HierarchicalGrid.cc \
	$(SRCDIR)/TexturedTri.cc \
	$(SRCDIR)/TimeCycleMaterial.cc \
	$(SRCDIR)/pcube.c \
	$(SRCDIR)/fpcube.c \
	$(SRCDIR)/templates.cc \
	$(SRCDIR)/BumpMaterial.cc \
	$(SRCDIR)/UVSphere.cc \
	$(SRCDIR)/BumpObject.cc \
	$(SRCDIR)/NormalMapMaterial.cc \
	$(SRCDIR)/SelectableGroup.cc \
	$(SRCDIR)/CutGroup.cc \
	$(SRCDIR)/CutMaterial.cc \
	$(SRCDIR)/CutVolumeDpy.cc \
	$(SRCDIR)/CutPlaneDpy.cc \
	$(SRCDIR)/ColorMap.cc \
	$(SRCDIR)/AirBubble.cc \
	$(SRCDIR)/SeaLambertian.cc \
	$(SRCDIR)/TimeVaryingCheapCaustics.cc \
	$(SRCDIR)/ColorMap.cc \
	$(SRCDIR)/plyfile.cc \
	$(SRCDIR)/VideoMap.cc \
	$(SRCDIR)/SpinningInstance.cc  \
	$(SRCDIR)/DynamicInstance.cc \
	$(SRCDIR)/SpinningInstance.cc \
	$(SRCDIR)/plyfile.c \
	$(SRCDIR)/PhongColorMapMaterial.cc \
	$(SRCDIR)/ColorMapDpy.cc \
	$(SRCDIR)/DynamicInstance.cc \
	$(SRCDIR)/CellGroup.cc \
	$(SRCDIR)/SolidNoise3.cc \
	$(SRCDIR)/PerlinBumpMaterial.cc \
	$(SRCDIR)/Instance.cc \
	$(SRCDIR)/InstanceWrapperObject.cc \
	$(SRCDIR)/Satellite.cc \
	$(SRCDIR)/Glyph.cc \
	$(SRCDIR)/RingSatellite.cc \
	$(SRCDIR)/Grid2.cc \
	$(SRCDIR)/TimeVaryingInstance.cc \
	$(SRCDIR)/TrisReader.cc \
	$(SRCDIR)/Parallelogram2.cc \
	$(SRCDIR)/MIPMaterial.cc \
	$(SRCDIR)/RServer.cc \
	$(SRCDIR)/GeoProbeReader.cc
#	$(SRCDIR)/LumiDpy.cc \
#	$(SRCDIR)/LumiCamera.cc \

SUBDIRS := $(SRCDIR)/Shadows \
#	   $(SRCDIR)/LightField \

include $(SRCTOP)/scripts/recurse.mk

PSELIBS :=  \
	Core/Thread Core/Exceptions Core/Persistent Core/Geometry Packages/rtrt/visinfo 

LIBS := $(GLUI_LIBRARY) $(GLUT_LIBRARY) $(GL_LIBRARY) $(FASTM_LIBRARY) $(M_LIBRARY) $(THREAD_LIBRARY) $(PERFEX_LIBRARY) -lpng -lz

include $(SCIRUN_SCRIPTS)/smallso_epilogue.mk
