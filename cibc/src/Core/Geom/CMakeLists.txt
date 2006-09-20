# Sources of Core/Geom classes
SET(Core_Geom_SRCS 
  BBoxCache.cc		                
  CallbackOpenGLContext.cc              
  ColorMap.cc		                
  ColorMapTex.cc	                
  ColorMappedNrrdTextureObj.cc          
  DirectionalLight.cc	                
  DrawInfoOpenGL.cc	                
  FreeType.cc		                
  FreeTypeTextTexture.cc                
  GeomArrows.cc		                
  GeomBillboard.cc	                
  GeomBox.cc		                
  GeomColorMap.cc	                
  GeomCone.cc		                
  GeomContainer.cc	                
  GeomCull.cc	                
  GeomCylinder.cc	                
  GeomDL.cc                             
  GeomDisc.cc		                
  GeomEllipsoid.cc	                
  GeomGrid.cc		                
  GeomGroup.cc		                
  GeomLine.cc		                
  GeomObj.cc                            
  GeomOpenGL.cc		                
  Path.cc		    	        
  GeomPick.cc		                
  GeomPoint.cc		    		
  GeomPolyline.cc	                
  GeomQMesh.cc		                
  GeomQuads.cc		                
  GeomRenderMode.cc	                
  GeomScene.cc		                
  GeomSphere.cc		                
  GeomSticky.cc		                
  GeomSwitch.cc		                
  GeomStippleOccluded.cc		
  GeomTetra.cc		                
  GeomTexSlices.cc	                
  GeomTexRectangle.cc	    		
  GeomText.cc		                
  GeomTimeGroup.cc	                
  GeomTorus.cc		                
  GeomTransform.cc	                
  GeomTri.cc		                
  GeomTriStrip.cc	                
  GeomTriangles.cc	                
  GeomTube.cc		                
  GeomVertexPrim.cc	                
  GeomViewerItem.cc			
  HeadLight.cc		    		
  HistogramTex.cc	    		
  IndexedGroup.cc	    		
  Light.cc		    		
  Lighting.cc		    		
  Material.cc		    		
  NrrdTextureObj.cc	    		
  OpenGLViewport.cc	    		
  OpenGLContext.cc	    		
  OSXOpenGLContext.cc
  Pbuffer.cc		    		
  PointLight.cc		    		
  SpotLight.cc		    		
  ShaderProgramARB.cc                   
  TextRenderer.cc 	    		
  FontManager.cc 	    		
  TextureObj.cc 	    		
  TexSquare.cc		    		
  TimeGrid.cc		    		
  View.cc		    		
  X11Lock.cc		    		
  X11OpenGLContext.cc
  tGrid.cc		    		
  templates.cc
)

ADD_LIBRARY(Core_Geom ${Core_Geom_SRCS})

TARGET_LINK_LIBRARIES( Core_Geom 
  Core_Persistent
  Core_Geometry
  Core_Containers 
  Core_Exceptions
  Core_Datatypes
  Core_Math
  Core_Util
  Core_Thread
  ${FTGL_LIBRARY}
  ${GL_LIBRARY}
  ${PNG_LIBRARY}
  ${ZLIB_LIBRARY}
  ${M_LIBRARY}
  ${TEEM_LIBRARY}
  ${FREETYPE_LIBRARY}
)
#  ${DEFAULT_LIBRARY}


IF(NOT BUILD_SHARED_LIBS)
  ADD_DEFINITIONS(-DBUILD_Core_Geom)
ENDIF(NOT BUILD_SHARED_LIBS)

