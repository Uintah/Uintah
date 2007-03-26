#include <Network/NetDispatchManager.h>
#include <UI/UserInterface.h>
#include <UI/MiscUI.h>
#include <Logging/Log.h>
#include <Rendering/ImageRenderer.h>
#include <Rendering/GeometryRenderer.h>
#include <Rendering/ZTexRenderer.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

using namespace SemotusVisum;


void 
setViewingMethod( void * obj, MessageData * message ) {

  SetViewingMethod *svm = (SetViewingMethod *)(message->message);

  /* If this message refers to a renderer that is not our current renderer,
     we delete our current renderer (if applicable), and create a new
     renderer. */
  bool switchRenderers = ( UserInterface::renderer() == NULL ||
			   strcasecmp( svm->getRendererName(),
				       UserInterface::renderer()->Name() ) );
  
  /* This barfs when the setviewingmethod function is called on
     current renderer.
     if ( switchRenderers && currentRenderer )
     delete currentRenderer;
  */
  
  if ( switchRenderers )
    Log::log( DEBUG, "Switching to renderer " + svm->getRendererName() );
  else 
    Log::log( DEBUG, "New name: " + svm->getRendererName() +
	      " Old name: " + UserInterface::renderer()->Name() );
  
  /* Image renderer */
  if ( !strcasecmp( svm->getRendererName(), ImageRenderer::getName() ) ) {

    if ( switchRenderers ) {
      Log::log( DEBUG, "Switching to image rendering!" );

      ImageRenderer *tempIR = new ImageRenderer;
      UserInterface::renderer( tempIR );
      UserInterface::renderer()->redraw = &redraw;
      
      /* Set renderer callbacks */
      string result;
      eval( string("setRendererName \"") + ImageRenderer::getName() + "\"",
	    result );
      
      /* Set the current size */
      UserInterface::renderer()->setSize( 640, 512 ); // tmp
    }
    else {
      /* Adjust viewing parameters */
    }
  }

  /* Geometry renderer */
  else if ( !strcasecmp( svm->getRendererName(),
			 GeometryRenderer::getName() ) ) {
    
    if ( switchRenderers ) {
      Log::log( DEBUG, "Switching to geometry rendering!" );

      GeometryRenderer *tempGR = new GeometryRenderer;
      UserInterface::renderer( tempGR );
      UserInterface::renderer()->redraw = &redraw;
      
      /* Set renderer callbacks */
      
      /* Initialize */
      Log::log( DEBUG, "Initializing geom renderer..." );
      //((GeometryRenderer *)UserInterface::renderer())->initialize();

      tempGR->initialize();
      
      Log::log( DEBUG, "Geom renderer initialized" );
      
      /* Set the current size */
      UserInterface::renderer()->setSize( 640, 512 ); //tmp
 
      //UserInterface::lock();
      string result;
      eval( string("setRendererName \"") + GeometryRenderer::getName() + "\"",
	    result );
      eval( "doHome", result );
    }
    else {
      /* Adjust viewing parameters? */
    }
    
  }
  /* ZTex renderer */
  else if ( !strcasecmp( svm->getRendererName(),
			 ZTexRenderer::getName() ) ) {
    
    if ( switchRenderers ) {
      Log::log( DEBUG, "Switching to ztex rendering!" );

      ZTexRenderer *tempZR = new ZTexRenderer;
      UserInterface::renderer( tempZR );
      UserInterface::renderer()->redraw = &redraw;
      
      /* Set renderer callbacks */
      
            
      /* Initialize */
      Log::log( DEBUG, "Initializing ztex renderer..." );
      ((ZTexRenderer *)UserInterface::renderer())->initialize();
      Log::log( DEBUG, "ZTex renderer initialized" );
      
      
      /* Set the current size */
      UserInterface::renderer()->setSize( 640, 512 ); //tmp

   
      string result;
      eval( string("setRendererName \"") + ZTexRenderer::getName() + "\"",
	    result );
      eval( "doHome", result );
      eval( "doZTex", result );
    }
    else {
      /* Adjust viewing parameters? */
    }
    
  }
}

int
main( int argc, char ** argv ) {

  NetDispatchManager::getInstance().registerCallback( SET_VIEWING_METHOD,
						      setViewingMethod,
						      NULL,
						      true );
  Log::log( DEBUG, "In TkClient main" );
  
  UserInterface::initialize( argc, argv );
}
