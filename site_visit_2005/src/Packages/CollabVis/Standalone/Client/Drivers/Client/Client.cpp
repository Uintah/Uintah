#include <Network/NetInterface.h>
#include <Thread/Thread.h>
#include <Rendering/ImageRenderer.h>
#include <Rendering/GeometryRenderer.h>
#include <Rendering/ZTexRenderer.h>
#include <UI/glutUI.h>
#include <Util/ClientProperties.h>

using namespace SemotusVisum;

Renderer * currentRenderer = NULL;
glutUI& theUI = glutUI::getInstance();

void 
setViewingMethod( void * obj, MessageData * message ) {

  SetViewingMethod *svm = (SetViewingMethod *)(message->message);

  /* If this message refers to a renderer that is not our current renderer,
     we delete our current renderer (if applicable), and create a new
     renderer. */
  bool switchRenderers = ( currentRenderer == NULL ||
			   strcasecmp( svm->getRendererName(),
				       currentRenderer->Name() ) );
    
  /* This barfs when the setviewingmethod function is called on
     current renderer.
     if ( switchRenderers && currentRenderer )
     delete currentRenderer;
  */
  
  if ( switchRenderers )
    Log::log( DEBUG, "Switching to renderer " + svm->getRendererName() );
  else 
    Log::log( DEBUG, "New name: " + svm->getRendererName() +
	      " Old name: " + currentRenderer->Name() );
  
  /* Image renderer */
  if ( !strcasecmp( svm->getRendererName(), ImageRenderer::getName() ) ) {

    if ( switchRenderers ) {
      Log::log( DEBUG, "Switching to image rendering!" );
      
      currentRenderer = scinew ImageRenderer();
      
      /* Set renderer callbacks */
      theUI.setRenderer( currentRenderer );
      theUI.setRenderer( ImageRenderer::getName() );
      
      /* Set the current size */
      currentRenderer->setSize( theUI.getWidth()-glutUI::LIGHT_WINDOW_WIDTH,
				theUI.getHeight()-glutUI::GEOM_WINDOW_HEIGHT );

      theUI.hideGeomControls();
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
      currentRenderer = scinew GeometryRenderer();
      
      
      /* Set renderer callbacks */
      theUI.setRenderer( currentRenderer );
      Log::log( DEBUG, "New geometry renderer!" );
      theUI.setRenderer( GeometryRenderer::getName() );
      
      /* Initialize */
      Log::log( DEBUG, "Initializing geom renderer..." );
      ((GeometryRenderer *)currentRenderer)->initialize();
      Log::log( DEBUG, "Geom renderer initialized" );
      
      theUI.showGeomControls();
      Log::log( DEBUG, "Showed geom controls" );
      
      /* Set the current size */
      currentRenderer->setSize( theUI.getWidth()-glutUI::LIGHT_WINDOW_WIDTH,
				theUI.getHeight()-glutUI::GEOM_WINDOW_HEIGHT );
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
      currentRenderer = scinew ZTexRenderer();
      
      /* Set renderer callbacks */
      theUI.setRenderer( currentRenderer );
      theUI.setRenderer( ZTexRenderer::getName() );
      
      /* Initialize */
      Log::log( DEBUG, "Initializing ztex renderer..." );
      ((ZTexRenderer *)currentRenderer)->initialize();
      Log::log( DEBUG, "ZTex renderer initialized" );
      
      theUI.showZTexControls( true );
      Log::log( DEBUG, "Shown ZTex controls" );

      /* Set the current size */
      currentRenderer->setSize( theUI.getWidth()-glutUI::LIGHT_WINDOW_WIDTH,
				theUI.getHeight()-glutUI::GEOM_WINDOW_HEIGHT );
    }
    else {
      /* Adjust viewing parameters? */
    }
    
  }
}

int
main( int argc, char ** argv ) {
  /* Initialize some callbacks. */
  NetDispatchManager::getInstance().registerCallback( SET_VIEWING_METHOD,
						      setViewingMethod,
						      NULL,
						      true );

  /* Make the UI */
  theUI.initialize( argc, argv );
  theUI.show();
  
}
