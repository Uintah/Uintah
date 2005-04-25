/*
 *
 * Renderer: Superclass for renderers
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: January 2001
 *
 */

#include <Rendering/Renderer.h>
#include <Rendering/RenderGroup.h>

namespace SemotusVisum {
namespace Rendering {


RendererHelper::RendererHelper( Renderer *parent ) :
  parent( parent )
#ifndef NO_MBOX
  , mailbox("RenderHelperMailbox", 10 )
#endif
{
}

RendererHelper::~RendererHelper() {

}

void
RendererHelper::run() {
  /* Monitor our mailbox for messages, and call the appropriate
     function when necessary */
  dataItem data;

  int result;
  char buffer[1000];
  Compressor *c = NULL;
  char * output;
  RenderGroup *rg = NULL;

  Log::log( Logging::DEBUG, "[RendererHelper::run] Render helper starting..." );
  
  for (;;) {
    Log::log( Logging::DEBUG, "[RendererHelper::run] Receiving..." );
#ifndef NO_MBOX
    data = mailbox.receive();
#endif
    Log::log( Logging::DEBUG, "[RendererHelper::run] Received" );

    rg = parent->group;

    if ( rg != NULL ) {

      // Preprocess data
      int size = data.getSize();
      char * _data = NULL;
      _data = parent->preprocess( data.getData(), size );
      if ( _data != data.getData() ) {
	data.purge();
	data = dataItem( _data, size, true );
      }
      
      // Compress data
      c = rg->getCompressor();
      
      if ( c != NULL ) { // We have a compressor!
	snprintf( buffer, 1000,
		  "Compressing %d bytes with compressor %s",
		  data.getSize(), c->getName() );
	Log::log( Logging::DEBUG, buffer );
	if ( parent->x * parent->y * parent->bps != data.getSize() )
	  result = c->compress( (DATA*)data.getData(),
				data.getSize(),
				1,
				(DATA **)&output,
				parent->bps );
	else 
	  result = c->compress( (DATA*)data.getData(),
				parent->x,
				parent->y,
				(DATA **)&output,
				parent->bps );
	if ( result < 0 ) {
	  snprintf( buffer, 1000, "Error compressing data: %d", result );
	  Log::log( Logging::ERROR, buffer );
	}
	else {
	  snprintf( buffer, 1000, "Compressed data now %d bytes", result );
	  Log::log( Logging::DEBUG, buffer );
	}

	if ( result < 0 ) {
	  /* Transmit the uncompressed data as a fallback. */
	  parent->sendViewFrame( data.getSize(), data.getData(),
				 parent->x, parent->y );
	}
	else 
	  /* Transmit the compressed data to the client. */
	  parent->sendViewFrame( result, output, parent->x, parent->y,
				 data.getSize() );
	
	data.purge();
	//delete output;
      }
      else {
	Log::log( Logging::DEBUG, "Sending uncompressed data");

	/* Transmit the uncompressed data to the client. */
	parent->sendViewFrame( data.getSize(), data.getData(),
			       parent->x, parent->y );
	data.purge();
	Log::log( Logging::DEBUG, "Purged viewframe data" );
		  
      }
    }
    else 
      data.purge();
  }
}


const char * const
Renderer::name = "Renderer";

const char * const
Renderer::version = "0.0";

void
Renderer::setSuperCallbacks() {
  
  // Compression
  if ( !callbackSet ) {
    compressionCallbackID =
      NetDispatchManager::getInstance().
      registerCallback( COMPRESSION,
			Renderer::compressCallback,
			this,
			true );
    callbackSet = 1;
  }
  
}

void
Renderer::removeSuperCallbacks() {
  if ( callbackSet ) {
    NetDispatchManager::getInstance().deleteCallback( compressionCallbackID );
    compressionCallbackID = -1;
    callbackSet = 0;
  }
}
  
void
Renderer::transmitData( const char * data, const int numBytes ) {

  // If we have no group, return.
  if ( group == NULL )
    return ;
  
  NetInterface &net = NetInterface::getInstance();

  // Transmit data to all clients in our list
  // if we're not in multicast mode.
  if ( group->getMulticastGroup() == NULL ) {
  
    list<char *> clients = group->getClients();
    list<char *>::const_iterator i;
    
    for ( i = clients.begin(); i != clients.end(); i++ ) {
      char buffer[ 1000 ];
      snprintf( buffer, 1000, "Sending %d bytes of viewing data to client %s",
		numBytes, *i );
      Log::log( Logging::DEBUG, buffer );
      if ( net.sendDataToClient( *i,
				 data,
				 (DataTypes)-1,
				 numBytes ) == false ) {
      }
    }
  }
  else {
    // Do things the multicast way...
    std::cerr << "Sending multicast..." << endl;
    if ( net.sendDataToGroup( group->getMulticastGroup(),
			      data,
			      (DataTypes)-1,
			      numBytes ) == false )
      if ( !net.validGroup( group->getMulticastGroup() ) ) 
	group->setMulticastGroup( NULL );
  }
  
  /*
    std::cerr <<  "Verifying client list...";
  //verifyClients();
  std::cerr <<  "Verified" << endl;

  list< clientRenderInfo >::const_iterator i;
  // Transmit data to all clients in our list
  //   if we're not in multicast mode.
  if ( mGroup == NULL ) {
    for (i = clients.begin(); i != clients.end(); i++) {
      char buffer[ 1000 ];
      snprintf( buffer, 1000, "Sending %d bytes of viewing data to client %s",
		numBytes, i->getName() );
      Log::log( Logging::DEBUG, buffer );
      if ( net.sendDataToClient( i->getName(),
				 data,
				 (DataTypes)-1,
				 numBytes ) == false ) {
      }
    }
  }
  else {
    // Otherwise, do things the multicast way.
    std::cerr << "Sending multicast..." << endl;
    if ( net.sendDataToGroup( mGroup,
			      data,
			      (DataTypes)-1,
			      numBytes ) == false )
      if ( !net.validGroup( mGroup ) ) {
	delete mGroup;
	mGroup = NULL;
	// Should we retransmit data in another way?
      }
  }
  */
}

#if 0
void 
Renderer::sendViewFrame( const int size, const char * data,
			 const int origSize,
			 const int indexed, const int replace,
			 const int vertices, const int indices,
			 const int polygons, const int texture ) {
  std::cerr <<"Sending\t";
  Log::log( Logging::DEBUG, data );
  char buffer[ 100 ];

  if ( origSize == -1 )
    snprintf( buffer, 100,
	      "Sending a view frame. Size = %d bytes", size );
  else
    snprintf( buffer, 100,
	      "Sending a view frame. Full size = %d bytes. Original size = %d bytes", size, origSize );
  
  Log::log( Logging::DEBUG, buffer );

  ViewFrame f;

  if ( origSize == -1 )
    f.setSize( (unsigned)size );
  else
    f.setSize( (unsigned)origSize );

  f.setIndexed( indexed );
  f.setReplace( replace );
  f.setVertices( vertices );
  f.setIndices( indices );
  f.setPolygons( polygons );
  f.setTextureSize( texture );
  
  std::cerr << "Finishing building viewframe " << endl;
  f.finish();
  std::cerr << "Finished building viewframe " << endl;
  char * Output = f.getOutput();

  std::cerr << "Got output from viewframe " << (void *)Output << endl;
  // Create a full message!

  Log::log( Logging::DEBUG, Output );
  
  if ( fullMessage == NULL ) {
    fullSize = size + strlen( Output );
    fullMessage = scinew char[ fullSize ];
  }

  if ( (unsigned)fullSize != size + strlen( Output ) ) {
    delete[] fullMessage;
    fullSize = size + strlen( Output );
    fullMessage = scinew char[ fullSize ];
  }

  memset( fullMessage, 0, fullSize );
  memcpy( fullMessage, Output, strlen( Output ) );
  {
      FILE *f = fopen("ZTex.in2", "w");
      fwrite( data, 1, size, f );
      fclose(f);
  }
  char * databuffer = scinew char[ size ];
  memcpy( databuffer, data, size );
  {
    FILE *f = fopen("ZTex.in22", "w");
    fwrite( databuffer, 1, size, f );
    fclose(f);
  }
  memcpy( fullMessage + strlen( Output ), databuffer, size );
  {
    FILE *f = fopen("ZTex.in3", "w");
    fwrite( fullMessage, 1, fullSize, f );
    fclose(f);
  }
  delete databuffer;
  Log::log( Logging::DEBUG, fullMessage );

  std::cerr << "Transmitting" << endl;
  // Now write it to the network.
  transmitData( fullMessage, fullSize );
}
#else

char *
mkOutput( const int size, const int origSize,
	  const int x, const int y,
	  const int indexed, const int replace,
	  const int vertices, const int indices,
	  const int polygons, const int texture ) {
  /* Build a viewframe message */
  ViewFrame * vf = new ViewFrame();

  // Size 
  if ( origSize == -1 ) 
    vf->setSize( size );
  else 
    vf->setSize( origSize );
  
  // Dimensions
  if ( x > 0 && y > 0)
    vf->setDimensions( x, y );
  
  // Geometry info
  vf->setIndexed( indexed );
  vf->setReplace( replace );
  vf->setVertices( vertices );
  vf->setIndices( indices );
  vf->setPolygons( polygons );
  vf->setTextureSize( texture );

  // Finish the message
  vf->finish();

  // Grab the output
  char * msg = strdup(vf->getOutput());
  delete vf;
  return msg;
}

void  
Renderer::sendViewFrame( const int size, const char * data,
			 const int x, const int y,
			 const int origSize,
			 const int offX, const int offY,
			 const int fullX, const int fullY,
			 const char background[3] ) {
  std::cerr << "IR SEND VIEW FRAME" << endl;
  /* Build a viewframe message */
  ViewFrame * vf = new ViewFrame();

  // Size 
  if ( origSize == -1 ) 
    vf->setSize( size );
  else 
    vf->setSize( origSize );
  
  // Dimensions
  if ( x > 0 && y > 0)
    vf->setDimensions( x, y );
  
  // Image info
  vf->setSubimage( offX, offY, fullX, fullY, background );

  // Finish the message
  vf->finish();

  // Grab the output
  char * message = strdup(vf->getOutput());
  delete vf;

  int messageSize = strlen( message );
  
  // Concatenate viewframe with data
  char * fullmessage = scinew char[ messageSize + size ];
  
  int i;
  for ( i = 0; i < messageSize; i++ ) {
    fullmessage[i] = message[i];
  }
  // Verify
  for ( i = 0; i < messageSize; i++ ) {
    if ( fullmessage[i] != message[i] )
      std::cerr <<"1F: " << i << endl;
  }

  int j;
  for ( i = 0, j = messageSize; i < size; i++, j++ ) {
    fullmessage[j] = data[i]; 
  }
  // Verify
  for ( i = 0, j = messageSize; i < size; i++, j++ ) {
    if ( fullmessage[j] != data[i] )
      std::cerr <<"2F: " << i << endl;
  }
  // Verify
  for ( i = 0; i < messageSize; i++ ) {
    if ( fullmessage[i] != message[i] )
      std::cerr <<"3F: " << i << endl;
  }
  // Transmit data.
  transmitData( fullmessage, messageSize + size );

  // Clean up.
  delete message;
  delete fullmessage;    
}


void 
Renderer::sendViewFrame( const int size, const char * data,
			 const int x, const int y,
			 const int origSize,
			 const int indexed, const int replace,
			 const int vertices, const int indices,
			 const int polygons, const int texture ) {

  std::cerr << "NEW SEND VIEW FRAME" << endl;
  char * message = mkOutput( size, origSize, x, y, indexed, replace,
			     vertices, indices, polygons, texture );
  int messageSize = strlen( message );
  
  // Concatenate viewframe with data
  char * fullmessage = scinew char[ messageSize + size ];

  int i;
  for ( i = 0; i < messageSize; i++ ) {
    fullmessage[i] = message[i];
  }
  // Verify
  for ( i = 0; i < messageSize; i++ ) {
    if ( fullmessage[i] != message[i] )
      std::cerr <<"1F: " << i << endl;
  }

  int j;
  for ( i = 0, j = messageSize; i < size; i++, j++ ) {
    fullmessage[j] = data[i]; 
  }
  // Verify
  for ( i = 0, j = messageSize; i < size; i++, j++ ) {
    if ( fullmessage[j] != data[i] )
      std::cerr <<"2F: " << i << endl;
  }
  // Verify
  for ( i = 0; i < messageSize; i++ ) {
    if ( fullmessage[i] != message[i] )
      std::cerr <<"3F: " << i << endl;
  }
  /*  memcpy( fullmessage, message, messageSize );
      memcpy( fullmessage+messageSize, data, size );
  */
  
  // Transmit data.
  transmitData( fullmessage, messageSize + size );

  // Clean up.
  delete message;
  delete fullmessage;
}
    
#endif

void
Renderer::sendViewFrame( const int size, const char * data,
			 const int x, const int y,
			 const int origSize ) {
  //std::cerr <<"Sending\t";
  char buffer[ 100 ];

  if ( origSize == -1 )
    snprintf( buffer, 100,
	      "Sending a view frame. Size = %d bytes", size );
  else
    snprintf( buffer, 100,
	      "Sending a view frame. Full size = %d bytes. Original size = %d bytes", size, origSize );
  
  Log::log( Logging::DEBUG, buffer );

#if 0
  static int g = 0;
  char bb[200];
  sprintf(bb,"frame%d.out", g);
  FILE * ff = fopen(bb, "w" );
  fwrite( data, 1, size, ff );
  fclose(ff);
  g++;
#endif
  
  ViewFrame f;

  if ( origSize == -1 )
    f.setSize( (unsigned)size );
  else
    f.setSize( (unsigned)origSize );

  // Dimensions
  //  if ( x > 0 && y > 0)
  f.setDimensions( x, y );
  
  f.finish();
  
  char * Output = f.getOutput();

  /* Bless the data, as we'll be aggregating multiple types of data */
  
  // Create a full message!

  Log::log( Logging::DEBUG, Output );
  
  if ( fullMessage == NULL ) {
    fullSize = size + strlen( Output );
    fullMessage = scinew char[ fullSize ];
  }

  if ( (unsigned)fullSize != size + strlen( Output ) ) {
    delete fullMessage;
    fullSize = size + strlen( Output );
    fullMessage = scinew char[ fullSize ];
  }

  // Zero out the space
  memset( fullMessage, 0, fullSize );
  
  //sprintf( fullMessage, "%s%s", Output, data );
  memcpy( fullMessage, Output, strlen( Output ) );
  memcpy( fullMessage + strlen( Output ), data, size );
  
  Log::log( Logging::DEBUG, fullMessage );

  //d::cerr << "Transmitting" << endl;
  // Now write it to the network.
  transmitData( fullMessage, fullSize );
}


void
Renderer::compressCallback( void * obj, MessageData *input ) {
  
  SemotusVisum::Message::Compression *compressMessage;
  bool okay = true;
  int type = 0;
  char buffer[1000];
  RenderGroup *rg = NULL;
  
  if ( obj ) {
    compressMessage =
      ( SemotusVisum::Message::Compression * )(input->message);
    if ( compressMessage ) {
      snprintf( buffer, 1000,
		"Client requesting compression change: %s",
		input->clientName );
      
      Log::log( Logging::DEBUG, buffer );
      
      // Get the render group for this client.
      rg = RenderGroup::getRenderGroup( input->clientName );

      if ( rg != NULL && compressMessage->isRequest() ) {
	// Get the compression method

	Log::log( Logging::DEBUG, "Got render group. Making a compressor.");
	// See if this compression method is allowed by the current renderer.
	// DO THIS HERE
	Compressor * cc = mkCompressor( compressMessage->getName(), type );
	if ( cc == NULL && type == CERROR ) {
	  Log::log( Logging::ERROR, "Error creating compressor!");
	  okay = false;
	}
	else 
	  // Set the compression. 
	  rg->setCompressor( cc );
	
      }
      else
	okay = false;
      
      // Reply
      SemotusVisum::Message::Compression *reply =
	scinew SemotusVisum::Message::Compression(false);
      
      reply->setOkay( okay, compressMessage->getName() );
      reply->finish();
      
      
      if ( okay )
	snprintf( buffer, 1000, "Switching compression for client %s to %s",
		  input->clientName, compressMessage->getName() );
      else 
	snprintf( buffer, 1000, "Bad compression: %s",
		  compressMessage->getName() );
      
      Log::log( Logging::MESSAGE, buffer );

      // Send reply to all clients in the group.
      if ( rg != NULL ) {
	list<char *>::const_iterator i;
	list<char *> clientList = rg->getClients();
	
	snprintf( buffer, 1000,
		  "We have %d clients in this render group.",
		  clientList.size() );
	Log::log( Logging::DEBUG, buffer );
	for ( i = clientList.begin(); i != clientList.end(); i++)
	  NetInterface::getInstance().sendPriorityDataToClient( *i, *reply );
      }
      else
	// Send the reply to the client.
	NetInterface::getInstance().
	  sendPriorityDataToClient(input->clientName,
				   *reply );
      Log::log( Logging::DEBUG, "Sent compression reply");
      std::cerr << "Sent compression reply" << endl;
    }
	
  }
}


}
}
