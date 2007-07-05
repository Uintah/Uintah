#include <Message/MessageBase.h>
#include <Message/MakeMessage.h>
#include <XML/XML.h>
#include <XML/XMLReader.h>

namespace SemotusVisum {

MessageBase::MessageBase() : output( "" ), finished( false ) {
    XMLI::initialize();
}
MessageBase::~MessageBase() {
}

MessageBase *
MakeMessage::makeMessage( void * data, message_t &type ) {

  if ( !data ) {
    type = ERROR_M;
    Log::log( ERROR, "No data passed to makeMessage()" );
    return NULL;
  }

  /* Parse the XML */
  char * input = ::strdup( (char *)data );
  //  cerr << "Did strdup" << endl;
  XMLI::initialize();
  
  XMLReader reader( input );
  reader.parseInputData();
  	    
  string tag = XMLI::getChar( reader.nextElement() );
  //cerr << "Parsed XML" << endl;
  //delete input;
  
  if ( tag.empty() ) {
    Log::log( ERROR, "MakeMessage(): No tag from XML!" );
    Log::log( DEBUG, input );
    type = ERROR_M;
    return NULL;
  }
  
  /* Switch among the different types of messages */
  if ( !strcasecmp( tag, "clientList" ) ) {
    type = GET_CLIENT_LIST;
    return GetClientList::mkGetClientList( data );
  }
  else if ( !strcasecmp( tag, "GetZTex" ) ) {
    type = GET_Z_TEX;
    return GetZTex::mkGetZTex( data );
  }
  else if ( !strcasecmp( tag, "Handshake" ) ) {
    type = HANDSHAKE;
    return Handshake::mkHandshake( data );
  }
  else if ( !strcasecmp( tag, "MouseMove" ) ) {
    type = MOUSE_MOVE;
    return NULL; // Outbound only!
  }
  else if ( !strcasecmp( tag, "SetViewingMethod" ) ) {
    type = SET_VIEWING_METHOD;
    return SetViewingMethod::mkSetViewingMethod( data );
  }
  else if ( !strcasecmp( tag, "Multicast" ) ) {
    type = MULTICAST;
    return Multicast::mkMulticast( data );
  }
  else if ( !strcasecmp( tag, "Goodbye" ) ) {
    type = GOODBYE;
    return NULL; // Outbound only!
  }
  else if ( !strcasecmp( tag, "Compression" ) ) {
    type = COMPRESSION;
    return Compression::mkCompression( data );
  }
  else if ( !strcasecmp( tag, "Transfer" ) || !strcasecmp( tag, "transfer" )) { // FIXME, shouldn't include case fot 'transfer'
    type = TRANSFER;
    return Transfer::mkTransfer( data );
  }
  else if ( !strcasecmp( tag, "Chat" ) ) {
    type = CHAT;
    return Chat::mkChat( data );
  }
  else if ( !strcasecmp( tag, "Collaborate" ) ) {
    type = COLLABORATE;
    return Collaborate::mkCollaborate( data );
  }
  else if ( !strcasecmp( tag, "getXDisplay" ) ||
	    !strcasecmp( tag, "moduleSetup" ) ) {
    type = XDISPLAY;
    return XDisplay::mkXDisplay( data );
  }
  else if ( !strcasecmp( tag, "GroupViewer" ) ) {
    type = GROUP_VIEWER;
    return GroupViewer::mkGroupViewer( data );
  }
  else if ( !strcasecmp( tag, "ViewFrame" ) ) {
    type = VIEW_FRAME;
    return ViewFrame::mkViewFrame( data );
  }
  else {
    Log::log( ERROR, "Unknown tag " + tag + " in makeMessage()" );
    type = UNKNOWN_M;
    return NULL;
  }
}

}
