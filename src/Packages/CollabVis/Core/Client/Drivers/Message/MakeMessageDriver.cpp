/*
 *
 * MakeMessageDriver: Tests MakeMessage ( return a message from XML )
 *                    functionality.
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: January 2001
 *
 */

#include <Message/MakeMessage.h>

using namespace SemotusVisum;

char * garbage = "<?xml version=\"1.0\" encoding=\"UTF-8\"?><barf/>";

char * test1 = "<?xml version='1.0' encoding='ISO-8859-1' ?><clientList><client name = \"foo\" address = \"127.0.0.1\"/><client name = \"bar\" group = \"GreatRenderer1\" address = \"156.56.453\"/><client name = \"baz\" group = \"YoMamaRenderer\" address = \"120.123.0.4\"/></clientList>";
//char * test2 = "<?xml version=\"1.0\" encoding=\"UTF-8\"?><mouseMove x=\"1\" y=\"213\" button=\"r\"></mouseMove>";
char * test3 = "<?xml version=\"1.0\" encoding=\"UTF-8\"?><GetZTex><eyePoint x=\"1.0\" y=\"2.3\" z=\"-12\"></eyePoint><lookAtPoint x=\"1.0\" y=\"2.3\" z=\"-123.3\"></lookAtPoint><upVector x=\"0\" y=\"1\" z=\"0\"></upVector></GetZTex>";
char * test4 = "<?xml version='1.0' encoding='ISO-8859-1' ?><setViewingMethod><method method = \"ZTex\">okay</method></setViewingMethod>";
char * test5 = "<?xml version='1.0' encoding='ISO-8859-1' ?><handshake><imageFormats><format>bar</format><format>BYTE_GRAY</format></imageFormats><viewingMethods><method name = \"Image Streaming\" version = \"Revision: 1.15\"/></viewingMethods><compressionFormats><format>JPEG</format></compressionFormats></handshake>";
char * test51 = "<?xml version='1.0' encoding='ISO-8859-1' ?><handshake><imageFormats><format>bar</format><format>BYTE_GRAY</format></imageFormats><viewingMethods><method name = \"Image Streaming\" version = \"Revision: 1.15\"/></viewingMethods><compressionFormat/></handshake>";
char * test6 = "<?xml version=\"1.0\" encoding=\"UTF-8\"?><multicast>Disconnect</multicast>";
//char * test7 = "<?xml version='1.0' encoding='ISO-8859-1' ?><goodbye/>";
char * test8 = "<?xml version='1.0' encoding='ISO-8859-1' ?><compression compression=\"UberCompressor\">okay</compression>";
char * test9 = "<?xml version=\"1.0\" encoding=\"UTF-8\"?><chat>Who's your daddy?</chat>";
char * test10 = "<?xml version=\"1.0\" encoding=\"UTF-8\"?><collaborate><pointer erase=\"false\" id=\"1\" width=\"1\" theta=\"0.5\" x=\"1\" y=\"2\" z=\"3\" red = \"255\" blue = \"3\" green = \"8\"/><pointer width=\"1\" erase=\"false\" id=\"2\" theta=\"2.0\" x=\"3\" y=\"2\" z=\"1\" red = \"255\" blue = \"3\" green = \"8\" /><text erase=\"false\" id=\"3\" x=\"0\" y=\"0\" red = \"255\" blue = \"3\" green = \"8\" size=\"1\" >Who's your daddy?</text><drawing width=\"1\" erase=\"false\" id=\"4\" red = \"255\" blue = \"3\" green = \"8\"><segment x=\"2\" y=\"4\" z=\"6\"/><segment x=\"2\" y=\"4\" z=\"8\"/></drawing></collaborate>";
char * test11 = "<?xml version=\"1.0\" encoding=\"UTF-8\"?><getXDisplay response=\"okay\"></getXDisplay>";
char * test12 ="<?xml version=\"1.0\" encoding=\"UTF-8\"?><GroupViewer><group name=\"foo\" viewer=\"bar\" /></GroupViewer>";

int
main() {

  message_t type;
  
  // Test no input.
  if ( MakeMessage::makeMessage( NULL, type ) ||
       type != ERROR_M )
    std::cerr << "Oops! No input gives wrong result!" << endl;
  
  // Test garbage input.
  if ( MakeMessage::makeMessage( garbage, type ) ||
       type != UNKNOWN_M )
    std::cerr << "Oops! Bad input gives wrong result!" << endl;
  
  // Test several different types of messages.
  if ( !MakeMessage::makeMessage( test1, type ) ||
       type != GET_CLIENT_LIST )
    std::cerr << "Oops! GetClientList failed." << endl;

  //  if ( !MakeMessage::makeMessage( test2, type ) ||
  //       type != MOUSE_MOVE )
  //    std::cerr << "Oops! MouseMove failed." << endl;

  if ( !MakeMessage::makeMessage( test3, type ) ||
       type != GET_Z_TEX )
    std::cerr << "Oops! GetZTex failed." << endl;

  if ( !MakeMessage::makeMessage( test4, type ) ||
       type != SET_VIEWING_METHOD )
    std::cerr << "Oops! SetViewingMethod failed." << endl;
  
  if ( !MakeMessage::makeMessage( test5, type ) ||
       type != HANDSHAKE )
    std::cerr << "Oops! Handshake failed." << endl;

  if ( !MakeMessage::makeMessage( test6, type ) ||
       type != MULTICAST )
    std::cerr << "Oops! Multicast failed." << endl;
  //if ( !MakeMessage::makeMessage( test7, type ) ||
  //     type != GOODBYE )
  //  std::cerr << "Oops! Goodbye failed." << endl;
  if ( !MakeMessage::makeMessage( test8, type ) ||
       type != COMPRESSION )
    std::cerr << "Oops! Compression failed." << endl;
  if ( !MakeMessage::makeMessage( test9, type ) ||
       type != CHAT )
    std::cerr << "Oops! Chat failed." << endl;
  if ( !MakeMessage::makeMessage( test10, type ) ||
       type != COLLABORATE )
    std::cerr << "Oops! Collaborate failed." << endl;
  if ( !MakeMessage::makeMessage( test11, type ) ||
       type != XDISPLAY )
    std::cerr << "Oops! X Display failed." << endl;
  if ( !MakeMessage::makeMessage( test12, type ) ||
       type != GROUP_VIEWER )
    std::cerr << "Oops! Group Viewer failed." << endl;
  
  std::cerr << "Tests done. If you saw no failure messages, all is well."
	    << endl;
}
  
