/*
 *
 * XMLDriver: Driver for XML reader and writer.
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: January 2001
 *
 */

#include <iostream>
#include <XML/XML.h>
#include <XML/XMLWriter.h>
#include <XML/XMLReader.h>


using namespace SemotusVisum;

int
main() {
  XMLWriter writer;
  String s;
  
  // Start a new document
  writer.newDocument();

  // Create Elements
  Attributes attributes; 


  writer.addElement("LoneElement", attributes, String(0));
  
  attributes.setAttribute("Attribute", "Value");
  writer.addElement("LevelTwo", attributes, String(0));

  attributes.clear();
  attributes.setAttribute("AnotherAttribute", "AnotherValue");
  writer.addElement("AlsoLevelTwo", attributes, "Who's your daddy?");
  
  writer.push();
  attributes.clear();
  writer.addElement("LevelThree", attributes, "I like level three!");
  
  writer.pop();
  writer.addElement("LastLevelTwo", attributes, "Level Two is best!");
  
  writer.push();
  writer.addElement("AnotherLevelThree",attributes,String(0));
  
  writer.push(); 
  writer.addElement("LevelFourAlready",attributes,String(0)); 
  
  writer.pop(); 
  writer.addElement("OkayLastLevelThree",attributes,String(0)); 

  string output = writer.writeOutputData();

  std::cout << output << endl;

  XMLByte * input = (XMLByte *)output.data();

  MemBufInputSource inputSource(input, strlen((char *)input), "foobar");
  XMLReader reader( &inputSource );
  reader.parseInputData();

  while (1) {

    s = reader.nextElement();
    
    if (s == 0) break;

    std::cout << s.transcode() << "\t";
    std::cout << reader.getText().transcode() << endl;

    attributes = reader.getAttributes();
    attributes.list();
  }

  std::cout << endl << "Done!" << endl;
}

//
// $Log$
// Revision 1.1  2003/07/22 20:59:18  simpson
// Adding CollabVis Client directory to Packages/CollabVis/Standalone
//
// Revision 1.1  2003/06/18 20:55:05  simpson
// Adding CollabVis files/dirs
//
// Revision 1.2  2001/05/29 03:43:12  luke
// Merged in changed to allow code to compile/run on IRIX. Note that we have a problem with network byte order in the networking code....
//
// Revision 1.1  2001/05/12 03:32:27  luke
// Moved driver to new location
//
// Revision 1.5  2001/04/04 21:35:33  luke
// Added XML initialization to reader and writer constructors
//
// Revision 1.4  2001/02/08 23:53:33  luke
// Added network stuff, incorporated SemotusVisum namespace
//
// Revision 1.3  2001/01/31 20:45:34  luke
// Changed Properties to Attributes to avoid name conflicts with client and server properties
//
// Revision 1.2  2001/01/29 18:48:47  luke
// Commented XML
//
