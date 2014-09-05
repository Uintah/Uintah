/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


/*
  Parser
  
  Written by: 
    Darby J Van Uitert
    August 2003
*/

package SCIRun;

import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.w3c.dom.NamedNodeMap;
import org.w3c.dom.Text;
import org.apache.xerces.parsers.*;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;


// Imported java classes
import java.io.FileOutputStream;
import java.io.FileNotFoundException;
import java.io.IOException;

public class Parser
{
  public Parser()
  {
    
  }

  public static Node read_input_file(String filename)
  {
    Node result = null;
    
    DOMParser parser = new DOMParser();

    try {
      parser.setFeature("http://xml.org/sax/features/validation", true);
      
      parser.parse( filename );
      
      Document doc = parser.getDocument();

      if( doc == null ) {
	System.out.println("Parse failed");
	return null;
      }

      result = doc.getDocumentElement();
      
      resolve_includes( result );
      resolve_imports( result );
      //traverse( doc );
    }
    catch (IOException e) {
      System.out.println( filename + " does not exist");
      System.out.println( e.getMessage() );
    }
    catch (Exception e) {
      System.out.println( filename + " is not valid."); 
      System.out.println( e.getMessage() );
    }
    
    return result;
  }

   //  Traverse DOM Tree.  Print out Element Names
  public static void traverse (Node node) {
    int type = node.getNodeType();
    if (type == Node.ELEMENT_NODE) {
      NamedNodeMap attributes = node.getAttributes();
      if(attributes.getLength() == 0) {
	System.out.print ("<" +node.getNodeName() + ">");
      }
      else {
	System.out.print("<" +node.getNodeName() + " ");
	for(int i=0 ; i<attributes.getLength(); i++) {
	  Node attribute = attributes.item(i);
	  System.out.print(attribute.getNodeName() + "=\"" + attribute.getNodeValue() + "\" ");
	}
	System.out.print(">");
      }
    }

    if (type == Node.TEXT_NODE)
      System.out.print(" " + node.getNodeValue() + "\n");


    NodeList children = node.getChildNodes();
    if (children != null) {
      for (int i=0; i< children.getLength(); i++) 
	traverse (children.item(i));  
    }
  }

  public static void resolve_includes(Node root)
  {
    
    NodeList children = root.getChildNodes();

    for(int i=0; i<children.getLength(); i++) {
      Node child = children.item(i);
      if(child.getNodeType() == Node.ELEMENT_NODE) {
	String name = child.getNodeName();
	if(name.equals("include")) {
	  
	  NamedNodeMap attr = child.getAttributes();
	  int num_attr = attr.getLength();
	  
	  for(int j=0; j<num_attr; j++) {

	    String attrName = attr.item(j).getNodeName();
	    String attrValue = attr.item(j).getNodeValue();
	    
	    if(attrName.equals("href")) {

	      String file = full_path_to_package + attrValue;
	      
	      Node include = read_input_file( file );
	      Node to_insert = child.getOwnerDocument().importNode(include, true);
	      
	      root.appendChild( to_insert );

	      Text leader = child.getOwnerDocument().createTextNode("\n");

	      root.appendChild( leader );
	      root.removeChild( child );
	    }
	    resolve_includes( child );
	  }
	  child = child.getNextSibling();
	}
      }
    }
  }

  public static void resolve_imports(Node root)
  {
    NodeList children = root.getChildNodes();
    
    for(int i=0; i<children.getLength(); i++) {
      Node child = children.item(i);
      if(child.getNodeType() == Node.ELEMENT_NODE) {
	String name = child.getNodeName();
	if(name.equals("import")) {
	  NamedNodeMap attr = child.getAttributes();
	  int num_attr =  attr.getLength();
	  
	  for(int j=0; j<num_attr; j++) {
	    String attrName = attr.item(i).getNodeName();
	    String attrValue = attr.item(i).getNodeValue();
	    
	    if(name.equals("name")) {
	      Node include = read_input_file( attrValue );
	      Node to_insert = child.getOwnerDocument().importNode(include, true);
	      root.appendChild( to_insert );
	      Text leader = child.getOwnerDocument().createTextNode("\n");
	      root.appendChild( leader );
	      root.removeChild( child );
	    }
	    resolve_imports( child );
	  }
	  child = child.getNextSibling();
	}
      }
    }
  }

  
  public void set_full_path_to_package(String path)
  {
    full_path_to_package = path;
  }

  public static boolean has_errors;
  public static String full_path_to_package;

}
  
