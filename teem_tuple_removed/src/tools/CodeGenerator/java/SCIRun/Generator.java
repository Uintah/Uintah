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
  Generator
  
  Written by: 
    Darby J Van Uitert
    August 2003
*/

package SCIRun;

// Imported TraX classes
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.Transformer;
import javax.xml.transform.stream.StreamSource;
import javax.xml.transform.stream.StreamResult;
import javax.xml.transform.TransformerException;
import javax.xml.transform.TransformerConfigurationException;
import javax.xml.transform.dom.DOMSource;


// Imported java classes
import java.io.FileOutputStream;
import java.io.FileNotFoundException;
import java.io.IOException;

import org.w3c.dom.Node;


//import Parser;

public class Generator
{
  public Generator()
  {
    
  }
  
  public void set_xml_file(String file) 
  {
    xml_file = file;
  }

  public void set_xsl_file(String file)
  {
    xsl_file = file;
  }

  public void set_output_file(String out) 
  {
    output_file = out;
  }

  public void set_full_path_to_package(String path) 
  {
    full_path_to_package = path;
    parser.set_full_path_to_package( path );
  }

  /////////////////////////////////////////////////////////////////////////////////
  // generate
  /////////////////////////////////////////////////////////////////////////////////
  public static boolean generate() 
    throws TransformerException, TransformerConfigurationException,
            FileNotFoundException, IOException
  {
    // parse file for include and import tags
    Node node = parser.read_input_file( xml_file );

    if( node == null ) {
      System.out.println("Node is null after initial call to read_input_file");
      return false;
    }

    if( parser.has_errors ) {
      System.out.println("Parse had errors");
      return false;
    }


    // Use the static TransformerFactory.newInstance() method to instantiate 
    // a TransformerFactory. The javax.xml.transform.TransformerFactory 
    // system property setting determines the actual class to instantiate --
    // org.apache.xalan.transformer.TransformerImpl.
    TransformerFactory tFactory = TransformerFactory.newInstance();

	
    // Use the TransformerFactory to instantiate a Transformer that will work with  
    // the stylesheet you specify. This method call also processes the stylesheet
    // into a compiled Templates object.
    Transformer transformer = tFactory.newTransformer(new StreamSource( xsl_file ));


    // Use the Transformer to apply the associated Templates object to an XML document
    // (foo.xml) and write the output to a file (foo.out).

    //transformer.transform(new StreamSource( xml_file ), new StreamResult(new FileOutputStream( output_file )));
    DOMSource domSource = new DOMSource(node.getOwnerDocument());
    transformer.transform(domSource, new StreamResult(new FileOutputStream( output_file )));


    return true;
  }

  public static String xml_file = "";
  public static String xsl_file = "";
  public static String output_file = "";
  public static String full_path_to_package = "";

  public static Parser parser = new Parser();
  

}
