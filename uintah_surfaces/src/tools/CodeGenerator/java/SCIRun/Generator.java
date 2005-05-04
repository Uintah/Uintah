/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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
