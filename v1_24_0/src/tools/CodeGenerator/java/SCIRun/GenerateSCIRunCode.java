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
  GenerateSCIRunCode 
  
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

// Imported java classes
import java.io.FileOutputStream;
import java.io.FileNotFoundException;
import java.io.IOException;

//import Generator;

public class GenerateSCIRunCode
{
  /////////////////////////////////////////////////////////////////////////////////
  // main
  //   args[0] full path to package
  //   args[1] XML File
  //   args[2] XSL File
  //   args[3] Output File
  /////////////////////////////////////////////////////////////////////////////////
  public static void main(String[] args)
    throws TransformerException, TransformerConfigurationException,
           FileNotFoundException, IOException
  {
    if(args.length == 4) {
      
      Generator gen = new Generator();
      
      // set generators XML file, XSL file, and Ouput file
      full_path_to_package = args[0] + "/";
      xml_file = args[1];
      xsl_file = args[2];
      output_file = args[3];

      gen.set_full_path_to_package( full_path_to_package );
      gen.set_xml_file( xml_file );
      gen.set_xsl_file( xsl_file );
      gen.set_output_file( output_file );
      
      if(!gen.generate()) {
	System.out.println("************************************");
	System.out.println("*     MODULE GENERATOR FAILED      *");
	System.out.println("* [Error] means XML file does not  *");
	System.out.println("*         conform to DTD           *");
	System.out.println("* [Fatal Error] means XML file is  *");
	System.out.println("*        well formated             *");
	System.out.println("************************************");
      }
      
    }
    else {
      System.out.println("Error - Incorrect number of arguments");
      System.out.println("\tFull Path to Package");
      System.out.println("\tXML File");
      System.out.println("\tXSL File");
      System.out.println("\tOutput File");
    }
  }
  
  
  public static String xsl_file = "";
  public static String xml_file = "";
  public static String output_file = "";
  public static String full_path_to_package = "";

}
