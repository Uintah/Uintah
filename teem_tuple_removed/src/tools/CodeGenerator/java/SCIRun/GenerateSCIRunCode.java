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
