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

/****************************************
 *
 * This program takes a sci_filter.xml
 *   and parses it for include and 
 *   import xml tags, and then passes
 *   it to the xsl translator.
 *
 ******************************************/

#include "Generator.h"

int main(int argc, const char* argv[])
{

  // argv[1] = full path to Insight package
  // argv[2] = module Category
  // argv[3] = Module name
  // argv[4] = generator flag (optional)
  //           -cc  = generate cc files
  //           -gui = generate gui files
  //           -xml = generate xml files

  if((argc == 4) || (argc == 5)) {

    Generator* gen = new Generator();

    // determine which file we are generating
    FileFormat format = ALL;
    if(argc == 4) {
      // no flag specified so generate
      // all files
      format = ALL;
    }
    else if( (strcmp(argv[4],"-cc")==0) || (strcmp(argv[4],"-CC")==0)) {
      format = CC;
    }
    else if( (strcmp(argv[4],"-gui")==0) || (strcmp(argv[4],"-GUI")==0)) {
      format = GUI;
    }
    else if( (strcmp(argv[4],"-xml")==0) || (strcmp(argv[4],"-XML")==0)) {
      format = XML;
    }

    // set generators path to the insight package,
    // category, and module to know where to put generated file
    string path = argv[1];
    path += "/";
    gen->set_path_to_insight_package( path );
    gen->set_category(argv[2]);
    gen->set_module(argv[3]);

    // set the sci xml file
    string xml_file  = argv[1];
    xml_file += "/Dataflow/Modules/";
    xml_file += argv[2];
    xml_file += "/XML/sci_";
    xml_file += argv[3];
    xml_file += ".xml";
    gen->set_xml_file(xml_file);
    
    string xsl_file = argv[1];
    string output_file = argv[1];

    // set the appropriate XSL file
    if( format != ALL) {
      if( format == CC) {
	xsl_file += "/Core/CodeGenerator/XSL/SCIRun_generateCC.xsl";
      }
      else if( format == GUI ) {
	xsl_file += "/Core/CodeGenerator/XSL/SCIRun_generateTCL.xsl";
      }
      else if( format == XML ) {
	xsl_file += "/Core/CodeGenerator/XSL/SCIRun_generateXML.xsl";
      }
      gen->set_xsl_file(xsl_file);
      
      // set the appropriate output file
      if( format == CC) {
	output_file += "/Dataflow/Modules/";
	output_file += argv[2];
	output_file += "/";
	output_file += argv[3];
	output_file += ".cc";
      }
      else if( format == GUI ) {
	output_file += "/Dataflow/GUI/";
	output_file += argv[3];
	output_file += ".tcl";
	
      }
      else if( format == XML ) {
	output_file += "/Dataflow/XML/";
	output_file += argv[3];
	output_file += ".xml";
      }
      gen->set_output_file(output_file);
      
      
      if(!gen->generate(format)) {
	cerr << "**********************************************\n";
	cerr << "**     INSIGHT MODULE GENERATOR FAILED      **\n";
	cerr << "**  This could be due to errors in any of   **\n";
	cerr << "**  the 3 XML files.                        **\n";
	cerr << "**********************************************\n";
	return 0;
      }
    }
    else {
      // generate CC
      xsl_file = argv[1];
      xsl_file += "/Core/CodeGenerator/XSL/SCIRun_generateCC.xsl";
      gen->set_xsl_file(xsl_file);
      
      output_file = argv[1];
      output_file += "/Dataflow/Modules/";
      output_file += argv[2];
      output_file += "/";
      output_file += argv[3];
      output_file += ".cc";
      gen->set_output_file(output_file);

      if(!gen->generate(CC)) {
	cerr << "**********************************************\n";
	cerr << "**     INSIGHT MODULE GENERATOR FAILED      **\n";
	cerr << "**  This could be due to errors in any of   **\n";
	cerr << "**  the 3 XML files.                        **\n";
	cerr << "**********************************************\n";
	return 0;
      }     

      // generate GUI
      xsl_file = argv[1];
      xsl_file += "/Core/CodeGenerator/XSL/SCIRun_generateTCL.xsl";
      gen->set_xsl_file(xsl_file);

      output_file = argv[1];     
      output_file += "/Dataflow/GUI/";
      output_file += argv[3];
      output_file += ".tcl";
      gen->set_output_file(output_file);

      if(!gen->generate(GUI)) {
	cerr << "**********************************************\n";
	cerr << "**     INSIGHT MODULE GENERATOR FAILED      **\n";
	cerr << "**  This could be due to errors in any of   **\n";
	cerr << "**  the 3 XML files.                        **\n";
	cerr << "**********************************************\n";
	return 0;
      }

      // generate xml
      xsl_file = argv[1];
      xsl_file += "/Core/CodeGenerator/XSL/SCIRun_generateXML.xsl";
      gen->set_xsl_file(xsl_file);

      output_file = argv[1];     
      output_file += "/Dataflow/XML/";
      output_file += argv[3];
      output_file += ".xml";
      gen->set_output_file(output_file);

      if(!gen->generate(XML)) {
	cerr << "**********************************************\n";
	cerr << "**     INSIGHT MODULE GENERATOR FAILED      **\n";
	cerr << "**  This could be due to errors in any of   **\n";
	cerr << "**  the 3 XML files.                        **\n";
	cerr << "**********************************************\n";
	return 0;
      }
    }
    // check GUI/sub.mk

    // check Dataflow/CATEGORY/sub.mk
  }
  return 0;
}
