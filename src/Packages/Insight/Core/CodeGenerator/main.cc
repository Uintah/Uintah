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

  if(argc == 4) {

    Generator* gen = new Generator();
    string path = argv[1];
    path += "/";
    gen->set_path_to_insight_package( path );
    string cc_xsl_file = argv[1];
    cc_xsl_file += "/Core/CodeGenerator/XSL/SCIRun_generateCC.xsl";
    gen->set_cc_xsl_file(cc_xsl_file);
    string gui_xsl_file = argv[1];
    gui_xsl_file += "/Core/CodeGenerator/XSL/SCIRun_generateTCL.xsl";
    gen->set_gui_xsl_file(gui_xsl_file);
    string xml_xsl_file = argv[1];
    xml_xsl_file += "/Core/CodeGenerator/XSL/SCIRun_generateXML.xsl";
    gen->set_xml_xsl_file(xml_xsl_file);

    string xml_file  = argv[1];
    xml_file += "/Dataflow/Modules/";
    xml_file += argv[2];
    xml_file += "/XML/sci_";
    xml_file += argv[3];
    xml_file += ".xml";
    gen->set_xml_file(xml_file);

    string cc_out = argv[1];
    cc_out += "/Dataflow/Modules/";
    cc_out += argv[2];
    cc_out += "/";
    cc_out += argv[3];
    cc_out += ".cc";
    gen->set_cc_out(cc_out);

    string gui_out = argv[1];
    gui_out += "/Dataflow/GUI/";
    gui_out += argv[3];
    gui_out += ".tcl";
    gen->set_gui_out(gui_out);

    string xml_out = argv[1];
    xml_out += "/Dataflow/XML/";
    xml_out += argv[3];
    xml_out += ".xml";
    gen->set_xml_out(xml_out);

    if(!gen->generate())
      cerr << "GENERATOR FAILED!\n";

    // check GUI/sub.mk

    // check Dataflow/CATEGORY/sub.mk
  }
  return 0;
}
