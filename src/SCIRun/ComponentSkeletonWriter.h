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
 * ComponentSkeletonWriter.h
 *
 * Written by:
 *  <author>
 *  Scientific Computing and Imaging Institute
 *  University of Utah
 *  <date>
 *
 */



#ifndef SCIRun_ComponentSkeletonWriter_h
#define SCIRun_ComponentSkeletonWriter_h

#include <vector>
#include <string>
#include <fstream>

namespace GUIBuilder {

class PortDescriptor {
public:
  PortDescriptor(const std::string& cName, const std::string& type, const std::string& uName)
    : cName(cName), type(type) , uName(uName) {}
  const std::string& GetClassName() const { return cName; }
  const std::string& GetType() const { return type; }
  const std::string& GetUniqueName() const { return uName; }

private:
  std::string cName; // name of the concrete port class to be provided or used by the component
                     // (inherits from SIDL port type)
  std::string type; // SIDL port type
  std::string uName; // port name (must be unique in component class)
};

class ComponentSkeletonWriter {
public:
  /** throws internalException if directory does not exist */
  ComponentSkeletonWriter(const std::string &cname,
                          const std::string &dir,
                          const std::vector<PortDescriptor*> pp,
                          const std::vector<PortDescriptor*> up,
                          const bool& iws);
  ComponentSkeletonWriter();
  /** file basenames only */
  void generate(std::string headerFile, std::string sourceFile, std::string makeFile);
  void GenerateWithSidl(std::string headerFilename, std::string sourceFilename, std::string makeFilename, std::string sidlFilename);
  // frequently used string tokens
  const static std::string SP;
  const static std::string QT;
  const static std::string DIR_SEP;
  const static std::string OPEN_C_COMMENT;
  const static std::string CLOSE_C_COMMENT;
  const static std::string UNIX_SHELL_COMMENT;
  const static std::string NEWLINE;

  const static std::string DEFAULT_NAMESPACE;
  const static std::string DEFAULT_SIDL_NAMESPACE;
  const static std::string DEFAULT_PORT_NAMESPACE;
  const static std::string DEFAULT_SIDL_PORT_NAMESPACE;

  std::string getTempDirName();
  std::string getCompDirName();
private:
  void ComponentClassDefinitionCode(std::ofstream& fileStream);
  void ComponentSourceFileCode(std::ofstream& fileStream);
  void ComponentMakefileCode(std::ofstream& fileStream);

  void writeLicense(std::ofstream& fileStream);
  void writeMakefileLicense(std::ofstream& fileStream);
  

  // generate header file
  void writeHeaderInit(std::ofstream& fileStream);
  void writeComponentDefinitionCode(std::ofstream& fileStream);
  void writePortClassDefinitionCode(std::ofstream& fileStream);

  // generate implementation
  void writeLibraryHandle(std::ofstream& fileStream);
  void writeSourceInit(std::ofstream& fileStream);
  void writeSourceClassImpl(std::ofstream& fileStream);
  void writeSourceFileHeaderCode(std::ofstream& fileStream);
  void writeConstructorandDestructorCode(std::ofstream& fileStream);
  void writeSetServicesCode(std::ofstream& fileStream);
  void writeGoAndUIFunctionsCode(std::ofstream& fileStream);

  void writeSidlFile(std::ofstream& fileStream);
  // more frequently used string tokens
  const std::string SERVICES_POINTER;
  const std::string TYPEMAP_POINTER;

  // Component name
  std::string compName;
  std::string directory;

  // List of Ports added
  std::vector<PortDescriptor*> providesPortsList;
  std::vector<PortDescriptor*> usesPortsList;

  // File Handles
  std::ofstream componentSourceFile;
  std::ofstream componentHeaderFile;
  std::ofstream componentMakefile;
  
  bool isWithSidl;
};

}

#endif
