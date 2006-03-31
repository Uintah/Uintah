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



#ifndef ComponentSkeletonWriter_h
#define ComponentSkeletonWriter_h

#include <vector>
#include <string>
#include<fstream>

namespace GUIBuilder {

class PortDescriptor {
public:
  PortDescriptor(const std::string& name, const std::string& type) : name(name), type(type) {}
  const std::string& GetName() const { return name; }
  const std::string& GetType() const { return type; }

private:
  std::string name;
  std::string type;
};

class ComponentSkeletonWriter {
public:
  ComponentSkeletonWriter(const std::string &cname, const std::vector<PortDescriptor*> pp,const std::vector<PortDescriptor*> up);

  void ComponentClassDefinitionCode();
  void ComponentSourceFileCode();

  void PortClassDefinitionCode();
  void GenerateCode();

private:
  const std::string sp;
  std::string compName;
  std::vector<PortDescriptor*> providesPortsList;
  std::vector<PortDescriptor*> usesPortsList;
  std::ofstream componentSourceFile;
  std::ofstream componentHeaderFile;
};

}

#endif
