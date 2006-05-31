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
 *  World.h
 *
 *  Written by:
 *   Keming Zhang 
 *   Department of Computer Science
 *   University of Utah
 *   May 2003
 *
 */

#ifndef SCIRun_CCA_Components_World_h
#define SCIRun_CCA_Components_World_h

#include <Core/CCA/spec/cca_sidl.h>

namespace SCIRun {

class World;

class WUIPort : public sci::cca::ports::UIPort {
public:
    int ui();
    void setParent(World *com) { this->com = com; }
    World *com;
};

class StringPort : public sci::cca::ports::StringPort {
public:
    std::string getString();
    void setParent(World *com) { this->com = com; }
    World *com;
};


class ComponentIcon : public virtual sci::cca::ports::ComponentIcon {
public:
  virtual ~ComponentIcon() {}

  virtual std::string getDisplayName();
  virtual std::string getDescription();
  virtual std::string getIconShape();
  virtual int getProgressBar();
};


class World : public sci::cca::Component {
public:
    World();
    virtual ~World();
    void setServices(const sci::cca::Services::pointer& svc);
    std::string text;

private:
    World(const World&);
    World& operator=(const World&);
    void setCommunicator(int comm) {
        //MPI_COMM_COM=MPI_COMM_WORLD; //*(MPI_Comm*)(comm);
    }
    sci::cca::Services::pointer services;
};


} //namepace SCIRun


#endif
