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




#ifndef BRIDGE_CONNECTION_H
#define BRIDGE_CONNECTION_H

#include <CCA/Components/Builder/Connection.h>

using namespace SCIRun;

class BridgeConnection : public Connection 
{
public:
  BridgeConnection(PortIcon* p1, PortIcon* p2, const sci::cca::ConnectionID::pointer &connID, QCanvasView *cv)
  : Connection(p1, p2, connID, cv) { }
  using Connection::resetPoints;
  using Connection::isConnectedTo;
  using Connection::getConnectionID;
  using Connection::highlight;
  using Connection::setDefault;
  using Connection::usesPort;
  using Connection::providesPort;
  virtual std::string getConnectionType();

protected:
  void drawShape(QPainter& );
};

#endif



