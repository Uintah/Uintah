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



#ifndef BRIDGE_CONNECTION_H
#define BRIDGE_CONNECTION_H

#include <CCA/Components/Builder/Connection.h>

using namespace SCIRun;

class BridgeConnection : public Connection 
{
public:
  BridgeConnection(Module* m1, const std::string& s1, Module* m2, const std::string& s2, const sci::cca::ConnectionID::pointer &connID, QCanvasView *cv)
  : Connection(m1,s1,m2,s2,connID,cv) { }
  using Connection::resetPoints;
  using Connection::isConnectedTo;
  using Connection::getConnectionID;
  using Connection::highlight;
  using Connection::setDefault;
  using Connection::getUsesModule;
  using Connection::getProvidesModule;
  using Connection::getUsesPortName;
  using Connection::getProvidesPortName;
  virtual std::string getConnectionType();

protected:
  void drawShape(QPainter& );
};

#endif



