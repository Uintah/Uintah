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
 *  Port.h: Classes for module ports
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

#ifndef SCIRun_Dataflow_Network_Port_h
#define SCIRun_Dataflow_Network_Port_h

#include <string>
#include <vector>

namespace SCIRun {
class Connection;
class Module;


class Port {
public:
  Port(Module* module, const std::string& type_name,
       const std::string& port_name, const std::string& color_name);
  virtual ~Port();

  int num_unblocked_connections();
  int nconnections();
  Connection* connection(int);
  virtual void attach(Connection* conn);
  virtual void detach(Connection* conn);
  virtual void reset()=0;
  virtual void finish()=0;

  Module* get_module();
  int get_which_port();
  std::string get_typename();
  std::string get_colorname();
  std::string get_portname();

  enum ConnectionState {
    Connected,
    Disconnected
  };
protected:
  enum PortState {
    Off,
    Resetting,
    Finishing,
    On
  };
  Module* module;
  int which_port;
  PortState portstate;

  std::vector<Connection*> connections;

  void turn_on(PortState st=On);
  void turn_off();
  friend class Module;
  void set_which_port(int);
  virtual void update_light() = 0;
private:
  std::string type_name;
  std::string port_name;
  std::string color_name;

  Port(const Port&);
  Port& operator=(const Port&);
};


class IPort : public Port {
public:
  IPort(Module* module, const std::string& type_name,
	const std::string& port_name, const std::string& color_name);
  virtual ~IPort();
private:
  IPort(const IPort&);
  IPort& operator=(const IPort&);

  virtual void update_light();
};


class OPort : public Port {
public:
  OPort(Module* module, const std::string& type_name,
	const std::string& port_name, const std::string& color_name);
  virtual ~OPort();
  virtual bool have_data()=0;
  virtual void resend(Connection*)=0;
private:
  OPort(const OPort&);
  OPort& operator=(const OPort&);

  virtual void update_light();
};

}

#endif
