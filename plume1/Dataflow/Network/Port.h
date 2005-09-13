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

  int nconnections();
  Connection* connection(int);
  virtual void attach(Connection* conn);
  virtual void detach(Connection* conn, bool blocked);
  virtual void reset()=0;
  virtual void finish()=0;
  virtual void synchronize();

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
protected:
  static void issue_no_port_caching_warning();
};

}

#endif
