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
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_Port_h
#define SCI_project_Port_h 1

#include <Dataflow/share/share.h>
#include <Dataflow/Comm/MessageBase.h>
#include <vector>
#include <string>

using std::vector;
using std::string;

namespace SCIRun {


class Connection;
class Module;

class PSECORESHARE Port {
    string type_name;
    string portname;
    string colorname;
    int protocols;
    int u_proto;
protected:
    Module* module;
    int which_port;
    vector<Connection*> connections;
    int xlight, ylight;
    enum PortState {
	Off,
	Resetting,
	Finishing,
	On
    };
    PortState portstate;
public:
    Port(Module*, const string&, const string&,
	 const string&, int protocols);
    virtual ~Port();
    void set_port(int which_port);
    int using_protocol();
    int nconnections();
    Connection* connection(int);
    Module* get_module();
    int get_which_port();
    void set_which_port(int);
    virtual void attach(Connection*);
    virtual void detach(Connection*);
    virtual void reset()=0;
    virtual void finish()=0;
    string get_typename();
    string get_portname();
    string get_colorname();
    void set_portname(string&);
};

class PSECORESHARE IPort : public Port {
    void update_light();
protected:
    IPort(Module*, const string&, const string&,
	  const string&, int protocols);
    void turn_on(PortState st=On);
    void turn_off();
public:
    virtual ~IPort();
};

class PSECORESHARE OPort : public Port {    
    void update_light();
protected:
    OPort(Module*, const string&, const string&,
	  const string&, int protocols);
    void turn_on(PortState st=On);
    void turn_off();
public:
    virtual ~OPort();
    virtual int have_data()=0;
    virtual void resend(Connection*)=0;
};

} // End namespace SCIRun


#endif /* SCI_project_Port_h */
