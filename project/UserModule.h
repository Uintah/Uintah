
/*
 *  UserModule.h: Base class for defined modules
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_UserModule_h
#define SCI_project_UserModule_h 1

#include <Module.h>
#include <Multitask/Task.h>
class MUI_window;
class MUI_widget;
class MUI_onoff_switch;
class InData;
class OutData;

class UserModule : public Module, public Task {
    MUI_window* window;

    virtual void activate();
    // Implementation of members from Task
    virtual int body(int);
public:
    UserModule(const clString& name);
    UserModule(const UserModule&, int deep);
    virtual ~UserModule();

    // Port manipulations
    void add_iport(InData*, const clString&, int);
    void add_oport(OutData*, const clString&, int);
    void remove_iport(int);
    void remove_oport(int);
    void rename_iport(int, const clString&);
    void rename_oport(int, const clString&);

    // Execute Condition
    enum CommonEC {
	NewDataOnAllConnectedPorts,
	Always,
	OnOffSwitch,
    };
    CommonEC ec;
    MUI_onoff_switch* sw;
    int swval;
    void execute_condition(CommonEC);
    void execute_condition(MUI_onoff_switch*, int value);

    // User Interface Manipulations
    void remove_ui(MUI_widget*);
    void add_ui(MUI_widget*);
    void reconfigure_ui();

    // Misc stuff for module writers
    void error(const clString&);

    // Callbacks...
    virtual void connection(Module::ConnectionMode, int, int);
};

#endif /* SCI_project_UserModule_h */

