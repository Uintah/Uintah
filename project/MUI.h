
/*
 *  MUI.h: Module User Interface classes
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_MUI_h
#define SCI_project_MUI_h 1

#include <Classlib/Array1.h>
#include <Classlib/String.h>
#include <MessageBase.h>

class clString;
class CallbackData;
class DrawingAreaC;
class EncapsulatorC;
class FileSelectionBoxC;
class MUI_widget;
class MUI_window_private;
class Point;
class ScaleC;
class UserModule;
class XQColor;

class MUI_window {
    MUI_window_private* priv;
    UserModule* module;
    Array1<MUI_widget*> widgets;
    int activated;
    int popup_on_activate;
public:
    MUI_window(UserModule*);
    ~MUI_window();
    void activate();
    void attach(MUI_widget*);
    void detach(MUI_widget*);
    void reconfigure();

    void popup();
    void popdown();

    UserModule* get_module();
};

class MUI_widget {
protected:
    clString name;
    MUI_window* window;
    void* cbdata;

    void dispatch(const clString&, clString*, int);
    void dispatch(double, double*, int);
    void dispatch(int, int*, int);
public:
    enum DispatchPolicy {
	Immediate,
	NotExecuting,
    };
    MUI_widget(const clString& name, void* cbdata,
	       DispatchPolicy);
    virtual ~MUI_widget();
    virtual void attach(MUI_window*, EncapsulatorC*)=0;
    enum Orientation {
	Horizontal,
	Vertical,
    };
    void set_title(const clString&);
    void set_dimensions(int width, int height);
private:
    DispatchPolicy dispatch_policy;
};

class MUI_slider_real : public MUI_widget {
    ScaleC* scale;
    double* data;
    int dispatch_drag;
    double base;
    void drag_callback(CallbackData*, void*);
    void value_callback(CallbackData*, void*);
public:
    enum Style {
	Slider,
	Dial,
	ThumbWheel,
	Guage,
    };
    MUI_slider_real(const clString& name, double* data,
		    DispatchPolicy, int dispatch_drag,
		    Style=Slider, Orientation=Horizontal,
		    void* cbdata=0);
    virtual ~MUI_slider_real();
    virtual void attach(MUI_window*, EncapsulatorC*);

    void set_minmax(double, double, int);
    void set_value(double);
    void set_orientation(MUI_widget::Orientation);
    void set_style(Style);
    enum Event {
	Drag, Value,
    };
};

class MUI_point : public MUI_widget {
    Point* data;
    int dispatch_drag;
public:
    MUI_point(const clString& name, Point* data,
	      DispatchPolicy, int dispatch_drag,
	      void* cbdata=0);
    virtual ~MUI_point();
    virtual void attach(MUI_window*, EncapsulatorC*);
};

class MUI_onoff_switch : public MUI_widget {
    int* data;
    DrawingAreaC* sw;
    int anim;
    XQColor* bgcolor;
    XQColor* top_shadow;
    XQColor* bot_shadow;
    XQColor* inset_color;
    XQColor* text_color;
    int fh;
    int width, height;
    int descent;
    void* vgc;
    void event_callback(CallbackData*, void*);
    void expose_callback(CallbackData*, void*);
public:
    MUI_onoff_switch(const clString& name, int* data,
		     DispatchPolicy,
		     void* cbdata=0);
    ~MUI_onoff_switch();
    virtual void attach(MUI_window*, EncapsulatorC*);
    enum Event {
	Value,
    };
};

class MUI_file_selection : public MUI_widget {
    FileSelectionBoxC* sel;
    clString* filename;

    void ok_callback(CallbackData*, void*);
    void cancel_callback(CallbackData*, void*);
public:
    MUI_file_selection(const clString& name, clString* filename,
		       DispatchPolicy,
		       void* cbdata=0);
    ~MUI_file_selection();
    virtual void attach(MUI_window*, EncapsulatorC*);
};

class MUI_Module_Message : public MessageBase {
    clString newstr;
    clString* str;
    double newddata;
    double* ddata;
    int newidata;
    int* idata;
    enum Type {
	DoubleData,
	StringData,
	IntData,
    };
    Type type;
public:
    MUI_Module_Message(UserModule* module,
		       const clString&, clString*, void*, int);
    MUI_Module_Message(UserModule* module,
		       double, double*, void*, int);
    MUI_Module_Message(UserModule* module,
		       int, int*, void*, int);
    virtual ~MUI_Module_Message();

    void do_it();
    UserModule* module;
    void* cbdata;
    int flags;
};

#endif /* SCI_project_MUI_h */
