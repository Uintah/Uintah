
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

class clString;
class CallbackData;
class EncapsulatorC;
class FileSelectionBoxC;
class MUI_widget;
class MUI_window_private;
class ScaleC;
class UserModule;

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

    UserModule* get_module();
};

class MUI_widget {
protected:
    clString name;
    MUI_window* window;
    void* cbdata;
public:
    MUI_widget(const clString& name, void* cbdata);
    virtual ~MUI_widget();
    virtual void attach(MUI_window*, EncapsulatorC*)=0;
    enum Orientation {
	Horizontal,
	Vertical,
    };
    void set_title(const clString&);
    void set_dimensions(int width, int height);
};

class MUI_slider_real : public MUI_widget {
    ScaleC* scale;
    double* data;
    void drag_callback(CallbackData*, void*);
    void value_callback(CallbackData*, void*);
public:
    enum Style {
	Slider,
	Dial,
	ThumbWheel,
    };
    MUI_slider_real(const clString& name, double* data,
		    Style=Slider, Orientation=Horizontal,
		    void* cbdata=0);
    virtual ~MUI_slider_real();
    virtual void attach(MUI_window*, EncapsulatorC*);

    void set_minmax(double, double, int);
    void set_value(double);
    void set_orientation(MUI_widget::Orientation);
    void set_style(Style);
};

class MUI_onoff_switch : public MUI_widget {
public:
    MUI_onoff_switch(const clString& name, int* data,
		     int value=0, int cbdata=0);
    ~MUI_onoff_switch();
    virtual void attach(MUI_window*, EncapsulatorC*);
};

class MUI_file_selection : public MUI_widget {
    FileSelectionBoxC* sel;
    MUI_window* window;
    clString* filename;
public:
    MUI_file_selection(const clString& name, clString* filename,
		       void* cbdata=0);
    ~MUI_file_selection();
    virtual void attach(MUI_window*, EncapsulatorC*);
};

#endif /* SCI_project_MUI_h */
