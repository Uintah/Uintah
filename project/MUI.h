
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

class UserModule;
class MUI_widget;
class clString;

class MUI_window {
public:
    MUI_window(UserModule*);
    ~MUI_window();
    void attach(MUI_widget*);
    void detach(MUI_widget*);
    void reconfigure();
};

class MUI_widget {
public:
    enum Orientation {
	Horizontal,
	Vertical,
    };
    void set_title(const clString&);
    void set_dimensions(int width, int height);
};

class MUI_slider_real : public MUI_widget {
public:
    MUI_slider_real(const clString& name, double* data,
		   double value=0, int cbdata=0);
    ~MUI_slider_real();

    void set_minmax(double, double, int);
    void set_value(double);
    void set_orientation(MUI_widget::Orientation);
    enum Style {
	Slider,
	Dial,
	Guage,
    };
    void set_style(Style);
};

class MUI_onoff_switch : public MUI_widget {
public:
    MUI_onoff_switch(const clString& name, int* data,
		     int value=0, int cbdata=0);
    ~MUI_onoff_switch();
};

#endif /* SCI_project_MUI_h */
