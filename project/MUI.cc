
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

// Someday, we should delete these four lines, when the
// compiler stops griping about const cast away...
#include <X11/Intrinsic.h>
#include "myStringDefs.h"
#include "myXmStrDefs.h"
#include "myShell.h"

#include <MUI.h>
#include <MtXEventLoop.h>
#include <NetworkEditor.h>
#include <NotFinished.h>
#include <UserModule.h>
#include <Mt/DialogShell.h>
extern MtXEventLoop* evl;

struct MUI_window_private {
    DialogShellC* app;
};

MUI_window::MUI_window(UserModule* module)
{
    priv=new MUI_window_private;
    priv->app=new DialogShellC;
    priv->app->Create("sci", "sci", evl->get_display());
//module->netedit->window, module->name());
}

MUI_window::~MUI_window()
{
    delete priv->app;
    delete priv;
}

void MUI_window::attach(MUI_widget*)
{
    NOT_FINISHED("MUI");
}

void MUI_window::detach(MUI_widget*)
{
    NOT_FINISHED("MUI");
}

void MUI_window::reconfigure()
{
    NOT_FINISHED("MUI");
}

void MUI_window::popup()
{
    evl->lock();
    XtPopup(*priv->app, XtGrabNone);
    evl->unlock();
    NOT_FINISHED("MUI");
}

void MUI_widget::set_title(const clString&)
{
    NOT_FINISHED("MUI");
}

MUI_slider_real::MUI_slider_real(const clString&, double*, double, int)
{
    NOT_FINISHED("MUI");
}

MUI_onoff_switch::MUI_onoff_switch(const clString&, int*, int, int)
{
    NOT_FINISHED("MUI");
}
