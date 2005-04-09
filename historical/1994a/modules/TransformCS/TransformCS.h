
/*
 *  TransformCS.h
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_module_TransformCS_h
#define SCI_project_module_TransformCS_h

#include <UserModule.h>
#include <ContourSet.h>
#include <ContourSetPort.h>

class TransformCS : public UserModule {
    ContourSetIPort* icontour;
    ContourSetOPort* ocontour;
    DBContext *dbcontext_st;
    void lace_contours(ContourSetHandle);
    void transform_cs();
    void initDB();
    void DBCallBack(DBContext*, int, double, double, void*);
    double spacing;
    ContourSetHandle contours;
public:
    TransformCS();
    TransformCS(const TransformCS&, int deep);
    virtual ~TransformCS();
    virtual Module* clone(int deep);
    virtual void execute();
    virtual void ui_button();
    int abort_flag;
//    virtual void mui_callback(void*, int);
};

#endif
