/*
 *  TransformContours.cc:  Transform a contour set, and output it
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Core/Util/NotFinished.h>
#include <Dataflow/Dataflow/Module.h>
#include <Dataflow/Datatypes/ContourSet.h>
#include <Dataflow/Datatypes/ContourSetPort.h>
// #include <Devices/Dialbox.h>
// #include <Devices/DBContext.h>
#include <Core/Geometry/Vector.h>
#include <Core/Math/Expon.h>
#include <Core/Math/Trig.h>


using namespace SCIRun;

class TransformContours : public Module {
    ContourSetIPort* icontour;
    ContourSetOPort* ocontour;
//    DBContext *dbcontext_st;
    void lace_contours(ContourSetHandle);
    void transform_cs();
//    void initDB();
//    void DBCallBack(DBContext*, int, double, double, void*);
    double spacing;
    ContourSetHandle contours;
public:
    TransformContours(const clString&);
    TransformContours(const TransformContours&, int deep);
    virtual ~TransformContours();
    virtual Module* clone(int deep);
    virtual void execute();
    virtual void ui_button();
};

extern "C" {
Module* make_TransformContours(const clString& id)
{
    return new TransformContours(id);
}
}

TransformContours::TransformContours(const clString& id)
: Module("TransformContours", id, Filter)
{
    // Create the input port
    icontour=new ContourSetIPort(this, "ContourSet", ContourSetIPort::Atomic);
    add_iport(icontour);
    ocontour=new ContourSetOPort(this, "ContourSet", ContourSetIPort::Atomic);
    add_oport(ocontour);
    spacing=0;
//    dbcontext_st=0;
}

void TransformContours::ui_button() {
/*    if (!dbcontext_st) {
	initDB();
    }
    Dialbox::attach_dials(dbcontext_st);*/
}

TransformContours::TransformContours(const TransformContours&copy, int deep)
: Module(copy, deep)
{
    NOT_FINISHED("TransformContours::TransformContours");
}

TransformContours::~TransformContours()
{
}

Module* TransformContours::clone(int deep)
{
    return new TransformContours(*this, deep);
}

void TransformContours::transform_cs() {
/*    contours->translate(Vector(dbcontext_st->get_value(6),
			       dbcontext_st->get_value(4),
			       dbcontext_st->get_value(2)));
    cerr << "Bases after translate:\n";
    cerr << contours->basis[0].string() << "\n";
    cerr << contours->basis[1].string() << "\n";
    cerr << contours->basis[2].string() << "\n";
    contours->rotate(Vector(DtoR(dbcontext_st->get_value(7)),
			    DtoR(dbcontext_st->get_value(5)),
			    DtoR(dbcontext_st->get_value(3))));
    cerr << "Bases after rotate:\n";
    cerr << contours->basis[0].string() << "\n";
    cerr << contours->basis[1].string() << "\n";
    cerr << contours->basis[2].string() << "\n";
    contours->scale(Exp10(dbcontext_st->get_value(0)));
    contours->space=dbcontext_st->get_value(1);*/
}
/*
void TransformContours::initDB() {
#ifdef OLDUI
    dbcontext_st=new DBContext(clString("TransformContours"));
    dbcontext_st->set_knob(0, "Scale",
			   new DBCallback<TransformContours>
			   FIXCB2(&netedit->mailbox, this,
				  &TransformContours::DBCallBack, 0));
    dbcontext_st->set_range(0, -.1, .1);
    dbcontext_st->set_value(0, 0.0);
    dbcontext_st->set_knob(1, "Spacing",
			   new DBCallback<TransformContours>
			   FIXCB2(&netedit->mailbox, this,
				  &TransformContours::DBCallBack, 0));
    dbcontext_st->set_scale(1, 5);
    dbcontext_st->set_value(1, 0);
    dbcontext_st->set_knob(6, "Translate X",
			   new DBCallback<TransformContours>
			   FIXCB2(&netedit->mailbox, this,
				  &TransformContours::DBCallBack, 0));
    dbcontext_st->set_scale(6, 100);	
    dbcontext_st->set_knob(4, "Translate Y",
			   new DBCallback<TransformContours>
			   FIXCB2(&netedit->mailbox, this,
				  &TransformContours::DBCallBack, 0));
    dbcontext_st->set_scale(4, 100);
    dbcontext_st->set_knob(2, "Translate Z",
			   new DBCallback<TransformContours>
			   FIXCB2(&netedit->mailbox, this,
				  &TransformContours::DBCallBack, 0));
    dbcontext_st->set_scale(2, 100);
    dbcontext_st->set_knob(7, "Rotate X",
			   new DBCallback<TransformContours>
			   FIXCB2(&netedit->mailbox, this,
				  &TransformContours::DBCallBack, 0));
    dbcontext_st->set_wraprange(7, 0, 10);
    dbcontext_st->set_knob(5, "Rotate Y",
			   new DBCallback<TransformContours>
			   FIXCB2(&netedit->mailbox, this,
				  &TransformContours::DBCallBack, 0));
    dbcontext_st->set_wraprange(5, 0, 10);
    dbcontext_st->set_knob(3, "Rotate Z",
			   new DBCallback<TransformContours>
			   FIXCB2(&netedit->mailbox, this,
				  &TransformContours::DBCallBack, 0));
    dbcontext_st->set_wraprange(3, 0, 10);
#endif
}*/
void TransformContours::execute()
{
/*    if (!dbcontext_st) {
	initDB();
    }
    */    if (!icontour->get(contours))
	return;
    if (contours->space!=spacing) {
	spacing=contours->space;
//	dbcontext_st->set_value(1, spacing);
    }
    contours.detach();
    cerr << "Bases before:\n";
    cerr << contours->basis[0].string() << "\n";
    cerr << contours->basis[1].string() << "\n";
    cerr << contours->basis[2].string() << "\n";
    transform_cs();
    cerr << "Bases after scale:\n";
    cerr << contours->basis[0].string() << "\n";
    cerr << contours->basis[1].string() << "\n";
    cerr << contours->basis[2].string() << "\n";
    ocontour->send(contours);
}
/*
void TransformContours::DBCallBack(DBContext*, int, double, double, void*) {
    if(!abort_flag){
	abort_flag=1;
	want_to_execute();
    }
}
*/
