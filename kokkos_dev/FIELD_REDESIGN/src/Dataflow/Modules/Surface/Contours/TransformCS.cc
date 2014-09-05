/*
 *  TransformCS.cc:  Transform a contour set, and output it
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <SCICore/Util/NotFinished.h>
#include <PSECommon/Dataflow/Module.h>
#include <PSECommon/Datatypes/ContourSet.h>
#include <PSECommon/Datatypes/ContourSetPort.h>
// #include <Devices/Dialbox.h>
// #include <Devices/DBContext.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Math/Expon.h>
#include <SCICore/Math/Trig.h>


class TransformCS : public Module {
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
    TransformCS(const clString&);
    TransformCS(const TransformCS&, int deep);
    virtual ~TransformCS();
    virtual Module* clone(int deep);
    virtual void execute();
    virtual void ui_button();
};

extern "C" {
Module* make_TransformCS(const clString& id)
{
    return new TransformCS(id);
}
}

TransformCS::TransformCS(const clString& id)
: Module("TransformCS", id, Filter)
{
    // Create the input port
    icontour=new ContourSetIPort(this, "ContourSet", ContourSetIPort::Atomic);
    add_iport(icontour);
    ocontour=new ContourSetOPort(this, "ContourSet", ContourSetIPort::Atomic);
    add_oport(ocontour);
    spacing=0;
//    dbcontext_st=0;
}

void TransformCS::ui_button() {
/*    if (!dbcontext_st) {
	initDB();
    }
    Dialbox::attach_dials(dbcontext_st);*/
}

TransformCS::TransformCS(const TransformCS&copy, int deep)
: Module(copy, deep)
{
    NOT_FINISHED("TransformCS::TransformCS");
}

TransformCS::~TransformCS()
{
}

Module* TransformCS::clone(int deep)
{
    return new TransformCS(*this, deep);
}

void TransformCS::transform_cs() {
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
void TransformCS::initDB() {
#ifdef OLDUI
    dbcontext_st=new DBContext(clString("TransformCS"));
    dbcontext_st->set_knob(0, "Scale",
			   new DBCallback<TransformCS>
			   FIXCB2(&netedit->mailbox, this,
				  &TransformCS::DBCallBack, 0));
    dbcontext_st->set_range(0, -.1, .1);
    dbcontext_st->set_value(0, 0.0);
    dbcontext_st->set_knob(1, "Spacing",
			   new DBCallback<TransformCS>
			   FIXCB2(&netedit->mailbox, this,
				  &TransformCS::DBCallBack, 0));
    dbcontext_st->set_scale(1, 5);
    dbcontext_st->set_value(1, 0);
    dbcontext_st->set_knob(6, "Translate X",
			   new DBCallback<TransformCS>
			   FIXCB2(&netedit->mailbox, this,
				  &TransformCS::DBCallBack, 0));
    dbcontext_st->set_scale(6, 100);	
    dbcontext_st->set_knob(4, "Translate Y",
			   new DBCallback<TransformCS>
			   FIXCB2(&netedit->mailbox, this,
				  &TransformCS::DBCallBack, 0));
    dbcontext_st->set_scale(4, 100);
    dbcontext_st->set_knob(2, "Translate Z",
			   new DBCallback<TransformCS>
			   FIXCB2(&netedit->mailbox, this,
				  &TransformCS::DBCallBack, 0));
    dbcontext_st->set_scale(2, 100);
    dbcontext_st->set_knob(7, "Rotate X",
			   new DBCallback<TransformCS>
			   FIXCB2(&netedit->mailbox, this,
				  &TransformCS::DBCallBack, 0));
    dbcontext_st->set_wraprange(7, 0, 10);
    dbcontext_st->set_knob(5, "Rotate Y",
			   new DBCallback<TransformCS>
			   FIXCB2(&netedit->mailbox, this,
				  &TransformCS::DBCallBack, 0));
    dbcontext_st->set_wraprange(5, 0, 10);
    dbcontext_st->set_knob(3, "Rotate Z",
			   new DBCallback<TransformCS>
			   FIXCB2(&netedit->mailbox, this,
				  &TransformCS::DBCallBack, 0));
    dbcontext_st->set_wraprange(3, 0, 10);
#endif
}*/
void TransformCS::execute()
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
void TransformCS::DBCallBack(DBContext*, int, double, double, void*) {
    if(!abort_flag){
	abort_flag=1;
	want_to_execute();
    }
}
*/
