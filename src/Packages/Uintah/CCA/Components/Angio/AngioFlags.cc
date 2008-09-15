#include <Packages/Uintah/CCA/Components/Angio/AngioFlags.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/LinearInterpolator.h>
#include <Packages/Uintah/Core/Grid/Node27Interpolator.h>
#include <Packages/Uintah/Core/Grid/TOBSplineInterpolator.h>
#include <Packages/Uintah/Core/Grid/BSplineInterpolator.h>
//#include <Packages/Uintah/Core/Grid/AMRInterpolator.h>
#include <Core/Util/DebugStream.h>
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

using namespace Uintah;
using namespace SCIRun;
using namespace std;

static DebugStream dbg("AngioFlags", false);

AngioFlags::AngioFlags()
{
  d_interpolator_type = "linear";
  d_integrator_type = "explicit";
  d_interpolator = scinew LinearInterpolator();
  d_integrator = Explicit;
}

AngioFlags::~AngioFlags()
{
  delete d_interpolator;
}

void
AngioFlags::readAngioFlags(ProblemSpecP& ps)
{
  ProblemSpecP root = ps->getRootNode();
  ProblemSpecP angio_flag_ps = root->findBlock("Angio");

  if (!angio_flag_ps)
    return;

  angio_flag_ps->get("time_integrator", d_integrator_type);
  if (d_integrator_type == "implicit") 
    d_integrator = Implicit;
  else{
    d_integrator = Explicit;
  }
  angio_flag_ps->get("time_integrator", d_integrator_type);
  int junk=0;
  angio_flag_ps->get("nodes8or27", junk);
  if(junk!=0){
     cerr << "nodes8or27 is deprecated, use " << endl;
     cerr << "<interpolator>type</interpolator>" << endl;
     cerr << "where type is one of the following:" << endl;
     cerr << "linear, gimp, 3rdorderBS" << endl;
    exit(1);
  }

  angio_flag_ps->get("Growth_Parameter_a",  d_Grow_a);
  angio_flag_ps->get("Growth_Parameter_b",  d_Grow_b);
  angio_flag_ps->get("Growth_Parameter_x0", d_Grow_x0);
  angio_flag_ps->get("Growth_Parameter_y0", d_Grow_y0);

  angio_flag_ps->get("Branch_Parameter_a1",   d_Branch_a1);
  angio_flag_ps->get("Branch_Parameter_a2",   d_Branch_a2);
  angio_flag_ps->get("Branch_Parameter_a3",   d_Branch_a3);

  delete d_interpolator;

  if(d_interpolator_type=="linear"){
    d_interpolator = scinew LinearInterpolator();
    d_8or27 = 8;
  } else if(d_interpolator_type=="gimp"){
    d_interpolator = scinew Node27Interpolator();
    d_8or27 = 27;
  } else if(d_interpolator_type=="3rdorderBS"){
    d_interpolator = scinew TOBSplineInterpolator();
    d_8or27 = 27;
  } else if(d_interpolator_type=="4thorderBS"){
    d_interpolator = scinew BSplineInterpolator();
    d_8or27 = 64;
  } else{
    ostringstream warn;
    warn << "ERROR:Angio: invalid interpolation type ("<<d_interpolator_type << ")"
         << "Valid options are: \n"
         << "linear\n"
         << "gimp\n"
         << "3rdorderBS\n"
         << "4thorderBS\n"<< endl;
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__ );
  }

  if (dbg.active()) {
    dbg << "---------------------------------------------------------\n";
    dbg << "Angio Flags " << endl;
    dbg << "---------------------------------------------------------\n";
    dbg << " Time Integration            = " << d_integrator_type << endl;
    dbg << " Interpolation type          = " << d_interpolator_type << endl;
    dbg << "---------------------------------------------------------\n";
  }
}

void
AngioFlags::outputProblemSpec(ProblemSpecP& ps)
{
  ps->appendElement("time_integrator", d_integrator_type);
  ps->appendElement("interpolator", d_interpolator_type);
}
