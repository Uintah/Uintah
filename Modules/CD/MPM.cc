/*
 *  MPM.cc: Read in tecplot file and create scalar field, vector
 *             field, and particle set.
 *
 *  Written by:
 *    Philip Sutton
 *   Department of Computer Science
 *   University of Utah
 *   May 1998
 *
 *  Copyright (C) 1999 SCI Group
 */
  

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Classlib/Array1.h>
#include <Classlib/Array2.h>
#include <Datatypes/ScalarFieldRG.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/VectorFieldRG.h>
#include <Datatypes/VectorFieldPort.h>
#include <Modules/CD/mpmParticleSet.h>
#include <Datatypes/ParticleSetPort.h>
#include <Geometry/Point.h>
#include <TCL/TCLvar.h>
#include <stdio.h>
#include <string.h>
#include <iostream.h>
#include <fstream.h>
#include <stdlib.h>
#include <ctype.h>
#include <Classlib/String.h>
#include <Classlib/Assert.h>

#include "SAFE/src/CD/src/grid/grid.h"

class MPM : public Module {
private:
  // PRIVATE VARIABLES
  TCLstring filebase;
  ScalarFieldOPort *sfout;
  VectorFieldOPort *vfout;
  ParticleSetOPort *psout;
public:
  MPM(const clString& id);
  MPM(const MPM&, int deep);
  virtual ~MPM();
  virtual Module* clone(int deep);
  virtual void execute();
};

extern "C" {
Module* make_MPM(const clString& id)
{
    return new MPM(id);
}
}

MPM::MPM(const clString& id)
: Module("MPM", id, Filter), filebase("filebase", id, this)
{
    // Create the output port
    sfout=new ScalarFieldOPort(this, "ScalarField", ScalarFieldIPort::Atomic);
    vfout=new VectorFieldOPort(this, "VectorField", VectorFieldIPort::Atomic);
    psout=new ParticleSetOPort(this, "ParticleSet", ParticleSetIPort::Atomic);
    add_oport(sfout);
    add_oport(vfout);
    add_oport(psout);
}

MPM::MPM(const MPM& copy, int deep)
: Module(copy, deep), filebase("filebase", id, this)
{
}

MPM::~MPM()
{
}

Module* MPM::clone(int deep)
{
    return new MPM(*this, deep);
}

void MPM::execute()
{
  grid g1;

  string fb(filebase.get()());
  g1.readPoints(fb+".pts");
  g1.readVelocity(fb+".vel");
  g1.readElements(fb+".hex");
  g1.readParticles(fb+".part");
  g1.classifyParticles();

  cout << "Begin time stepping . . " << endl;
 
  for (int i = 0; i <= 75; i++) {
    double time_step = .01;
    //g1.printParticles();
    mpmParticleSet* ps=new mpmParticleSet();
    mpmTimeStep* ts=new mpmTimeStep();
    ts->time=i*time_step;
    arrayIterator<cell> elem_itr(g1.get_elements());
    for (elem_itr.init(); ! elem_itr; elem_itr++) {
	cout << "This cell has " << elem_itr().particle_count() << " particles" << endl;
	list<particle>& list=elem_itr().particleList();
	listIterator<particle> part_itr(list);
	for(part_itr.init(); !part_itr; part_itr++){
	    ts->scalars.add(part_itr().mass());
	    ts->positions.add(Vector(part_itr().real_x(), part_itr().real_y(),
				     part_itr().real_z()));
	}
    }
    ps->add(ts);
    g1.interpolateGridToParticles();
    g1.moveParticles(time_step);
    g1.checkParticlesLeftCells();
    g1.classifyParticles();
    g1.interpolateParticlesToGrid();
    psout->send_intermediate(ps);
  }
}
