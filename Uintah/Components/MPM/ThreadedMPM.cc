
#include <Uintah/Components/MPM/ThreadedMPM.h>

#include <Uintah/Grid/Array3Index.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Grid/Grid.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Components/MPM/Matrix3.h>
#include <Uintah/Grid/NCVariable.h>
#include <Uintah/Grid/NodeSubIterator.h>
#include <Uintah/Grid/ParticleSet.h>
#include <Uintah/Grid/ParticleVariable.h>
#include <Uintah/Grid/ProblemSpec.h>
#include <Uintah/Parallel/ProcessorContext.h>
#include <Uintah/Grid/Region.h>
#include <Uintah/Interface/Scheduler.h>
#include <Uintah/Grid/SoleVariable.h>
#include <Uintah/Grid/SubRegion.h>
#include <Uintah/Grid/Task.h>
#include <Uintah/Components/MPM/CompMooneyRivlin.h> // TEMPORARY
#include <SCICore/Geometry/Vector.h>
using SCICore::Geometry::Vector;
#include <SCICore/Geometry/Point.h>
using SCICore::Geometry::Point;
#include <SCICore/Math/MinMax.h>
using SCICore::Math::Max;
#include <SCICore/Thread/Mutex.h>
#include <SCICore/Thread/Time.h>
using SCICore::Thread::Time;
#include <iostream>
using std::cerr;
using std::string;
#include <fstream>
using std::ofstream;

#include "Problem.h"
    static SCICore::Thread::Mutex io("io lock");


ThreadedMPM::ThreadedMPM()
{
}

ThreadedMPM::~ThreadedMPM()
{
}

void ThreadedMPM::problemSetup(const ProblemSpecP&, GridP& grid,
			     DataWarehouseP& dw)
{
    string infile("in.mpm");
    Problem Enigma;
    Enigma.preProcessor(infile);
    LevelP level = grid->getLevel(0);
    double bnds[7];
    Enigma.getBnds(bnds);
    Point lower(bnds[1], bnds[3], bnds[5]);
    Point upper(bnds[2], bnds[4], bnds[6]);
    double dx[4];
    Enigma.getDx(dx);
    Vector diag = upper-lower;
    int nx = (int)(diag.x()/dx[1]+0.5);
    int ny = (int)(diag.y()/dx[2]+0.5);
    int nz = (int)(diag.z()/dx[3]+0.5);
    
    level->addRegion(lower, upper, nx, ny, nz);
    for(Level::const_regionIterator iter=level->regionsBegin();
	iter != level->regionsEnd(); iter++){
	const Region* region=*iter;
	ParticleSet* pset = new ParticleSet();
	ParticleSubset* psubset = new ParticleSubset(pset);
	ParticleVariable<Vector> px(psubset);
	dw->put(px, "p.x", region, 0);
	ParticleVariable<double> pvolume(psubset);
	dw->put(pvolume, "p.volume", region, 0);
	ParticleVariable<double> pmass(psubset);
	dw->put(pmass, "p.mass", region, 0);
	ParticleVariable<Vector> pvel(psubset);
	dw->put(pvel, "p.velocity", region, 0);
	ParticleVariable<Vector> pexternalforce(psubset);
	dw->put(pexternalforce, "p.externalforce", region, 0);
	ParticleVariable<CompMooneyRivlin> pconmod(psubset);
	dw->put(pconmod, "p.conmod", region, 0);
	cerr << "Creating particles for region\n";
	Enigma.createParticles(region, dw);
	cerr << "Done creating " << pset->numParticles() << " particles\n";

	NCVariable<double> gmass;
	dw->allocate(gmass, "g.mass", region, 0);
	NCVariable<Vector> gvelocity;
	dw->allocate(gvelocity, "g.velocity", region, 0);
	NCVariable<Vector> externalforce;
	dw->allocate(externalforce, "g.externalforce", region, 0);
	NCVariable<Vector> internalforce;
	dw->allocate(internalforce, "g.internalforce", region, 0);
	NCVariable<Vector> acceleration;
	dw->allocate(acceleration, "g.acceleration", region, 0);
	NCVariable<Vector> velocity_star;
	dw->allocate(velocity_star, "g.velocity_star", region, 0);

    }
    cerr << "ThreadedMPM::problemSetup not done\n";
}

void ThreadedMPM::computeStableTimestep(const LevelP& level,
				      SchedulerP& sched, DataWarehouseP& dw)
{
    for(Level::const_regionIterator iter=level->regionsBegin();
	iter != level->regionsEnd(); iter++){
	const Region* region=*iter;

	Task* t = new Task("ThreadedMPM::computeStableTimestep", region, dw, dw,
			   this, ThreadedMPM::actuallyComputeStableTimestep);
	t->requires(dw, "velocity", region, 0,
		    ParticleVariable<Vector>::getTypeDescription());
	t->requires(dw, "params", ProblemSpec::getTypeDescription());
	t->computes(dw, "delt", SoleVariable<double>::getTypeDescription());
	t->usesThreads(true);
	sched->addTask(t);
    }
}

void ThreadedMPM::timeStep(double t, double dt,
			 const LevelP& level, SchedulerP& sched,
			 const DataWarehouseP& old_dw, DataWarehouseP& new_dw)
{
    for(Level::const_regionIterator iter=level->regionsBegin();
	iter != level->regionsEnd(); iter++){
	const Region* region=*iter;
	{
	    /*
	     * findOwners
	     *   in(P.X)
	     * out(P.OWNERS)
	     */
	    Task* t = new Task("ThreadedMPM::findOwners",
			       region, old_dw, new_dw,
			       this, ThreadedMPM::findOwners);
	    t->requires(old_dw, "p.x", region, 0,
			ParticleVariable<Point>::getTypeDescription());
	    t->computes(new_dw, "p.owner", region, 0,
			ParticleVariable<ParticleSet::index>::getTypeDescription());
	    t->usesThreads(true);
	    sched->addTask(t);
	}
	{
	    /*
	     * interpolateParticlesToGrid
	     *   in(P.MASS, P.VELOCITY, P.NAT_X)
	     *   operation(interpolate the P.MASS and P.VEL to the grid
	     *             using P.NAT_X and some shape function evaluations)
	     * out(G.MASS, G.VELOCITY)
	     */
	    Task* t = new Task("ThreadedMPM::interpolateParticlesToGrid",
			       region, old_dw, new_dw,
			       this, ThreadedMPM::interpolateParticlesToGrid);
	    t->requires(old_dw, "p.mass", region, 0,
			ParticleVariable<double>::getTypeDescription());
	    t->requires(old_dw, "p.velocity", region, 0,
			ParticleVariable<Vector>::getTypeDescription());
	    t->requires(old_dw, "p.externalforce", region, 0,
			ParticleVariable<Vector>::getTypeDescription());
	    t->requires(old_dw, "p.x", region, 0,
			ParticleVariable<Point>::getTypeDescription());
	    t->computes(new_dw, "g.mass", region, 0,
			NCVariable<double>::getTypeDescription());
	    t->computes(new_dw, "g.velocity", region, 0,
			NCVariable<Vector>::getTypeDescription());
	    t->computes(new_dw, "g.externalforce", region, 0,
			NCVariable<Vector>::getTypeDescription());
	    t->usesThreads(true);
	    sched->addTask(t);
	}

	{
	    /*
	     * computeStressTensor
	     *   in(G.VELOCITY, P.NAT_X, P.CONMOD)
	     *   operation(evaluate the gradient of G.VELOCITY at P.NAT_X, feed
	     *             this into P.CONMOD, which will evaluate and store
	     *             the stress)
	     * out(P.CONMOD)
	     */
	    Task* t = new Task("ThreadedMPM::computeStressTensor",
			       region, old_dw, new_dw,
			       this, ThreadedMPM::computeStressTensor);
	    t->requires(new_dw, "g.velocity", region, 0,
			NCVariable<Vector>::getTypeDescription());
	    t->requires(old_dw, "p.conmod", region, 0,
			ParticleVariable<CompMooneyRivlin>::getTypeDescription());
	    t->requires(old_dw, "delt",
			SoleVariable<double>::getTypeDescription());
	    t->computes(new_dw, "p.conmod", region, 0,
			ParticleVariable<CompMooneyRivlin>::getTypeDescription());
	    t->usesThreads(true);
	    sched->addTask(t);
	}

	{
	    /*
	     * computeInternalForce
	     *   in(P.CONMOD, P.NAT_X, P.VOLUME)
	     *   operation(evaluate the divergence of the stress (stored in
	     *	       P.CONMOD) using P.NAT_X and the gradients of the
	     *             shape functions)
	     * out(G.F_INTERNAL)
	     */
	    Task* t = new Task("ThreadedMPM::computeInternalForce",
			       region, old_dw, new_dw,
			       this, ThreadedMPM::computeInternalForce);
	    t->requires(new_dw, "p.conmod", region, 0,
			ParticleVariable<CompMooneyRivlin>::getTypeDescription());
	    t->requires(old_dw, "p.volume", region, 0,
			ParticleVariable<double>::getTypeDescription());
	    t->computes(new_dw, "g.internalforce", region, 0,
			NCVariable<Vector>::getTypeDescription());
	    t->usesThreads(true);
	    sched->addTask(t);
	}

	{
	    /*
	     * solveEquationsMotion
	     *   in(G.MASS, G.F_INTERNAL)
	     *   operation(acceleration = f/m)
	     *   out(G.ACCELERATION)
	     * 
	     */
	    Task* t = new Task("ThreadedMPM::solveEquationsMotion",
			       region, old_dw, new_dw,
			       this, ThreadedMPM::solveEquationsMotion);
	    t->requires(new_dw, "g.mass", region, 0,
			NCVariable<double>::getTypeDescription());
	    t->requires(new_dw, "g.internalforce", region, 0,
			NCVariable<Vector>::getTypeDescription());
	    t->computes(new_dw, "g.acceleration", region, 0,
			NCVariable<Vector>::getTypeDescription());
	    t->usesThreads(true);
	    sched->addTask(t);
	}

	{
	    /*
	     * integrateAcceleration
	     *   in(G.ACCELERATION, G.VELOCITY)
	     *   operation(v* = v + a*dt)
	     *   out(G.VELOCITY_STAR)
	     * 
	     */
	    Task* t = new Task("ThreadedMPM::integrateAcceleration",
			       region, old_dw, new_dw,
			       this, ThreadedMPM::integrateAcceleration);
	    t->requires(new_dw, "g.acceleration", region, 0,
			NCVariable<Vector>::getTypeDescription());
	    t->requires(new_dw, "g.velocity", region, 0,
			NCVariable<Vector>::getTypeDescription());
	    t->requires(old_dw, "delt",
			SoleVariable<double>::getTypeDescription());
	    t->computes(new_dw, "g.velocity_star", region, 0,
			NCVariable<Vector>::getTypeDescription());
	    t->usesThreads(true);
	    sched->addTask(t);
	}

	{
	    /*
	     * interpolateToParticlesAndUpdate
	     *   in(G.ACCELERATION, G.VELOCITY_STAR, P.NAT_X)
	     *   operation(interpolate acceleration and v* to particles and
	     *             integrate these to get new particle velocity and
	     *             position)
	     * out(P.VELOCITY, P.X, P.NAT_X)
	     */
	    Task* t = new Task("ThreadedMPM::interpolateToParticlesAndUpdate",
			       region, old_dw, new_dw,
			       this, ThreadedMPM::interpolateToParticlesAndUpdate);
	    t->requires(new_dw, "g.acceleration", region, 0,
			NCVariable<Vector>::getTypeDescription());
	    t->requires(new_dw, "g.velocity_star", region, 0,
			NCVariable<Vector>::getTypeDescription());
	    t->requires(old_dw, "p.x", region, 0,
			ParticleVariable<Point>::getTypeDescription());
	    t->requires(old_dw, "delt",
			SoleVariable<double>::getTypeDescription());
	    t->computes(new_dw, "p.velocity", region, 0,
			ParticleVariable<Vector>::getTypeDescription());
	    t->computes(new_dw, "p.x", region, 0,
			ParticleVariable<Point>::getTypeDescription());
	    t->usesThreads(true);
	    sched->addTask(t);
	}
    }
}

#define HEConstant1 100000.0
#define HEConstant2 20000.0
#define HEConstant3 70000.0
#define HEConstant4 1320000.0

inline double getLambda()
{
    double C1 = HEConstant1;
    double C2 = HEConstant2;
    double C4 = HEConstant4;
  
    double PR = (2.*C1 + 5.*C2 + 2.*C4)/(4.*C4 + 5.*C1 + 11.*C2);
    double mu = 2.*(C1 + C2);
    double lambda = 2.*mu*(1.+PR)/(3.*(1.-2.*PR)) - (2./3.)*mu;

    return lambda;
}

inline double getMu()
{
  double mu = 2.*(HEConstant1 + HEConstant2);

  return mu;
}

static ParticleSubset* hack_owned[128];
static ParticleSubset* hack_ghost[128];
static string s_pmass("p.mass");
static string s_pvolume("p.volume");
static string s_px("p.x");
static string s_pvelocity("p.velocity");
static string s_pexternalforce("p.externalforce");
static string s_gmass("g.mass");
static string s_gvelocity("g.velocity");
static string s_gexternalforce("g.externalforce");
static string s_ginternalforce("g.internalforce");
static string s_gacceleration("g.acceleration");
static string s_gvelocity_star("g.velocity_star");
static string s_pconmod("p.conmod");
static string s_delt("delt");

void ThreadedMPM::actuallyComputeStableTimestep(const ProcessorContext* pc,
					      const Region* region,
					      const DataWarehouseP& old_dw,
					      DataWarehouseP& new_dw)
{
    ParticleVariable<double> pmass;
    new_dw->get(pmass, s_pmass, region, 0);
    ParticleVariable<double> pvolume;
    new_dw->get(pvolume, s_pvolume, region, 0);
    ParticleSubset* pset = pmass.getParticleSubset();
    ASSERT(pset == pvolume.getParticleSubset());

    double c_dil = 0;
    double c_rot = 0;
#if 0
    int numParticles = pset->numParticles();
    int start = pc->threadNumber()*numParticles/pc->numThreads();
    int end = (pc->threadNumber()+1)*numParticles/pc->numThreads();

    ParticleSubset::iterator my_end = pset->seek(end);
    for(ParticleSubset::iterator iter = pset->seek(start);
	iter != my_end; iter++){
#else
    pset = hack_owned[pc->threadNumber()];
    if(!pset){
	pset = pmass.getParticleSubset();
    int numParticles = pset->numParticles();
    int start = pc->threadNumber()*numParticles/pc->numThreads();
    int end = (pc->threadNumber()+1)*numParticles/pc->numThreads();

    ParticleSubset::iterator my_end = pset->seek(end);
    for(ParticleSubset::iterator iter = pset->seek(start);
	iter != my_end; iter++){
	ParticleSet::index idx = *iter;
	double lambda = getLambda();
	double mu = getMu();
	double density = pmass[idx]/pvolume[idx];
	c_dil = Max(c_dil, sqrt((lambda + 2.*mu)/density));
	c_rot = Max(c_rot, sqrt(mu/density));
    }
    } else {
    for(ParticleSubset::iterator iter = pset->begin();
       iter != pset->end(); iter++){
#endif
	ParticleSet::index idx = *iter;
	double lambda = getLambda();
	double mu = getMu();
	double density = pmass[idx]/pvolume[idx];
	c_dil = Max(c_dil, sqrt((lambda + 2.*mu)/density));
	c_rot = Max(c_rot, sqrt(mu/density));
    }
    }
    double c = Max(c_rot, c_dil);
    Vector dCell = region->dCell();
    double width = Max(dCell.x(), dCell.y(), dCell.z());
    double delt = 0.5*width/c;

    delt = pc->reduce_min(delt);
    if(pc->threadNumber() == 0)
	new_dw->put(SoleVariable<double>(delt), s_delt);
}

struct OwnerInfo {
    std::vector<ParticleSet::index> owned;
    std::vector<ParticleSet::index> ghost;
};

struct OwnerTable {
    OwnerInfo* info;
};

// This sucks, so we should get rid of it!
static OwnerTable* owners;
struct OI {
    int n;
    int idx[2];
};

static std::vector<OI> px;
static std::vector<OI> py;
static std::vector<OI> pz;

using std::cout;

extern void decompose(int numProcessors, int sizex, int sizey, int sizez,
		      int& numProcessors_x, int& numProcessors_y,
		      int& numProcessors_z);

void makeOwnerArrays(const Region* region,
		     int numProcessors)
{
    px.resize(region->getNx());
    py.resize(region->getNy());
    pz.resize(region->getNz());

    int npx, npy, npz;
    int nodesx = region->getNx()+1;
    int nodesy = region->getNy()+1;
    int nodesz = region->getNz()+1;
    decompose(numProcessors, nodesx, nodesy, nodesz, npx, npy, npz);
    for(int ipx = 0; ipx < npx; ipx++){
	int sx = ipx*nodesx/npx;
	int ex = (ipx+1)*nodesx/npx;
	if(ex >= nodesx-1)
	    ex=nodesx-1;
	int xpstart = ipx* npy*npz;
	if(ipx != 0){
	    px[sx-1].n=2;
	    px[sx-1].idx[1]=xpstart;
	}
	for(int i=sx;i<ex;i++){
	    px[i].n=1;
	    px[i].idx[0]=xpstart;
	}
    }
    for(int ipy = 0; ipy < npy; ipy++){
	int sy = ipy*nodesy/npy;
	int ey = (ipy+1)*nodesy/npy;
	if(ey >= nodesy-1)
	    ey=nodesy-1;
	int ypstart = ipy*npz;
	if(ipy != 0){
	    py[sy-1].n=2;
	    py[sy-1].idx[1]=ypstart;
	}
	for(int i=sy;i<ey;i++){
	    py[i].n=1;
	    py[i].idx[0]=ypstart;
	}
    }
    for(int ipz = 0; ipz < npz; ipz++){
	int sz = ipz*nodesz/npz;
	int ez = (ipz+1)*nodesz/npz;
	int zpstart = ipz;
	if(ez >= nodesz-1)
	    ez=nodesz-1;
	if(ipz != 0){
	    pz[sz-1].n=2;
	    pz[sz-1].idx[1]=zpstart;
	}
	for(int i=sz;i<ez;i++){
	    pz[i].n=1;
	    pz[i].idx[0]=zpstart;
	}
    }
#if 0
    for(int i=0;i<px.size();i++){
	cout << "px[" << i << "]=";
	for(int ii=0;ii<px[i].n;ii++)
	    cout << px[i].idx[ii] << " ";
	cout << '\n';
    }
    for(int i=0;i<py.size();i++){
	cout << "py[" << i << "]=";
	for(int ii=0;ii<py[i].n;ii++)
	    cout << py[i].idx[ii] << " ";
	cout << '\n';
    }
    for(int i=0;i<pz.size();i++){
	cout << "pz[" << i << "]=";
	for(int ii=0;ii<pz[i].n;ii++)
	    cout << pz[i].idx[ii] << " ";
	cout << '\n';
    }
#endif
}

static int findOwners(const Region* region,
		      const Vector& pos, ParticleSet::index list[8],
		      ParticleSet::index& owner)
{
    int ix, iy, iz;
    region->findCell(pos, ix, iy, iz);
    int nx = px[ix].n;
    int ny = py[iy].n;
    int nz = pz[iz].n;
    int total=0;
    owner = px[ix].idx[0]+py[iy].idx[0]+pz[iz].idx[0];
    for(int i=0;i<nx;i++){
	for(int j=0;j<ny;j++){
	    for(int k=0;k<nz;k++){
		if(i != 0 && j != 0 && k != 0)
		    list[total++]=px[ix].idx[i]+py[iy].idx[j]+pz[iz].idx[k];
	    }
	}
    }
    return total;
}

void ThreadedMPM::findOwners(const ProcessorContext* pc,
			     const Region* region,
			     const DataWarehouseP& old_dw,
			     DataWarehouseP& new_dw)
{
    ParticleVariable<Vector> px;
    old_dw->get(px, s_px, region, 0);

#if 0
    ParticleVariable<ParticleSet::index> powner;
    new_dw->create(powner, "p.owner", region, 0);
#endif

    if(pc->threadNumber() == 0){
	if(!owners){
	    owners=new OwnerTable[pc->numThreads()];
	    owners[0].info = new OwnerInfo[pc->numThreads()*pc->numThreads()];
	    makeOwnerArrays(region, pc->numThreads());
	}
    }
    pc->barrier_wait();
    OwnerTable* otab = owners+pc->threadNumber();
    otab->info = owners[0].info + pc->threadNumber()*pc->numThreads();
    for(int i=0;i<pc->numThreads();i++){
	otab->info[i].ghost.clear();
	otab->info[i].owned.clear();
    }

    int total = 0;
    ParticleSubset* pset = px.getParticleSubset();
#if 0
    int numParticles = pset->numParticles();
    int start = pc->threadNumber()*numParticles/pc->numThreads();
    int end = (pc->threadNumber()+1)*numParticles/pc->numThreads();
    ParticleSubset::iterator my_end = pset->seek(end);
    for(ParticleSubset::iterator iter = pset->seek(start);
	iter != my_end; iter++){
#endif
    pset = hack_owned[pc->threadNumber()];
    if(!pset){
	pset = px.getParticleSubset();
	int numParticles = pset->numParticles();
	int start = pc->threadNumber()*numParticles/pc->numThreads();
	int end = (pc->threadNumber()+1)*numParticles/pc->numThreads();
	ParticleSubset::iterator my_end = pset->seek(end);
	for(ParticleSubset::iterator iter = pset->seek(start);
	    iter != my_end; iter++){
	    ParticleSet::index idx = *iter;
	    ParticleSet::index owner;
	    ParticleSet::index glist[8];
	    int n = ::findOwners(region, px[idx], glist, owner);
	    for(int i=0;i<n;i++)
		otab->info[glist[i]].ghost.push_back(idx);
	    otab->info[owner].owned.push_back(idx);
	    total+=n+1;
	}
    } else {
	for(ParticleSubset::iterator iter = pset->begin();
	    iter != pset->end(); iter++){
	    ParticleSet::index idx = *iter;
	    ParticleSet::index owner;
	    ParticleSet::index glist[8];
	    int n = ::findOwners(region, px[idx], glist, owner);
	    for(int i=0;i<n;i++)
		otab->info[glist[i]].ghost.push_back(idx);
	    otab->info[owner].owned.push_back(idx);
	    total+=n+1;
	}
    }
#if 0
    io.lock();
    cerr << pc->threadNumber() << ": found " << total << '\n';
    io.unlock();
#endif
    pc->barrier_wait();

    int ohave = 0;
    int ghave = 0;
    for(int i=0;i<pc->numThreads();i++){
	ghave += owners[i].info[pc->threadNumber()].ghost.size()
	    + owners[i].info[pc->threadNumber()].owned.size();
	ohave += owners[i].info[pc->threadNumber()].owned.size();
    }

    if(!hack_owned[pc->threadNumber()])
       hack_owned[pc->threadNumber()] = new ParticleSubset(pset->getParticleSet());
    if(!hack_ghost[pc->threadNumber()])
       hack_ghost[pc->threadNumber()] = new ParticleSubset(pset->getParticleSet());
    ParticleSubset* osubs = hack_owned[pc->threadNumber()];
    ParticleSubset* gsubs = hack_ghost[pc->threadNumber()];
    osubs->resize(ohave);
    gsubs->resize(ghave);
    int gidx = 0;
    int oidx = 0;
    for(int i=0;i<pc->numThreads();i++){
	unsigned long ng = owners[i].info[pc->threadNumber()].ghost.size();
	std::vector<ParticleSet::index>& gpids = owners[i].info[pc->threadNumber()].ghost;
	for(int j=0;j<ng;j++)
	    gsubs->set(gidx++, gpids[j]);
	unsigned long no = owners[i].info[pc->threadNumber()].owned.size();
	std::vector<ParticleSet::index>& opids = owners[i].info[pc->threadNumber()].owned;
	for(int j=0;j<no;j++){
	    osubs->set(oidx++, opids[j]);
	    gsubs->set(gidx++, opids[j]);
	}
    }
    hack_owned[pc->threadNumber()] = osubs;
    hack_ghost[pc->threadNumber()] = gsubs;

#if 0
    io.lock();
    cerr << pc->threadNumber() << ": have " << have << ": " << subs->numParticles() << '\n';
    io.unlock();
#endif

#if 0
    char buf[100];
    sprintf(buf, "/local/sci/raid0/sparker/part.%02d", pc->threadNumber());
    ofstream out(buf);
    for(ParticleSubset::iterator iter = gsubs->begin();
	iter != gsubs->end(); iter++)
	out << *iter << '\n';
#endif

#if 0
    pc->barrier_wait();
    if(pc->threadNumber() == 0){
	delete[] owners[0].info;
	delete[] owners;
    }

    pc->barrier_wait();
    if(pc->threadNumber() == 0){
	new_dw->put(powner, s_powner, region, 0);
    }
#endif
}

void ThreadedMPM::interpolateParticlesToGrid(const ProcessorContext* pc,
					   const Region* region,
					   const DataWarehouseP& old_dw,
					   DataWarehouseP& new_dw)
{
    // Create arrays for the particle data
    ParticleVariable<Vector> px;
    old_dw->get(px, s_px, region, 0);
    ParticleVariable<double> pmass;
    old_dw->get(pmass, s_pmass, region, 0);
    ParticleVariable<Vector> pvelocity;
    old_dw->get(pvelocity, s_pvelocity, region, 0);
    ParticleVariable<Vector> pexternalforce;
    old_dw->get(pexternalforce, s_pexternalforce, region, 0);

    // Create arrays for the grid data
    NCVariable<double> gmass;
    old_dw->get(gmass, s_gmass, region, 0);
    NCVariable<Vector> gvelocity;
    old_dw->get(gvelocity, s_gvelocity, region, 0);
    NCVariable<Vector> externalforce;
    old_dw->get(externalforce, s_gexternalforce, region, 0);

    ParticleSubset* pset = px.getParticleSubset();
    ASSERT(pset == pmass.getParticleSubset());
    ASSERT(pset == pvelocity.getParticleSubset());

    // Interpolate particle data to Grid data.
    // This currently consists of the particle velocity and mass
    // Need to compute the lumped global mass matrix and velocity
    // Vector from the individual mass matrix and velocity vector (per cell).
    // GridMass * GridVelocity =  S^T*M_D*ParticleVelocity

    SubRegion subregion(region->subregion(pc->threadNumber(), pc->numThreads()));

    NodeSubIterator iter, end;
    region->subregionIteratorPair(pc->threadNumber(), pc->numThreads(),
				  iter, end);
    gmass.initialize(0, iter.x(), iter.y(), iter.z(),
		     end.x(), end.y(), end.z());
    gvelocity.initialize(Vector(0,0,0), iter.x(), iter.y(), iter.z(),
			 end.x(), end.y(), end.z());
    externalforce.initialize(Vector(0,0,0), iter.x(), iter.y(), iter.z(),
			     end.x(), end.y(), end.z());

    pset = hack_ghost[pc->threadNumber()];
    for(ParticleSubset::iterator iter = pset->begin();
	iter != pset->end(); iter++){
	ParticleSet::index idx = *iter;

#if 0
	if(!subregion.contains(px[idx])){
	   cerr << "PARTICLE SUBSET MESSED!\n";
	    continue;
	}
#endif

	// Get the node indices that surround the cell
	Array3Index ni[8];
	double S[8];
	if(!region->findCellAndWeights(px[idx], ni, S))
	    continue;
	// Add each particles contribution to the local mass & velocity 
	// Must use the node indices
	for(int k = 0; k < 8; k++) {
	    if(subregion.contains(ni[k])){
		gmass[ni[k]] += pmass[idx] * S[k];
		gvelocity[ni[k]] += pvelocity[idx] * pmass[idx] * S[k];
		externalforce[ni[k]] += pexternalforce[idx] * S[k];
	    }
	}
    }

    for(; iter != end; iter++){
	if(gmass[*iter] != 0.0){
	    gvelocity[*iter] *= 1./gmass[*iter];
	}
    }
    pc->barrier_wait();
    if(pc->threadNumber() == 0){
	new_dw->put(gmass, s_gmass, region, 0);
	new_dw->put(gvelocity, s_gvelocity, region, 0);
	new_dw->put(externalforce, s_gexternalforce, region, 0);
    }
}

void ThreadedMPM::computeStressTensor(const ProcessorContext* pc,
				      const Region* region,
				      const DataWarehouseP& old_dw,
				      DataWarehouseP& new_dw)
{

    Matrix3 velGrad;
    Vector dx = region->dCell();
    double oodx[3];
    oodx[0] = 1.0/dx.x();
    oodx[1] = 1.0/dx.y();
    oodx[2] = 1.0/dx.z();

    // Create arrays for the particle position
    // and the constitutive model
    ParticleVariable<Vector> px;
    old_dw->get(px, s_px, region, 0);
    ParticleVariable<CompMooneyRivlin> pconmod;
    old_dw->get(pconmod, s_pconmod, region, 0);

    NCVariable<Vector> gvelocity;
    new_dw->get(gvelocity, s_gvelocity, region, 0);
    SoleVariable<double> delt;
    old_dw->get(delt, s_delt);

    ParticleSubset* pset = px.getParticleSubset();
    ASSERT(pset == px.getParticleSubset());
    ASSERT(pset == pconmod.getParticleSubset());

#if 0
    int n = 0 ;
#endif
#if 0
    int numParticles = pset->numParticles();
    int start = pc->threadNumber()*numParticles/pc->numThreads();
    int end = (pc->threadNumber()+1)*numParticles/pc->numThreads();

    ParticleSubset::iterator my_end = pset->seek(end);
    for(ParticleSubset::iterator iter = pset->seek(start);
	iter != my_end; iter++){
#else
    pset = hack_owned[pc->threadNumber()];
    for(ParticleSubset::iterator iter = pset->begin();
       iter != pset->end(); iter++){
#endif
       ParticleSet::index idx = *iter;

       velGrad.set(0.0);
       // Get the node indices that surround the cell
       Array3Index ni[8];
       Vector d_S[8];
       if(!region->findCellAndShapeDerivatives(px[idx], ni, d_S))
	   continue;

        for(int k = 0; k < 8; k++) {
         // While this reflects the smpm version, I think it is
         // slightly wrong, but should give the same answer. JG 1/24/00
	    Vector& gvel = gvelocity[ni[k]];
	    for (int j = 0; j<3; j++){
		for (int i = 0; i<3; i++) {
		    velGrad(i+1,j+1)+=gvel(j) * d_S[k](i) * oodx[i];
		}
	    }
        }

	// Compute the stress tensor at each particle location.
        pconmod[idx].computeStressTensor(velGrad,delt);
#if 0
	n++;
#endif
    }

#if 0
    double sec = Time::currentSeconds();
    io.lock();
    cerr << pc->threadNumber() << ": " << n << " done at " << sec << '\n';
    io.unlock();
#endif
    pc->barrier_wait();

    if(pc->threadNumber() == 0)
	new_dw->put(pconmod, s_pconmod, region, 0);

}

void ThreadedMPM::computeInternalForce(const ProcessorContext* pc,
				     const Region* region,
				     const DataWarehouseP& old_dw,
				     DataWarehouseP& new_dw)
{

    Vector dx = region->dCell();
    double oodx[3];
    oodx[0] = 1.0/dx.x();
    oodx[1] = 1.0/dx.y();
    oodx[2] = 1.0/dx.z();

    // Create arrays for the particle position, volume
    // and the constitutive model
    ParticleVariable<Vector> px;
    old_dw->get(px, s_px, region, 0);
    ParticleVariable<double> pvol;
    old_dw->get(pvol, s_pvolume, region, 0);
    ParticleVariable<CompMooneyRivlin> pconmod;
    old_dw->get(pconmod, s_pconmod, region, 0);

    NCVariable<Vector> internalforce;
    old_dw->get(internalforce, s_ginternalforce, region, 0);

    ParticleSubset* pset = px.getParticleSubset();
    ASSERT(pset == px.getParticleSubset());
    ASSERT(pset == pvol.getParticleSubset());
    ASSERT(pset == pconmod.getParticleSubset());

    SubRegion subregion(region->subregion(pc->threadNumber(), pc->numThreads()));

    NodeSubIterator iter, end;
    region->subregionIteratorPair(pc->threadNumber(), pc->numThreads(),
				  iter, end);
    internalforce.initialize(Vector(0,0,0), iter.x(), iter.y(), iter.z(),
			     end.x(), end.y(), end.z());

    pset = hack_ghost[pc->threadNumber()];
    for(ParticleSubset::iterator iter = pset->begin();
       iter != pset->end(); iter++){
       ParticleSet::index idx = *iter;

#if 0
       if(!subregion.contains(px[idx])){
	   cerr << "PARTICLE SUBSET MESSED!\n";
	    continue;
       }
#endif

       // Get the node indices that surround the cell
       Array3Index ni[8];
       Vector d_S[8];
       if(!region->findCellAndShapeDerivatives(px[idx], ni, d_S))
	   continue;

       for (int k = 0; k < 8; k++){
	   Vector div(d_S[k].x()*oodx[0],d_S[k].y()*oodx[1],d_S[k].z()*oodx[2]);
	    if(subregion.contains(ni[k])){
		internalforce[ni[k]] -=
		    (div * pconmod[idx].getStressTensor() * pvol[idx]);
	    }
       }
    }

    pc->barrier_wait();
    if(pc->threadNumber() == 0)
	new_dw->put(internalforce, s_ginternalforce, region, 0);
}

void ThreadedMPM::solveEquationsMotion(const ProcessorContext* pc,
				     const Region* region,
				     const DataWarehouseP& old_dw,
				     DataWarehouseP& new_dw)
{
    // Get required variables for this region
    NCVariable<double> mass;
    new_dw->get(mass, s_gmass, region, 0);
    NCVariable<Vector> internalforce;
    new_dw->get(internalforce, s_ginternalforce, region, 0);
    NCVariable<Vector> externalforce;
    new_dw->get(externalforce, s_gexternalforce, region, 0);

    // Create variables for the results
    NCVariable<Vector> acceleration;
    old_dw->get(acceleration, s_gacceleration, region, 0);

    // Do the computation of a = F/m for nodes where m!=0.0
    NodeSubIterator iter, end;
    region->subregionIteratorPair(pc->threadNumber(), pc->numThreads(),
				  iter, end);
    for(; iter != end; iter++){
	if(mass[*iter]>0.0){
	  acceleration[*iter] = (internalforce[*iter] + externalforce[*iter])/
						mass[*iter];
	} else {
	  acceleration[*iter] = Vector(0, 0, 0);
	}
    }

    // Put the result in the datawarehouse
    pc->barrier_wait();
    if(pc->threadNumber() == 0)
	new_dw->put(acceleration, s_gacceleration, region, 0);

}

void ThreadedMPM::integrateAcceleration(const ProcessorContext* pc,
				      const Region* region,
				      const DataWarehouseP& old_dw,
				      DataWarehouseP& new_dw)
{
    // Get required variables for this region
    NCVariable<Vector> acceleration;
    new_dw->get(acceleration, s_gacceleration, region, 0);
    NCVariable<Vector> velocity;
    new_dw->get(velocity, s_gvelocity, region, 0);
    SoleVariable<double> delt;
    old_dw->get(delt, s_delt);

    // Create variables for the results
    NCVariable<Vector> velocity_star;
    old_dw->get(velocity_star, s_gvelocity_star, region, 0);

    // Do the computation
    NodeSubIterator iter, end;
    region->subregionIteratorPair(pc->threadNumber(), pc->numThreads(),
				  iter, end);
    for(; iter != end; iter++)
	velocity_star[*iter] = velocity[*iter] + acceleration[*iter] * delt;

    // Put the result in the datawarehouse
    pc->barrier_wait();
    if(pc->threadNumber() == 0)
	new_dw->put(velocity_star, s_gvelocity_star, region, 0);
}

void ThreadedMPM::interpolateToParticlesAndUpdate(const ProcessorContext* pc,
						const Region* region,
						const DataWarehouseP& old_dw,
						DataWarehouseP& new_dw)
{
    // Performs the interpolation from the cell vertices of the grid
    // acceleration and velocity to the particles to update their
    // velocity and position respectively

    // Get the arrays of particle values to be changed
    ParticleVariable<Vector> px;
    old_dw->get(px, s_px, region, 0);
    ParticleVariable<Vector> pvelocity;
    old_dw->get(pvelocity, s_pvelocity, region, 0);

    // Get the arrays of grid data on which the new particle values depend
    NCVariable<Vector> gvelocity_star;
    new_dw->get(gvelocity_star, s_gvelocity_star, region, 0);
    NCVariable<Vector> gacceleration;
    new_dw->get(gacceleration, s_gacceleration, region, 0);
    SoleVariable<double> delt;
    old_dw->get(delt, s_delt);

    ParticleSubset* pset = px.getParticleSubset();
    ASSERT(pset == pvelocity.getParticleSubset());

    Vector vel(0.0,0.0,0.0);
    Vector acc(0.0,0.0,0.0);

    double ke=0;
    pset = hack_owned[pc->threadNumber()];
    for(ParticleSubset::iterator iter = pset->begin();
       iter != pset->end(); iter++){
      ParticleSet::index idx = *iter;

      // Get the node indices that surround the cell
      Array3Index ni[8];
      double S[8];
      if(!region->findCellAndWeights(px[idx], ni, S))
	  continue;


      vel = Vector(0.0,0.0,0.0);
      acc = Vector(0.0,0.0,0.0);

      // Accumulate the contribution from each surrounding vertex
      for (int k = 0; k < 8; k++) {
          vel += gvelocity_star[ni[k]]  * S[k];
          acc += gacceleration[ni[k]]   * S[k];
      }

      // Update the particle's position and velocity
      px[idx]        += vel * delt;
      pvelocity[idx] += acc * delt;
      ke += pvelocity[idx].length2();

     // If we were storing particles in cellwise lists, this
     // is where we would update the lists so that each particle
     // is in the correct cells list
    }
    pc->barrier_wait();
    if(pc->threadNumber() == 0){
	static ofstream tmpout("tmp.out");
	static int ts=0;
	tmpout << ts << " " << ke << std::endl;
    
	static ofstream tmpout2("tmp2.out");
	tmpout2 << ts << " " << px[5] << std::endl;
	ts++;

	// Store the new result
	new_dw->put(px, s_px, region, 0);
	new_dw->put(pvelocity, s_pvelocity, region, 0);

	ParticleVariable<double> pmass;
	old_dw->get(pmass, s_pmass, region, 0);
	new_dw->put(pmass, s_pmass, region, 0);
	ParticleVariable<double> pvolume;
	old_dw->get(pvolume, s_pvolume, region, 0);
	new_dw->put(pvolume, s_pvolume, region, 0);
	ParticleVariable<Vector> pexternalforce;
	old_dw->get(pexternalforce, s_pexternalforce, region, 0);
	new_dw->put(pexternalforce, s_pexternalforce, region, 0);
    }
}
