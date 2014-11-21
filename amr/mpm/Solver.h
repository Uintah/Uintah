#pragma once

#include"Typedefs.h"
#include"Utils.h"
#include"BoundingBox.h"
#include"Basis.h"
#include"Node.h"
#include"Particle.h"
#include"Mesh.h"

using namespace std;

struct SolverFlags
{
    //default constructor
    SolverFlags()
    {
        SetDefaults();
    }
    //setting all the flags to default values (false)
    void SetDefaults()
    {
        solverInitialized = false;
        sceneInitialized = false;
        elementsUpdated = false;
        nodesUpdated = false;
        particlesUpdated = false;
        domainSet = false;
        icSet = false;
        sceneSet = false;
    }
    //has initial mesh been set up?
    bool solverInitialized;
    //did we initialized scene
    bool sceneInitialized;
    //was mesh changed recently
    bool elementsUpdated;
    bool nodesUpdated;
    //whether the domain is set
    bool domainSet;
    //whether initial condition (particle generating function) is set
    bool icSet;
    //whether the scene is set
    bool sceneSet;

    bool MeshUpdated()
    {
        return elementsUpdated && nodesUpdated;
    }
    void MeshNeedsUpdate()
    {
        elementsUpdated = false;
        nodesUpdated = false;
    }

    //do particles need update
    bool particlesUpdated;
};

class Solver
{
	public:
		//empty constructor (does not setup the initial mesh)
		Solver();
		//destructor
		~Solver();
        //setting various general variables: domain, particle generator, scene pointer
        void SetDomain(const BoundingBox& b, const int r = 0){domain = b; initial_refinement = r; flags.domainSet = true;}
        void SetIC(ParticleGenerator pg){ic = pg; flags.icSet = true;}
        void SetScene(QGraphicsScene *s){scene = s; flags.sceneSet = true;}
        //initializing solver
        void Initialize();

		//solves the problem given number of timesteps and timestep size
		//Output option indicates whther to output intermediate data to file, to screen or 
		//not to output it. per_n indicates the number of timesteps the output is produced
		void Solve(const int n, const double dt, const Output out = Output::NONE, const int per_n = 1);
		//performs one timestep
        void DoTimestep(const double dt, bool amr = true);

        void RefineElementByID(const unsigned int id);
		
        /////////////// mesh adaptation ////////////////////////////////////////
        void AdaptMesh();

		//////////////// printing out/debugging ////////////////////////////////
		void Print();
		void PrintNodeData();
		void PrintParticleData();
	private:
        ///////////////// mesh creation and initialization /////////////////////////
        //generate particles: initial condition essentially
        //for now will be hardcoded, but later will accept some sort of
        //generating function/class object
        void GenerateParticles();
        //updates the whole mesh: elements and nodes
        void UpdateMesh();
        ///////////////// Clean up / reset functionality //////////////////////////
		//resets the node data values to zero
		void ResetNodes();
        //delete all the Particle objects (used in Solver class destructor to clean up memory)
		void ClearParticles();

        ///////////////// helper functions ////////////////////////////////////
        Element* GetElementByID(const unsigned int id);
        Node* GetNodeByID(const unsigned int id);
        Vec2D GetSceneCoord(const Vec2D& c);
        Vec2D GetSceneDisp(const Vec2D& d);
		
        //////////////// visualizing with qt ///////////////////////////////////
        //every timestep we update particle positions
        void UpdateParticleImages();
        //setting size of the node images to visibly denote those with significant massUpdateParticleImages();
        void UpdateNodeImages();
        //scene initialization
        void InitQtScene();
        //if the mesh was changed we need to update QtScene (elements and nodes)
        void UpdateQtScene();
		
		///////////////// Sove related /////////////////////////////////////////
		//updating element to particle map
		void UpdateParticleMap();
		//projecting data from particles to nodes
		void ProjectToNodes();
		//computes node values at the next time step
		void TimeIntegrateNodes(const double dt);
		//updates particle data (including position)
		void TimeIntegrateParticles(const double dt);


        //////////////// actual mesh/solver data /////////////////////////////////
        //general settings: domain, refinement level, initial condition, scene
        BoundingBox domain;
        unsigned int initial_refinement;
        ParticleGenerator ic;
        QGraphicsScene *scene;
        //mesh
        Mesh *mesh;
        //flags
        SolverFlags flags;
        //number of steps that solver performed
        unsigned int n_steps;
		//current time
		double time;
        //list of active (leaf-nodes of the tree) elements
        ElementPtrList elements;
		//map from element id to particles
		ElementIDToParticlePtrMap p_map;
		//list of particles
		ParticlePtrList particles;
		//list of nodes
		NodePtrList nodes;
		//visualization related scaling/size parameters
        double s_size_x, s_size_y, scale_x, scale_y, p_radius, n_radius;

};
