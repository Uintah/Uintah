#include"Solver.h"

///////////////////////////////////////////////////////////////////////////////
// SOLVER CLASS
///////////////////////////////////////////////////////////////////////////////


Solver::Solver(const BoundingBox& b, const int r)
{
    Init(b,r);
}

//default constructor: the domain/mesh are not initialized
Solver::Solver(){};

Solver::~Solver()
{
	//deleting elements from mesh tree structure
    if(mesh) delete mesh;
    ClearParticles();
}

//if the default constructor was not called
void Solver::SetDomain(const BoundingBox& b, const int r)
{
    if(!flags.solverInitialized)
        Init(b,r);
}

void Solver::GenerateParticles(ParticleGenerator pg)
{
    ClearParticles();
	ParticleDataList p_data;
    pg(p_data, mesh->BBox());
	for(ParticleData d : p_data)
	{
        Particle *p = new Particle(d);
        particles.push_back(p);
	}
	UpdateParticleMap();
}

void Solver::ForceMeshUpdate()
{
    flags.MeshNeedsUpdate();
    mesh->Update();
}

void Solver::Solve(const int n, const double dt, const Output out, const int per_n)
{
	cout << "INITIAL CONDITION: " << endl;
	PrintParticleData();
    n_steps = 0;
    time = 0.0;
	for(int i = 1; i <= n; i++)
	{
        DoTimestep(dt);
		if(out == Output::TO_SCREEN && n_steps % per_n == 0)
		{
			cout << endl << "STEP " << i << endl;
			PrintParticleData();
		}
	}
	cout << "FINAL SOLUTION: " << endl;
	PrintParticleData();
}

void Solver::DoTimestep(const double dt, bool amr)
{
    cout << "\nTime: " << time << " step: " << n_steps << endl;
    //doing mesh adaptation only if amr flag is true
    if(amr)
    {
        AdaptMesh();
        UpdateMesh();
    }
    //project from particles to nodes
    ProjectToNodes();
    //nodes have values now, change node display accordingly
    UpdateNodeImages();
    //do time integration on nodes, projection to particles and time integration on particles
    TimeIntegrateNodes(dt);
    TimeIntegrateParticles(dt);
    //particle positions have been updated so we update particle images to reflect it
    UpdateParticleImages();
    time += dt;
    n_steps++;
}

void Solver::RefineElementByID(const unsigned int id)
{
    mesh->RefineByID(id);
    //flagging mesh for update
    flags.MeshNeedsUpdate();
}

//////////////////////////////////// QT VISUALIZATION	     //////////////////////////////////////////////
void Solver::InitQtScene(QGraphicsScene *s)
{
    //saving scene pointer locally as well as providing it to the mesh class
    scene = s;
    mesh->SetScene(scene);
    //in case the were some mesh changes performed
    UpdateMesh();
    //clearing the scene (in case this is the reset of the simulation)
    scene->clear();
	//Note: the QScene y-axis positive direction is downside, we'll need to invert y-coordinates
	//get domain size (we will scale everything to get a 1000x1000 QScene size
	s_size_x = 1000.0;
	s_size_y = 1000.0;
    //scene.setSceneRect(0.0, 0.0, s_size_x, s_size_y);
    Vec2D from = mesh->BBox().GetFrom();
    Vec2D to = mesh->BBox().GetTo();
	Vec2D dv = to - from;

	//we'll leave 5% of the size of scene for borders, the rest will be used to plot mesh
    scale_x = 0.9*s_size_x/dv[x1];
	scale_y = 0.9*s_size_y/dv[x2];
    flags.sceneInitialized = true;

    //after all the scaling factors are computed we are adding the qt objects
    //that represent elements/particles/nodes to the scene
    UpdateQtScene();
}

//////////////////////////////////// MESH ADAPTATION //////////////////////////////////////////////
void Solver::AdaptMesh()
{
    mesh->Adapt(particles);
    flags.MeshNeedsUpdate();
}


//////////////////////////////////// PRIVATE MEMBERS //////////////////////////////////////////////
// mesh creation and update
void Solver::Init(const BoundingBox& b, const int r)
{
    mesh = new Mesh();
    mesh->SetDomain(b);
    mesh->RefineUniformly(r);
    UpdateMesh();

    //setting time to zero
    time = 0.0;
    //setting number of steps to zero
    n_steps = 0;
    //mesh is not set up
    flags.solverInitialized = true;
}

void Solver::UpdateMesh()
{
    if(!flags.MeshUpdated())
    {
        mesh->Update();
        elements = mesh->Elements();
        nodes = mesh->Nodes();
        //in case of refinement coarsening, we need to update particle to element map
        UpdateParticleMap();
        //if the scene is set up, we need to update the graphical representation as well
        if(flags.sceneInitialized) UpdateQtScene();
        //mesh is now updated, setting corresponding flags
        flags.elementsUpdated = true;
        flags.nodesUpdated = true;
    }
}

void Solver::UpdateElements()
{
    if(!flags.elementsUpdated)
    {
        mesh->UpdateElements();
        elements = mesh->Elements();
        //since the mesh have changed we need to update particle positions with respect to it
        UpdateParticleMap();
        flags.elementsUpdated = true;
        //since we updated elements, we now need to update nodes as well
        //setting the corresponding flag
        flags.nodesUpdated = false;
    }
}

void Solver::UpdateNodes()
{
    if(!flags.nodesUpdated)
    {
        mesh->UpdateNodes();
        nodes = mesh->Nodes();
        flags.nodesUpdated = true;
        //since the nodes updated after elements, the QtScene updated is checked here
        if(flags.sceneInitialized) UpdateQtScene();
    }
}

////////////////////////////// Clean up / reset functionality ////////////////////////////////////

void Solver::ResetNodes()
{
	if(nodes.size() != 0)
	{
		for(Node* n : nodes)
			n->Reset();
	}
}

void Solver::ClearParticles()
{
	if(particles.size() != 0)
	{
		for(Particle* p : particles)
			delete p;
		particles.clear();
	}
}

//////////////////////////// Helper functions ////////////////////////////

Element* Solver::GetElementByID(const unsigned int id)
{
	for(Element* e : elements)
        if(e->ID() == id) return e;
	return nullptr;
}

Node* Solver::GetNodeByID(const unsigned int id)
{
	for(Node* n : nodes)
        if(n->data->id == id) return n;
	return nullptr;
}

Vec2D Solver::GetSceneCoord(const Vec2D &c)
{
    assert(flags.sceneInitialized);
    Vec2D result = {0.05*s_size_x + scale_x*c[x1], 0.05*s_size_y + scale_y*c[x2]};
    //Vec2D result = {0.05*s_size_x + scale_x*c[x1], 0.95*s_size_y - scale_y*c[x2]};
    return result;
}

Vec2D Solver::GetSceneDisp(const Vec2D &d)
{
    assert(flags.sceneInitialized);
    Vec2D result = {scale_x*d[x1],scale_y*d[x2]};
    //Vec2D result = {scale_x*d[x1],-scale_y*d[x2]};
    return result;
}

//////////////////////////// Qt related /////////////////////////////////
void Solver::UpdateParticleImages()
{
    Vec2D s_pos;
    for(Particle* p : particles)
    {
        if(p->hasImage)
        {
            //position update method
            s_pos = GetSceneCoord(p->data.pos);
            p->image->setPos(s_pos[x1],s_pos[x2]);
        }
    }
}

void Solver::UpdateNodeImages()
{
    for(Node* n : nodes)
    {
        /*
        if(abs(n->data->mass) < TOL)
            n->image->setScale(0.1);
        else
            n->image->setScale(1.0);
        */
        if(!n->isActive)
            n->image->setBrush(QBrush(Qt::lightGray));
        else
        {
            if(n->isRegular)
                n->image->setBrush(QBrush(Qt::green));
            else
                n->image->setBrush(QBrush(Qt::yellow));
        }
    }
}

void Solver::UpdateQtScene()
{
    //checking if the scene (scaling initialized)
    assert(flags.sceneInitialized);
    //checking if elements/particles/nodes have qt images and if they do not, create
    //corresponding graphical objects
    Vec2D from, to, dv, s_coord, s_disp;
    for(Element* e : elements)
    {
        if(!e->HasImage())
        {
            from = e->BBox().GetFrom();
            to = e->BBox().GetTo();
            dv = to - from;
            s_coord = GetSceneCoord(from);
            s_disp = GetSceneDisp(dv);
            e->SetImage( new QGraphicsRectItem( QRectF(s_coord[x1], s_coord[x2], s_disp[x1], s_disp[x2]) ) );
            scene->addItem( dynamic_cast<QGraphicsItem*>(e->GetImage()) );

            QPen rect_pen;
            rect_pen.setWidth(2);
            e->GetImage()->setPen(rect_pen);
            e->GetImage()->setZValue(0);
        }
    }
    //initializing particles
    for(Particle* p : particles)
    {
        if(!p->hasImage)
        {
            s_coord = GetSceneCoord(p->data.pos);
            p->image = new QParticleItem(s_coord[x1], s_coord[x2], 0);
            scene->addItem(dynamic_cast<QGraphicsItem*>(p->image));
            p->image->SetData(&(p->data));
            p->hasImage = true;
            p->image->setZValue(2);
        }
    }
    //initializing nodes
    for(Node* n : nodes)
    {
        if(!n->hasImage)
        {
            s_coord = GetSceneCoord(n->data->pos);
            n->SetImage( new QNodeItem(s_coord[x1], s_coord[x2], 0) );
            scene->addItem(dynamic_cast<QGraphicsItem*>(n->GetImage()));
            n->GetImage()->setZValue(1);
        }
    }
}

//////////////////////////// Solve related //////////////////////////////

//this one could be optimized by implementing a search function that returns 
//the first element that contains given particle. For now we'll just take 
//the first element from the list
void Solver::UpdateParticleMap()
{
	p_map.clear();
	for(Particle* p : particles)
	{
        ElementPtrList elements;
        //mesh->GetElementsContaining(e, p->data.pos);
        mesh->GetActiveElementsContaining(elements, p->data.pos);
        assert(!elements.empty());

        Element* cur_element = *elements.begin();
        //if the particle is on the boundary of two elements we place it
        //into the element that has more particles
        if(elements.size() > 1)
            for(Element* e : elements)
                if(e->GetNParticles() > cur_element->GetNParticles())
                    cur_element = e;
        //adding the particle to the map
        unsigned int id = cur_element->ID();
        p_map[id].push_back(p);
        p->data.e_id = id;

        //debug output
        if(elements.size() > 1)
        {
            cout << "\nParticle #" << p->data.id << " is in more than one elements: ";
            for(Element* e : elements)
                cout << e->ID() << " ";
            cout << endl << "We have placed it into element " << cur_element->ID() << endl;
        }
	}
    //once the map is created, we set the number of particles information to the elements
    for(Element* e : elements)
        e->SetNParticles(p_map[e->ID()].size());

}

void Solver::ProjectToNodes()
{
    //resetting node values
    ResetNodes();
    //going over the element to particle map and update node valuse in
    //those elements which has particles inside them
    for(auto& kv : p_map)
    {
        //getting element id and particle list pointer from the key-value pair
        int e_id = kv.first;
        ParticlePtrList p_list = kv.second;
        if(p_list.size() != 0)
        {
            //getting element pointer by its id
            Element *e = GetElementByID(e_id);
            assert(e != nullptr);
            //index of element nodes/bases
            int ind[] = {_BL, _BR, _TR, _TL};
            //for each particle in this element update node values

            //going over element nodes
            for(int i : ind)
            {
                //we doing the projection for all the nodes: regular and hanging alike:
                //the data will either be used or owerwiritten during interpolation
                Node *cur_node = e->GetNodes()[i];
                NodeData *n_data = cur_node->data;
                Basis *cur_basis = e->GetBasis()[i];
                for(Particle* p : p_list)
                {
                    //reference to particle data
                    //we don't need to modify particle data here so reference is const
                    const ParticleData &p_data = p->data;

                    //evaluating current basis function at the particle position (phi_ip)
                    double p_val = cur_basis->Eval(p_data.pos);

                    Vec2D grad = cur_basis->Grad(p_data.pos);
                    //momentum (check that right hand side works as intended)
                    //n_data->momentum[x1] += p_data.m * p_data.vel[x1] * p_val;
                    //n_data->momentum[x2] += p_data.m * p_data.vel[x2] * p_val;
                    n_data->momentum += p_data.m * p_data.vel * p_val;
                    //mass
                    n_data->mass += p_data.m * p_val;
                    //f_int (should invert it when the whole thing is assembled: the term has minus sign in formulation)
                    n_data->f_int[x1] += (p_data.sigma[a11]*grad[x1] + p_data.sigma[a12]*grad[x2])*p_data.V;
                    n_data->f_int[x2] += (p_data.sigma[a21]*grad[x1] + p_data.sigma[a22]*grad[x2])*p_data.V;
                    //f_ext
                    //same as for momentum: check that it is the same as
                    //n_data[x1]->f_ext += p_data.m * p_data.b[x1] * p_val;
                    //n_data[x2]->f_ext += p_data.m * p_data.b[x2] * p_val;
                    n_data->f_ext += p_data.m * p_data.b * p_val;
                    //values have been assigned to current node
                    cur_node->isActive = true;
                }
                //if the current node mass is exactly zero (e.g. one particle entering element and
                //is exactly at the corner node)
                if(n_data->mass == 0.0)
                    cur_node->isActive = false;

                //debug code, checks at which point the nonzero forcing appear at the nodes
                if(n_data->f_int[x1] != 0.0 || n_data->f_int[x2] != 0.0)
                {
                    cout << "\nElement #" << e_id << endl;
                    cout << "Particles: ";
                    for(Particle* p : p_list)
                    {
                        cout << p->data.id << " ";
                    }
                    cout << endl << "Node #" << n_data->id << " has nonzero internal forcing" << endl;
                }
            }
        }
    }
}

void Solver::TimeIntegrateNodes(const double dt)
{
	for(Node* n : nodes)
	{
		//we update all the nodes: hanging and regular.
		//If the hanging node can be interpolated from two active nodes later,
		//the values will be overwritten with the interpolant. But if not
		//we will have hanging nodes already updated by the code below
        if(n->isActive)
		{
			NodeData *n_data = n->data;
			//compute acceleration
            n_data->accel = (n_data->f_int + n_data->f_ext)/n_data->mass;
            //compute velocity from momentum
			n_data->vel = n_data->momentum / n_data->mass;
            /*if(n_data->vel[x1] == 0.0 && n_data->vel[x2] == 0.0)
                cout << "\nNode #" << n_data->id << ": velocity is zero" << endl;*/
            n_data->vel += n_data->accel*dt;
            /*if(n_data->vel[x1] == 0.0 && n_data->vel[x2] == 0.0)
                cout << "\nNode #" << n_data->id << ": velocity is zero again" << endl;*/
        }
	}
	//interpolating the values of the hanging nodes
	for(Node* n : nodes)
	{
		//if a hanging node
		if(!n->isRegular)
		{
			n->Interpolate();
		}
	}
}

void Solver::TimeIntegrateParticles(const double dt)
{
    //NEED TO CHECK: IF THERE IS INITIAL DISPLACEMENT WE NEED TO COMPUTE F
	//gradient of velocity (using node velocity)
	for(auto& kv : p_map)
	{
		//getting element id and particle list pointer from the key-value pair
		int e_id = kv.first;
		ParticlePtrList p_list = kv.second;
		//getting element pointer by its id
		Element *e = GetElementByID(e_id);
		assert(e != nullptr);
		//index of element nodes/bases
		int ind[] = {_BL, _BR, _TR, _TL};
		//for each particle in this element update velocity gradient 
		for(Particle* p : p_list)
		{
			//reference to particle data structure
			ParticleData &p_data = p->data;
            //going over element nodes
			for(int i : ind)
			{
                NodeData *n_data = e->GetNodes()[i]->data;
                Basis *cur_basis = e->GetBasis()[i];
				//evaluating current basis function at the particle position (phi_ip)
				double p_val = cur_basis->Eval(p_data.pos);
				Vec2D grad = cur_basis->Grad(p_data.pos);
				//computing velocity gradient (velocity component gradients form colums of this matrix: 
				//need to check that this is a correct approach)
				//also need to check that += operator works as intended
                p_data.grad_v += {grad[x1]*n_data->vel[x1], grad[x1]*n_data->vel[x2],
					grad[x2]*n_data->vel[x1], grad[x2]*n_data->vel[x2]};
				//updating velocity
				p_data.vel += p_val * n_data->accel * dt;
				//updating displacement
				p_data.disp += p_val * n_data->vel * dt;
			}
			//updating particle p_data
			//updating F
			Mat2D I = {1.0, 0.0, 0.0, 1.0};//identity matrix
			p_data.F = (I + p_data.grad_v*dt) * p_data.F;
			//updating particle positions
			p_data.pos += p_data.disp;
			//displacement is set to zero earlier in order to use displacement to
			//update graphical representation of particles
			//after the position is updated, displacement is set to zero
            p_data.disp = {0.0, 0.0};
            //updating stress tensor
            //p_data.sigma = p_data.y_mod/2.0*(p_data.F - Mat2DInv(p_data.F));
            //strain tensor E
            Mat2D E = (p_data.F + Mat2DTransp(p_data.F))/2.0 - I;
            double traceE = Mat2DTrace(E);
            p_data.sigma = p_data.Lames_param * traceE * I + 2.0 * p_data.shear_mod * E;
		}

	}
	//now, that particle position are updated we need to regenerate element to particle map	
	UpdateParticleMap();
}

//////////////////////////////////// PRINTING AND DEBUGGING  //////////////////////////////////////////////
void Solver::Print()
{
    string header(80,'-');
    assert(nodes.size() != 0);
    cout << endl << header << endl << "Node list:" << endl << header;
    for(Node* n : nodes)
    {
        cout << endl << string(n->isRegular ? "Regular" : "Hanging") << " node " << n->data->id << ": " << n->data->pos;
        cout <<	"\nNeighbor elements: ";
        for(Element* e : n->neighbors){cout << e->ID() << " ";}
        if(!(n->isRegular))
            cout << "\nInterpolated from nodes: " << n->interp[0]->data->id << " and " << n->interp[1]->data->id;
    }
    cout << endl << header << endl << "Element list:" << endl << header;
    for(Element* e : elements)
    {
        cout << endl << e;
    }
    cout << endl;
}

void Solver::PrintNodeData()
{
    string header(80,'-');
    assert(nodes.size() != 0);
    cout << endl << header << endl << "Node values:" << endl << header;
    for(Node* n : nodes)
    {
        cout << endl << endl << string(n->isRegular ? "Regular" : "Hanging") << " node " << n->data->id <<
            endl << "Position: " << n->data->pos;
        cout << endl;
        n->data->Print();
    }
}

void Solver::PrintParticleData()
{
    string header(80,'-');
    cout << endl << header << endl << "Elements and particles:" << endl << header;
    for(auto& kv : p_map)
    {
        cout << endl << "Element " << kv.first << " contains particle(s): ";
        for(Particle* p : kv.second)
            cout << p->data.id << " ";
    }
    cout << endl << endl << "Particle data: " << endl;
    for(Particle* p : particles)
    {
        cout << endl << "Particle " << p->data.id << endl;
        p->data.Print();
    }
    cout << endl;

}
