#include "Mesh.h"

using namespace std;
///////////////////////////////////////////////////////////////////////////////
// ELEMENT CLASS
///////////////////////////////////////////////////////////////////////////////
//static member
Mesh* Element::root;
//comparison and printing operators
bool operator<(const Element& a, const Element& b)
{
    if(a.ID() < b.ID())
		return true;
	else
		return false;
}
bool operator==(const Element& a, const Element& b)
{
    if(a.ID() == b.ID())
		return true;
	else
		return false;
}	

ostream& operator<<(ostream& os, const Element& e)
{
	std::ios state(NULL);
	state.copyfmt(os);
	os.precision(3);
	os << fixed;
	
	//os << "Element " << e.id << " bottom left: " << e.bBox.GetFrom() << " top right: " << e.bBox.GetTo();
    os << "Element " << e.ID() << " with nodes: ";
    for(Node* n : e.GetNodes()){cout << n->data->id << " ";}
	//int ind[] = {_BL, _BR, _TR, _TL};
	//for(int i : ind){cout << e.nodes[i]->id << " ";}
	
	os.copyfmt(state);
	return os;
}
/////////////////////////////////////////////////////////////////////////////////////
// CLASS IMPLEMENTATION
/////////////////////////////////////////////////////////////////////////////////////
//default constructor
Element::Element(){};
//constructor
Element::Element(const int l, const BoundingBox& b, Element* p):
    level(l), bBox(b), parent(p), score(0), action(MeshAction::Keep)
{
    id = root->GetNextID();
	//setting up vertices
	Vec2D bl = bBox.GetFrom();
	Vec2D tr = bBox.GetTo();
	Vec2D br = {tr[x1], bl[x2]};
	Vec2D tl = {bl[x1], tr[x2]};
	vertices[_BL] = bl;
	vertices[_BR] = br;
	vertices[_TR] = tr;
	vertices[_TL] = tl;
	//setting up basis functions
	Vec2D d = tr - bl;
	basis[_BL] = new Basis(bl, d[x1], d[x2]);
	basis[_BR] = new Basis(br, -d[x1], d[x2]);
	basis[_TR] = new Basis(tr, -d[x1], -d[x2]);
	basis[_TL] = new Basis(tl, d[x1], -d[x2]);
    //area of the element
    area = d[x1]*d[x2];
}

//destructor
Element::~Element()
{
    if(flags.hasChildren)
	{
		for(Element* c : children)
		{
			//cout << endl << "Deleting element " << c->id;
			if(c) delete c;
		}
	}

    for(Basis* b : basis)
		if(b) delete b;

    if(flags.hasImage) delete image;
}

//recursive refinement (refines n times)
void Element::Refine(const unsigned int n)
{
    if(n == 0) return;//recursion termination
    else
    {
        if(!flags.hasChildren)
        {
            //create 4 new bounding boxes
            //first create points at the two corners, center and at mid-sides
            Vec2D bl = bBox.GetFrom();
            Vec2D tr = bBox.GetTo();
            Vec2D cnt = bl + (tr - bl)/2.0;
            Vec2D b = {cnt[x1], bl[x2]};
            Vec2D t = {cnt[x1], tr[x2]};
            Vec2D l = {bl[x1], cnt[x2]};
            Vec2D r = {tr[x1], cnt[x2]};
            //now create bounding boxes
            BoundingBox b_bl(bl, cnt), b_br(b, r), b_tr(cnt, tr), b_tl(l, t);
            children.resize(4);
            int next_level = level + 1;
            children[_BL] = new Element(next_level, b_bl, this);
            children[_BR] = new Element(next_level, b_br, this);
            children[_TR] = new Element(next_level, b_tr, this);
            children[_TL] = new Element(next_level, b_tl, this);
            flags.hasChildren = true;
        }
        //refining children
        children[_BL]->Refine(n-1);
        children[_BR]->Refine(n-1);
        children[_TR]->Refine(n-1);
        children[_TL]->Refine(n-1);

        //cout << "\n\nRefining element: " << id;
		//cout << "\nbounding box " << bBox.GetFrom() << "  " << bBox.GetTo();
		//cout << "\nNew Elements: ";
		//int ind[] = {_BL, _BR, _TR, _TL};
		//for(int i : ind)
		//{
		//	cout << "\nChild Element " << children[i]->id;
		//	cout << "\nbounding box " << children[i]->bBox.GetFrom() << "  " << children[i]->bBox.GetTo();
		//}

	}
}

void Element::FillActiveElementList(ElementPtrList& e, const bool check_root)
{
    //performing that check_root is requred and that this node is indeed the root of the tree
    assert(!(check_root && this != root));

    //if element is not active and has children, continue the recursion
    if(!IsActive() && HasChildren())
    {

        //proceeding to element's children
        for(Element* ch : children)
        {
            ch->FillActiveElementList(e, false);//not checking for root anymore
        }
    }
    else
    {
        e.push_back(this);
    }
}

void Element::GetActiveElementsContaining(ElementPtrList& e, const Vec2D& v, const bool check_root)
{
    //performing that check_root is requred and that this node is indeed the root of the tree
    if(check_root)
    {
        assert(this == root);
        //if domain does not contain the point, throw an exception
        if(!this->BBox().Contains(v))
        {
            cout << "\nVertex " << v << "is outside domain" << endl;
            throw Error::P_OUTSIDE_DOMAIN;
        }
    }

    //if has children, recurse down the tree
    if(!IsActive() && HasChildren())
    {
        for(Element* ch : children)
        {
            if(ch->BBox().Contains(v))
            {
                ch->GetActiveElementsContaining(e, v, false);//not checking for root anymore
            }
        }
    }
    else
    {
        e.push_back(this);
    }
}

bool Element::IsVertex(const Vec2D& v)
{
	assert(bBox.Contains(v));
	for(Vec2D vt : vertices)
	{
		if(vt == v) return true;//one of the vertices
	}
	return false;//not one of the vertices, must be inside
}

void Element::AppendNodeCoords(CoordList& lst)
{
	lst.push_back(vertices[_BL]);
	lst.push_back(vertices[_BR]);
	lst.push_back(vertices[_TR]);
	lst.push_back(vertices[_TL]);
}

void Element::FillInterpolationNodes(Node* n)
{
    Vec2D p = n->data->pos;
	//check that the given node is actually inside the element
	assert(bBox.Contains(p));
	//if the node is on the bottom edge y = bl_y
	if(vertices[_BL][x2] == p[x2])
	{
		n->interp[0] = nodes[_BL];
		n->interp[1] = nodes[_BR];
	}
	//if the node is on the left edge x = bl_x
	else if(vertices[_BL][x1] == p[x1])
	{
		n->interp[0] = nodes[_BL];
		n->interp[1] = nodes[_TL];
	}
	//if the node is on the right edge x = tr_x
	else if(vertices[_TR][x1] == p[x1])
	{
		n->interp[0] = nodes[_BR];
		n->interp[1] = nodes[_TR];
	}
	//if the node is on the top edge y = tr_y
	else if(vertices[_TR][x2] == p[x2])
	{
		n->interp[0] = nodes[_TL];
		n->interp[1] = nodes[_TR];
	}
	else
	{
		cout << "\nElement::FillInterpolationNodes error: node is not on one of edges!\n";
	}
}

void Element::UpdateMetrics(const ParticlePtrList& p_list)
{
    ParticlePtrList inside;
    for(Particle *p : p_list)
    {
        //we can also experiment with StrictlyContains
        if(bBox.Contains(p->data.pos))
            inside.push_back(p);
    }
    //setting score and concentration
    CalculateMetrics(inside);
    //if there are no particles in this element terminating recursion
    if(inside.size() == 0) return;
    //otherwise continue recursion
    else
    {
        //when do we need to procede to child elements (create them if necessary)
        //1) when current element is much bigger than the volume of particles (concentration < 1/k)
        //2) when we have too many particles in current element (score > 1*k)
        if(concentration < 0.5 || score > 1.5)
        {
            //if the element does not have children, we refine it
            if(!HasChildren()) Refine(1);
            //continuing the recursive scoring process
            for(Element* ch : children)
            {
                ch->UpdateMetrics(inside);
            }
        }
        else return;
        /*
        //if score is less than one (fewer particles than would be "ideal") or one (ideal) we
        //ternimate the recursion
        if(score <= 1.0) return;
        //if there are still too many particles per element
        else
        {
            //if the element does not have children, we refine it
            if(!HasChildren()) Refine(1);
            //continuing the recursive scoring process
            for(Element* ch : children)
            {
                ch->UpdateMetrics(inside);
            }
        }
        */
    }
}

void Element::SetActiveElements(const bool enc_active)
{
    //if there are no child nodes or the score is zero (no particles), set active, terminate the recursion
    if(score == 0.0 || !flags.hasChildren)
    {
        SetActive();
        return;
    }
    else
    {
        //if the current element is active then for the following elements encoutered_active flag will
        //be set to true. If not then the flag that was given as an argument will be passed along
        bool active_flag = IsActive() ? true : enc_active;
        //children average score
        double ch_score, ch_conc;
        ChildrenAvgMetrics(ch_score, ch_conc);
        //if on average children are "better" than the current element (closer to one, but much smaller than one), we continue recursion
        //additional condition being that the active elemenent was not encountered yet: this check is done in order to not coarsen
        //elements excessively while there are still particles inside
        if((ch_score >= 0.95 && ch_score < score)
                || (ch_conc <= 0.5 && ch_conc > concentration)
                || !active_flag)
        {
            SetPassive();
            for(Element* ch : children)
            {
                ch->SetActiveElements(active_flag);
            }
        }
        //if the children on average are "worse" than the current element
        else
        {
                SetActive();
                return;
        }
    }
}

void Element::SetPassiveElements()
{
    if(HasChildren())
    {
        for(Element* ch : children)
        {
            //setting child to be passive
            ch->SetPassive();
            //continuing the recursion
            ch->SetPassiveElements();
        }
    }
}

void Element::ChildrenAvgMetrics(double &sc, double &conc)
{
    double sc_acc = 0.0;
    double conc_acc = 0.0;
    //int n_ch = children.size();
    int nnz = 0;
    for(Element* ch : children)
    {
        if(ch->GetScore() != 0.0)
        {
            nnz++;
            sc_acc += ch->GetScore();
            conc_acc += ch->GetConcentration();
        }
    }
    assert(nnz != 0);
    sc = sc_acc/nnz;
    conc = conc_acc/nnz;
}

void Element::CalculateMetrics(const ParticlePtrList& p_list)
{
    int np = p_list.size();
    //if there are no particles, both metrics are zero
    if(np == 0)
    {
        score = 0.0;
        concentration = 0.0;
    }
    else
    {
        //computing total volume for all particles in current element
        double acc = 0.0;
        for(Particle* p : p_list)
            acc += p->data.V;
        //updating score and concentration
        score = np/double(IDEAL_NP);
        concentration = acc/area;
    }
}


////////////////    GRAPHICS RELATED   ////////////////////////////////
void Element::RemoveImage()
{
    //checking if the element has image and passive
    //if this is the case, we removing the element's image from the scene
    if(HasImage() && !IsActive())
    {
        root->GetScene()->removeItem(dynamic_cast<QGraphicsItem*>(image));
        flags.hasImage = false;
        delete image;
    }
}

///////////////////////////////////////////////////////////////////////////////
// MESH CLASS
///////////////////////////////////////////////////////////////////////////////
Mesh::~Mesh()
{
	ClearNodes();
	//the rest will be taken care of by Element class destructor
}

Mesh::Mesh()
{
	upToDate = false;
    flags.hasChildren = false;
	maxID = 0;
	elements.clear();
	freeIDs.clear();
}

void Mesh::SetDomain(const BoundingBox& b)
{
    root = this;
	bBox = b;
	id = GetNextID();
	//setting up vertices
	Vec2D bl = bBox.GetFrom();
	Vec2D tr = bBox.GetTo();
	Vec2D br = {tr[x1], bl[x2]};
	Vec2D tl = {bl[x1], tr[x2]};
	vertices[_BL] = bl;
	vertices[_BR] = br;
	vertices[_TR] = tr;
	vertices[_TL] = tl;
	//setting up basis functions
	Vec2D d = tr - bl;
	basis[_BL] = new Basis(bl, d[x1], d[x2]);
	basis[_BR] = new Basis(br, -d[x1], d[x2]);
	basis[_TR] = new Basis(tr, -d[x1], -d[x2]);
	basis[_TL] = new Basis(tl, d[x1], -d[x2]);
}

void Mesh::RefineUniformly(const unsigned int n)
{
	Refine(n);
}

void Mesh::RefineByID(const unsigned int id)
{
    for(Element* e : elements)
    {
        if(e->ID() == id)
        {
            e->Refine(1);
            break;
        }
    }
}

unsigned int Mesh::GetNextID()
{
	unsigned int i;
	if(freeIDs.size() != 0)
	{
		i = freeIDs.back();
		freeIDs.pop_back();
	}
	else
	{
		i = maxID;
		maxID++;
	}
	return i;
}

void Mesh::Update()
{
    UpdateActiveElements();
    UpdateNodes();
}

void Mesh::UpdateActiveElements()
{
    elements.clear();
    FillActiveElementList(elements);
}

void Mesh::Adapt(const ParticlePtrList &p_list)
{
    //computing score
    UpdateMetrics(p_list);
    //marking active elements in the tree based on score
    SetActiveElements();
    //updating element and node lists
    UpdateActiveElements();
    UpdateNodes();
}

void Mesh::UpdateNodes()
{
	UpdateNodeList();//clear and collect the list of unique (by position) nodes
	UpdateNodeNeighbors();//fill the neighbor information for each unique node (elements to which the node belongs)
	UpdateElementNodes();//update node pointers for each element
	UpdateHangingNodes();//find hanging nodes and update their interpolation information
}

void Mesh::UpdateNodeList()
{
    //in general should be rewritten in a smarter way, without erasing and rewriting the whole list
	
	//resetting the list
	ClearNodes();
	//going over all the elements in the element list and gathering node coords
	CoordList coords;
	for(Element* e : elements)
		e->AppendNodeCoords(coords);

    //removing duplicated nodes
    coords.sort(Vec2DCompare);
	coords.unique();
	
	//creating node list from node coordinates
	int id = 0;
	for(Vec2D c : coords)
	{
		Node* n = new Node(c, id);
        mesh_nodes.push_back(n);
		id++;
	}
}

void Mesh::UpdateNodeNeighbors()
{
	//checking if nodes list is filled with data
    assert(mesh_nodes.size() != 0);
    for(Node* n : mesh_nodes)
	{
		n->neighbors.clear();
        GetActiveElementsContaining(n->neighbors, n->data->pos);
        //GetElementsContaining(n->neighbors, n->data->pos);
		n->neighbors.sort();
		n->neighbors.unique();
	}
}

void Mesh::UpdateElementNodes()
{
	//going over nodes
    for(Node* n : mesh_nodes)
	{
		//for each node going over neighbor elements
		for(Element* e : n->neighbors)
		{
			//inside element comparing vertices to node coordinates
			int ind[] = {_BL,_BR,_TR,_TL};
			//array<int,4> ind = {_BL,_BR,_TR,_TL};
			for(int i : ind)
			{
                if(e->GetVertices()[i] == n->data->pos)
				{
                    e->SetNodes()[i] = n;
					break;
				}
			}
		}
	}
}

void Mesh::UpdateHangingNodes()
{	
    for(Node* n : mesh_nodes)
	{
		n->isRegular = true;//assuming that node is regular
		for(Element* e : n->neighbors)
		{
			//if node belongs to the element but, not one of it's vertices ->
			//it should be on the side of the element and be a hanging node 
            if(!(e->IsVertex(n->data->pos)))
			{
				n->isRegular = false;
				e->FillInterpolationNodes(n);
                //initializing pointer to the element for which the node is on the side
                n->interp_el = e;
				break;
			}
			
		}
	}
}

void Mesh::ClearNodes()
{
    if(mesh_nodes.size() != 0)
	{
        for(Node* n : mesh_nodes)
        {
            if(n->hasImage) GetScene()->removeItem( dynamic_cast<QGraphicsItem*>(n->GetImage()) );
			delete n;
        }
        mesh_nodes.clear();
	}
}
