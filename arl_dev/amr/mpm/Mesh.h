#pragma once

#include<QGraphicsRectItem>
#include<QGraphicsScene>

#include"Typedefs.h"
#include"Utils.h"
#include"BoundingBox.h"
#include"Basis.h"
#include"Node.h"

class Mesh;

struct ElementFlags
{
    ElementFlags()
    {
        Reset();
    }
    void Reset()
    {
        isActive = false;
        hasImage = false;
        hasChildren = false;
    }

    //whether element is active or passive:
    //active element is included into the list of elements that comprise the
    //current mesh
    //the distinction between the isActive and hasChildren is made in order to
    //be able to keep the finer elements in the tree during coarsening operation
    //in case the mesh will be refined later on (saves a destructor/constructor call)
    bool isActive;
    //whether or not element has children
    bool hasChildren;
    //whether or not element has a graphical representation
    bool hasImage;
};

class Element
{
public:
    Element();
    Element(const int l, const BoundingBox& b, Element* p);
    ~Element();
    void Refine(const unsigned int n);
    void FillElementList(ElementPtrList& e, const bool check_root = true);
    void FillActiveElementList(ElementPtrList& e, const bool check_root = true);
    void FillInterpolationNodes(Node* n);
    //get a list of elements containing given point (element contains point if
    //it is on the boundary)
    void GetElementsContaining(ElementPtrList& e, const Vec2D& v, const bool check_root = true);
    void GetActiveElementsContaining(ElementPtrList& e, const Vec2D& v, const bool check_root = true);
    //check if the given point is an element vertex
    //potentially can be rewritten to return more info: on the dege / vertex / inside
    bool IsVertex(const Vec2D& v);
    void AppendNodeCoords(CoordList& lst);

    //setters/getters
    //setting action flag
    void SetAction(MeshAction a){action = a;}
    //void SetRoot(const Element* r){root = r;}
    const Mesh* Root(){return root;}
    //void SetParent(const Element* p){parent = p;}
    const Element* Parent(){return parent;}
    const BoundingBox& BBox(){return bBox;}
    //children functionality
    bool HasChildren(){return flags.hasChildren;}
    void SetHasChildren(){flags.hasChildren = true;}
    void SetNoChildren(){flags.hasChildren = false;}
    const ElementPtrVector& GetChildren(){return children;}
    //id and level information
    unsigned int ID() const {return id;};
    unsigned int Level() const {return level;};
    //nodes, vertices and basis
    const array<Node*,4>& GetNodes() const {return nodes;}
    array<Node*,4>& SetNodes(){return nodes;}
    const array<Vec2D,4>& GetVertices(){return vertices;}
    const LocalBasis& GetBasis(){return basis;}
    //isActive logic
    bool IsActive(){return flags.isActive;}
    //setting current element active and all the children passive
    void SetActive(){flags.isActive = true; SetPassiveElements();}
    void SetPassive(){flags.isActive = false; RemoveImage();}
    //score related
    int GetScore(){return score;}
    //graphics related
    QGraphicsRectItem* GetImage() {return image;}
    void SetImage(QGraphicsRectItem* i){image = i; flags.hasImage = true;}
    bool HasImage(){return flags.hasImage;}

protected:
    //mesh adaptation implementation
    //calculates score for an element
    //takes in a list of particle coordinates
    void CalculateScore(const ParticlePtrList &p_list);
    void SetActiveElements();
    //sets all the elements below current element to be passive
    void SetPassiveElements();
    int ChildrenAvgScore();

    //removes element image from the scene
    void RemoveImage();

    unsigned int level;//level of refinement of the element
    unsigned int id;//element id
    LocalBasis basis;//4 partial basis functions at the corresponding corners of the element
    BoundingBox bBox;//bounding box
    array<Vec2D,4> vertices;//vertices of the element
    array<Node*,4> nodes;//nodes of the element

    //logical flags
    ElementFlags flags;

    ElementPtrVector children;//list of pointers to the children elements
    //ParticlePtrList particles;//list of pointers to the particle data
    Element *parent;
    static Mesh *root;
    //mesh action to be performed on the element: coarsen, keep, refine
    MeshAction action;
    //score that determines whether element is a "good" one or not
    int score;

    //graphics related
    QGraphicsRectItem *image;
};

//these comparison operators are used for ElementPtrList sort and 
//removal of the duplicate elements

bool operator<(const Element& a, const Element& b);
bool operator==(const Element& a, const Element& b);
ostream& operator<<(ostream& os, const Element& e);


class Mesh : public Element
{
public:
    ~Mesh();
    //SetDomain and RefineUniformly is used to initialize mesh after the default constructor was used
    Mesh();
    void SetDomain(const BoundingBox& b);
    void RefineUniformly(const unsigned int n);
    void RefineByID(const unsigned int i);
    //calls corresponding update functions for elements and node
    //however, since node update is more involved, we allow
    //to call separate updates for elements and nodes
    void Update();
    //collects all the active elements in the elements list
    void UpdateElements();
    //collects all the unique nodes
    void UpdateNodes();
    unsigned int GetNextID();
    //fill the interpFrom array for a given node
    void FillInterpolationNodes(Node* n);
    //mesh adaptation: takes particle coordinates and determines
    //optimal mesh structure
    void Adapt(const ParticlePtrList& p_list);

    //data access functions
    //returns the list of the leaf nodes
    const ElementPtrList& Elements(){return elements;}
    const NodePtrList& Nodes(){return mesh_nodes;}
    inline void SetScene(QGraphicsScene *s){scene = s;}
    inline QGraphicsScene* GetScene(){return scene;}
private:
    void UpdateActiveElements();
    void UpdateNodeList();
    void UpdateNodeNeighbors();
    void UpdateElementNodes();
    void UpdateHangingNodes();
    void ClearNodes();

    //vector of leaf nodes of the mesh tree: represents the current state of the mesh
    ElementPtrList elements;
    //list of nodes
    NodePtrList mesh_nodes;

    //contains the max element ID
    unsigned int maxID;
    //list containing "free" element ids (in case the mesh was coarsened and some elements were deleted)
    list<unsigned int> freeIDs;
    //flag that indicates whether the elements object (list of leaf nodes) is up to date
    bool upToDate;

    //graphics related
    QGraphicsScene *scene;
};
