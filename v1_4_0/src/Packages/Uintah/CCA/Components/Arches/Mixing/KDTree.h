//----- KDTree.h --------------------------------------------------

#ifndef Uintah_Component_Arches_KDTree_h
#define Uintah_Component_Arches_KDTree_h

/***************************************************************************
CLASS
    KDTree
         KDTree class provides the data structure to store table information
       
GENERAL INFORMATION
    KDTree.h - Declaration of KDTree class

    Author: Rajesh Rawat (rawat@crsim.utah.edu)
    
    Creation Date : 05-30-2000

    C-SAFE
    
    Copyright U of U 2000

KEYWORDS
    
DESCRIPTION
      KDTree stores a table with k-parameters. It provides order log(N) for search
   and insert for a well-balanced tree. It is used to store sub-grid scale
   mixing model data where parameters can range anywhere from 2-20. Currently, its
   designed to work with uniform distribution in the parameters which is known apriori.


PATTERNS
    None

WARNINGS
    None

POSSIBLE REVISIONS
     1. Non-Uniform table spacing
***************************************************************************/

//#include <Packages/Uintah/CCA/Components/Arches/Mixing/PDFShape.h>

#include <vector>
using namespace std;
namespace Uintah {

  ///////////////////////////////////////////////////////////////////////////
  // structure KD_Node is the basic element of the KD_Tree .
  // Each cell contain the pair keys[] & Phi[] , and three pointers
  // left, right & parent.  


  
  struct KD_Node { 
    
    KD_Node *left;
    KD_Node *right;
    int dim;
    int Phi_dim;
    int* keys;
    vector<double> Phi;
  
    //Constructor Node transfer the value keyes & Data .

    KD_Node(){
        keys = 0;
        Phi = vector<double>();
        left =0;
        right = 0;
    }

    KD_Node(int dim, int Phi_dim, int Key[],vector<double> Phi, KD_Node *left=0, 
            KD_Node *right=0):   
      dim(dim), Phi_dim(Phi_dim),Phi(Phi),left(left),right(right){
        keys = new int[dim];
        for(int i=0;i< dim; i++){
          keys[i] = Key[i];
        }
    }   

  };

class PDFMixingModel; //??Should I change this to MixingModel to make it general??
class MixRxnTableInfo;
class KD_Tree {

  public:

    // GROUP: Constructors:
    ////////////////////////////////////////////////////////////////////////
    //
    // Constructs an instance of KD_Tree with the given number of 
    // parameters and number of statespace vars
    //
    // Constructor taking
    //   [in] d number of parameters
    //   [in] P_dim number of state space variables

  KD_Tree(int d, int P_dim);

  // GROUP: Destructor:
  ///////////////////////////////////////////////////////////////////////
  //
  // Destructor 
  //
  ~KD_Tree();

  // GROUP: Manipulate
  //////////////////////////////////////////////////////////////////////
  // Store a key and value pair into the KD_Node and inserts in   
  // the Tree.  

  bool Insert(int key[], vector<double> Phi);

  // GROUP: Access
  //////////////////////////////////////////////////////////////////////
  // Lookup function looks up the tree for key. If its found it stores the
  // statespcae vars in Phi vector and returns true, else it just returns
  // false.
  bool Lookup(int key[], vector<double>& Phi);

  // GROUP: Manipulate
  //////////////////////////////////////////////////////////////////////
  // Remove a KD_Node with the given key from the Tree.
  bool Delete(int key[]);


  // GROUP: Access
  //////////////////////////////////////////////////////////////////////
  // Returns the number of Nodes contained in the Tree. 
  //
  int getSize() const;

  // GROUP: Access    
  /////////////////////////////////////////////////////////////////////
  // Report if the Tree is empty.
  //

  bool IsEmpty () const;


 private:
   //** Insert a new Node z which contain a given key into the Tree.

   KD_Node* TreeInsert(KD_Node*& x, int key[], vector<double> phi, int lev);

  
   //** Given a key to search which contain this key in Tree
   KD_Node* TreeSearch(KD_Node *x, int key[]);

   //** Find the Node to be delete

   bool DeleteItem(KD_Node*& x, int key[], int lev);
   
   //** Delete the Node which contains the given key .

   void TreeDelete(KD_Node*& x, int key[]);


   //** Retrieves and deletes the leftmost descendant of a given node.
       
   void ProcessLeftMost(KD_Node*& x, vector<double>& phi);
   
   //** Destroy a tree;
   
   void DestroyTree(KD_Node *x);
  
   
   KD_Node *d_root;         // The root of KD_Tree.
   int d_size;              // The total size of KD_Tree.
   int d_dim;               // The dimension of the StateSpace(independent variables).
   int d_dimStateSpaceVars;

  }; // End Class KDTree
} // end namespace Uintah

#endif

