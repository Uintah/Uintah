
/*----------------------------------------------------------------------
CLASS
    AnalytAttrib

    Field attribute containing analytic expressions

GENERAL INFORMATION
    Created by:
    Alexei Samsonov
    Department of Computer Science
    University of Utah
    August 2000
    
    Copyright (C) 2000 SCI Group

KEYWORDS
   

DESCRIPTION
   
PATTERNS
   
WARNING
   
POSSIBLE REVISIONS
    Adding some functions that could turn out to be useful, for instance
    set_function(const GenFunctionHandle&)
----------------------------------------------------------------------*/

#include <SCICore/Datatypes/Attrib.h>
#include <SCICore/Datatypes/DiscreteAttrib.h>
#include <SCICore/Datatypes/GenFunction.h>
#include <SCICore/Datatypes/Geom.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/Point.h>

#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Util/NotFinished.h>
#include <SCICore/Exceptions/DimensionMismatch.h>
#include <SCICore/Exceptions/InternalError.h>
#include <string>
#include <vector>
#include <sstream>

namespace SCICore{
namespace Datatypes{

  using std::string;
  using SCICore::PersistentSpace::Piostream;
  using SCICore::PersistentSpace::PersistentTypeID;
  using SCICore::Geometry::Vector;
  using SCICore::Geometry::Point;
  using namespace std;
  using namespace SCICore::Exceptions;

  typedef vector<string>      StrV;
  typedef vector<double>      RealV;

  template <class T> class SCICORESHARE AnalytAttrib: public Attrib {
  
  public:  

    //////////
    // Constructors
    AnalytAttrib();
    explicit AnalytAttrib(const GenFunctionHandle&);
    AnalytAttrib(const AnalytAttrib&);

    AnalytAttrib& operator=(const AnalytAttrib&);

    //////////
    // Evaluate attribute at some point
    inline T eval(const Point&) const;
    inline T eval(double, double, double) const;
   
    //////////
    // Creates new sampled attrib corresponding to the geometry
    // and returns handle to it
    AttribHandle get_sampled(const Geom*);

    //////////
    // Returns handle to the attribute function
    GenFunctionHandle get_function() const;
    
    //////////
    // Returns vector of string representations 
    // of the attribute functions
    const StrV& get_fstr() const;
    
    //////////
    // Base class override
    string getInfo();

    //////////
    // Initializes pevalf pointer to whether precompiled
    // function or to Function::eval(double*) member
    // Default mode is slow eval
    void set_fasteval();
    void set_sloweval();
    
    //////////
    // Modifies the attribute by copying supplied function
    // Checks if domain and result dimensions are the same
    // Throw DimensionMismatch if different dimensions 
    void set_function(const StrV&);

    //////////
    // Persistent representation...
    // No function object is carried around while serializing
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
  
  private:
    GenFunctionHandle func;
    mutable StrV  fstrings;
    //////////
    // Should be specialized to return the number of components of the attribute
    // for every type T
    inline int ncomps();
  };

  //////////
  // Specializations for setting the attribute
  // Assuming T is built-in type
  template <class T> inline int AnalytAttrib<T>::ncomps(){
    return 1;
  }

  template <> inline int AnalytAttrib<Vector>::ncomps(){
    return 3;
  }

  //////////
  //Tensor spesialization - to be implemented
  //template <class T> inline int AnalytAttrib<T>::ncomps(){
  //}

  //////////
  // Constructors/Destructor
  
  template <class T> AnalytAttrib<T>::AnalytAttrib(){
    func=scinew GenFunction();
  }

  template <class T> void AnalytAttrib<T>::set_function(const StrV& strs){
    
    try {
      // checking if number of supplied strings matches the dimensionality
      // of the current template
      if (ncomps()!=strs.size())
	throw DimensionMismatch((long)strs.size(), ncomps());
    }
    catch (DimensionMismatch exc){
      cout << "Exception in AnalytAttrib(const vector<string>&) of type " 
	   << exc.type() << endl << "Message: " << exc.message() << endl;
      throw;
    }

    try {
      func->set_funcs(strs, 3);
    }
    catch (DimensionMismatch exc){
      cout << "Exception in AnalytAttrib(const vector<string>&) of type "
           << exc.type() << endl << "Message: " << exc.message() << endl;
    }
    catch (InternalError exc){
      cout << "Exception in AnalytAttrib(const vector<string>&) of type "
	   << exc.type() << endl << "Message: " << exc.message() << endl;
    }
  }

  /*  
  template <class T> AnalytAttrib<T>::AnalytAttrib(const StrV& strs){
    try {
      // checking if number of supplied strings matches the dimensionality
      // of the current template
      if (ncomps()!=strs.size())
	throw DimensionMismatch((long)strs.size(), ncomps());
    }
    catch (DimensionMismatch exc){
      cout << "Exception in AnalytAttrib(const vector<string>&) of type " 
	   << exc.type() << endl << "Message: " << exc.message() << endl;
      throw;
    }

    try {
      func=scinew GenFunction(strs, 3);
    }
    catch (DimensionMismatch exc){
      cout << "Exception in AnalytAttrib(const vector<string>&) of type "
           << exc.type() << endl << "Message: " << exc.message() << endl;
    }
    catch (InternalError exc){
      cout << "Exception in AnalytAttrib(const vector<string>&) of type "
	   << exc.type() << endl << "Message: " << exc.message() << endl;
    }
  }

  template <class T> AnalytAttrib<T>::AnalytAttrib(const GenFunctionHandle& fh){
    try {
      int nc=ncomps(), newnc=fh->get_num_comp();
      int newdd=fh->get_dom_dims();
      if (nc!=newnc)
	throw DimensionMismatch(newnc, nc);
      if (3!=newdd)
	throw DimensionMismatch(newdd, 3);
    }
    catch (DimensionMismatch exc){
      cout << "Exception in AnalytAttrib(const GenFunctionHandle&) of type "
           << exc.type() << endl << "Message: " << exc.message() << endl;
      throw;
    }
    func=fh;
  }
  */

  template <class T> AnalytAttrib<T>::AnalytAttrib(const AnalytAttrib<T>& copy){
    func=copy.func;
  }

  template <class T> AnalytAttrib<T>& AnalytAttrib<T>::operator=(const AnalytAttrib<T>& copy){
    func=copy.func;
    return *this;
  }

  template <class T> AttribHandle AnalytAttrib<T>::get_sampled(const Geom*){
    NOT_FINISHED("AttribHandle AnalytAttrib<T>::get_sampled()");
    return AttribHandle();
    //return AttribHandle(scinew DiscreteAttrib<T>);
  }


  template <class T> void AnalytAttrib<T>::set_fasteval(){
    func->set_speed(true);
  }

  template <class T> void AnalytAttrib<T>::set_sloweval(){
    func->set_speed(false);
  }

  template <class T> GenFunctionHandle AnalytAttrib<T>::get_function() const{
    return func;
  }

  template <class T> string AnalytAttrib<T>::getInfo()
  {
    ostringstream ostr;
    ostr << 
      "Name = " << d_name << '\n' <<
      "Type = AnalytAttrib" << '\n' <<
      "Defined in Domain R" << 3 <<'\n' <<
      "Output dimensions = " << func->get_num_comp() <<'\n'; 
    return ostr.str();  
  }

  template <class T> const StrV& AnalytAttrib<T>::get_fstr() const{
    int n=func->get_num_comp();
    fstrings.resize(n);
    for ( int i=0; i<n; i++)
      fstrings[i]=func->get_str_f(i);
    return fstrings;
  }

  //////////
  // Implementation of eval functions
  // Specializations should be provided for each new return type
  template <class T> inline T AnalytAttrib<T>::eval(const Point& p) const{   
    static double vrs[3];
    vrs[0]=p.x();
    vrs[1]=p.y();
    vrs[2]=p.z();
    return func->get_value(vrs, 3, 0);
  }

  template <class T> inline T AnalytAttrib<T>::eval(double x, double y,  double z) const{
    static double vrs[3];
    vrs[0]=x;
    vrs[1]=y;
    vrs[2]=z;
    return func->get_value(vrs, 3, 0);
  }
  
  template <> inline Vector AnalytAttrib<Vector>::eval(const Point& p) const{ 
    static double vrs[3];
    vrs[0]=p.x();
    vrs[1]=p.y();
    vrs[2]=p.z();
    return Vector(func->get_value(vrs, 3, 0), func->get_value(vrs, 3, 1), func->get_value(vrs, 3, 2));
  }

  template <> inline Vector AnalytAttrib<Vector>::eval(double x, double y,  double z) const{
    static double vrs[3];
    vrs[0]=x;
    vrs[1]=y;
    vrs[2]=z;
    return Vector(func->get_value(vrs, 3, 0), func->get_value(vrs, 3, 1), func->get_value(vrs, 3, 2));
  }


  template <class T>
  void
  AnalytAttrib<T>::io(Piostream &)
  {
  }


} // end Datatypes
} // end SCICore








