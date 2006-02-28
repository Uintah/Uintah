/* TensorFieldBase.h
   -----------------

   This is the base class for the templatized TensorField class.  We make this
   base class because the classes we have to deal with for passing data through
   tend to not play well with templates.  As such this is pretty much a dummy 
   class.

   Pretty much all we keep track of is what type we are (i.e., what our derived
   templatized class is) so that users can figure that out and act accordingly

   Eric Lundberg,  10/8/1998
   
   */
#ifndef SCI_Packages_DaveW_Datatypes_TensorFieldBase_h
#define SCI_Packages_DaveW_Datatypes_TensorFieldBase_h 1

#include <Core/Containers/Array1.h>
#include <Core/Containers/Array2.h>
#include <Core/Containers/Array3.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Datatypes/Datatype.h>
#include <Core/Datatypes/ScalarFieldRG.h>
#include <Core/Datatypes/VectorFieldRG.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <stdio.h>

namespace DaveW {
using namespace SCIRun;


#define TENSOR_ELEMENTS 6 /*Number of elements in the 3x3 tensor we car about*/
#define EVECTOR_ELEMENTS 3 /*Number of eigen vectors produced by the tensor matrix*/
/* The order in which vectors and elements area stored in the array of EVECTOR_ELEMENTS */ 
#define PRIMARY 0 
#define SECONDARY 1
#define TERIARY 2 
enum {CHAR, UCHAR, SHORT, USHORT, INT, UINT, LONG, ULONG,  FLOAT, DOUBLE};

class TensorFieldBase;
typedef LockingHandle<TensorFieldBase> TensorFieldHandle;

class TensorFieldBase : public Datatype
{
protected:  
  int m_type;
  Point bmin, bmax;
  Vector diagonal;
public:
  int m_slices;
  int m_width;
  int m_height;
  Array1< VectorFieldRG > m_e_vectors;
  Array1< ScalarFieldRGdouble > m_e_values;
  Array3< float > m_inside;
  int m_tensorsGood;
  int m_vectorsGood;
  int m_valuesGood;

  TensorFieldBase();
  TensorFieldBase(const TensorFieldBase&); /*Deep Copy Constructor*/

  virtual ~TensorFieldBase();

  virtual TensorFieldBase* clone() const=0;

  /* Type handling */
  void set_type(int in_type);
  int get_type(void);

  void get_bounds(Point &min, Point &max);
  void set_bounds(const Point& min, const Point& max);

  virtual int interpolate(const Point&, double[][3], int&, int=0)=0;
  virtual int interpolate(const Point&, double[][3])=0;

  /* Persistent representation...*/
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
};
} // End namespace DaveW



#endif /* SCI_Packages/DaveW_Datatypes_TensorFieldBase_h */
