/* TensorFieldBase.h
   -----------------

   This is the base class for the templatized TensorField class.  We make this
   base class because the classes we have to deal with for passing data through
   tend to not play well with templates.  As such this is pretty much a dummy class.

   Pretty much all we keep track of is what type we are (i.e., what our derived
   templatized class is) so that users can figure that out and act accordingly

   Eric Lundberg,  10/8/1998
   
   */
#ifndef SCI_Datatypes_TensorFieldBase_h
#define SCI_Datatypes_TensorFieldBase_h 1

#include <stdio.h>
#include <Geometry/Point.h>
#include <Geometry/Vector.h>
#include <Datatypes/Datatype.h>
#include <Classlib/LockingHandle.h>
#include <Classlib/Array3.h>
#include <Classlib/Array2.h>
#include <Classlib/Array1.h>

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
  TensorFieldBase();
  TensorFieldBase(const TensorFieldBase&); /*Deep Copy Constructor*/

  virtual ~TensorFieldBase();

  virtual TensorFieldBase* clone() const=0;

  /* Type handling */
  void set_type(int in_type);
  int get_type(void);
  virtual int interpolate(const Point&, double[][3], int&, int=0)=0;
  virtual int interpolate(const Point&, double[][3])=0;

  /* Persistent representation...*/
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
};

#endif
