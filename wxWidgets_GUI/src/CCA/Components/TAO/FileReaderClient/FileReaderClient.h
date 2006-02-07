//
// FileReader.h,v 1.4 2003/07/21 23:51:40 dhinton Exp
//

#ifndef FILEREADER_CLIENT_H
#define FILEREADER_CLIENT_H
#include <pre.h>
#include <SCIRun/Tao/Component.h>
#include <CCA/Components/TAO/FileReader/TestS.h>

#if defined (_MSC_VER)
# if (_MSC_VER >= 1200)
#  pragma warning(push)
# endif /* _MSC_VER >= 1200 */
# pragma warning (disable:4250)
#endif /* _MSC_VER */

class FileReaderClient
  : public SCIRun::tao::Component
{
public:
  /// Constructor
  FileReaderClient(CORBA::ORB_ptr orb);

  //TaoComponent has to inherit this in SR2
  void setServices(sci::cca::TaoServices::pointer services);

  virtual int go();
private:
  /// Use an ORB reference to conver strings to objects and shutdown
  /// the application.
  CORBA::ORB_var orb_;
};

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma warning(pop)
#endif /* _MSC_VER */

#include <post.h>
#endif /* FILEREADER_H */
