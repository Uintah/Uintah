//
// FileReader.h,v 1.4 2003/07/21 23:51:40 dhinton Exp
//

#ifndef FILEREADER_H
#define FILEREADER_H
#include <pre.h>
#include <SCIRun/Tao/Component.h>
#include <CCA/Components/TAO/FileReader/TestS.h>

#if defined (_MSC_VER)
# if (_MSC_VER >= 1200)
#  pragma warning(push)
# endif /* _MSC_VER >= 1200 */
# pragma warning (disable:4250)
#endif /* _MSC_VER */

/// Implement the Test::FileReader interface
class FileReader
  : public virtual POA_Test::FileReader
  , public virtual PortableServer::RefCountServantBase
  , public SCIRun::tao::Component
{
public:
  /// Constructor
  FileReader (CORBA::ORB_ptr orb);

  //TaoComponent has to inherit this in SR2
  void setServices(sci::cca::TaoServices::pointer services);

  // = The skeleton methods


    virtual CORBA::Long getPDEdescription (
        Test::FileReader::double_array_out nodes ,
        Test::FileReader::long_array_out boundaries,
        Test::FileReader::long_array_out dirichletNodes,
        Test::FileReader::double_array_out dirichletValues
        ACE_ENV_ARG_DECL_WITH_DEFAULTS
      )
      ACE_THROW_SPEC ((
        CORBA::SystemException
      ));

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
