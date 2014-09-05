
#include <Packages/Ptolemy/Core/jni/JNIUtil.h>

namespace Ptolemy {

class VergilThread : public SCIRun::Runnable {
public:
    VergilThread(const std::string& cp, const std::string& mp) :
        configPath(cp), modelPath(mp) {}
    virtual ~VergilThread() {}
    virtual void run();
private:
    const std::string configPath;
    const std::string modelPath;
};


class VergilWrapper {
public:
  // get Vergil class

protected:

private:
};

}
