#include <iostream>
using namespace std;

class InputDevice {
virtual void DisplayDevice () = 0 ;
};

class IBMKeyboard : public InputDevice {
void DisplayDevice () {
cout << "IBMKeyboard" << endl ;
}
};

class IBMPrinter : public InputDevice {
void DisplayDevice () {
cout << "IBMPrinter" << endl;
}
};


class OutputDevice {
virtual void DisplayDevice () = 0 ;
};

class SamsungMonitor : public OutputDevice {
void DisplayDevice () {
cout << "SamsungMonitor" << endl ;
}
};

class SamsungMouse : public OutputDevice {
void DisplayDevice () {
cout << "SamsungMouse" << endl ;
}
};



class DeviceFactory {
virtual InputDevice* FInputDevice () = 0;
virtual OutputDevice* FOutputDevice () = 0;
};


class IBMDeviceFactory : public DeviceFactory {
void FInputDevice () {
return new IBMKeyboard ;
}
void FOutputDevice () {
return new IBMPrinter ;
}
};

class SamsungDeviceFactory : public DeviceFactory {
void FInputDevice () {
return new SamsungMouse ;
}

void FOutputDevice () {
return new SamsungMonitor ;
}

};


main() {
DeviceFactory *df = new IBMDeviceFactory ;
InputDevice *Id ;
Id = df -> FInputDevice () ;
Id -> DisplayDevice () ;

}

