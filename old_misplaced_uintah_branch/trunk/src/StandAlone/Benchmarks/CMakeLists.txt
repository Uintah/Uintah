ADD_EXECUTABLE(SimpleMath SimpleMath.cc)

TARGET_LINK_LIBRARIES(SimpleMath
        Packages_Uintah_Core_Exceptions    
        Packages_Uintah_Core_Grid          
        Packages_Uintah_Core_Util          
        Packages_Uintah_Core_Math          
        Packages_Uintah_Core_ProblemSpec   
        Packages_Uintah_Core_Parallel      
        Packages_Uintah_Core_Disclosure    
        Core_Exceptions  
        Core_Geometry    
        Core_Persistent  
        Core_Util        
        Core_Thread      
        Core_Containers
)
