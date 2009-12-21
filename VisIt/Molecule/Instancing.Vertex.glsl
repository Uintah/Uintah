#version 120
#extension EXT_gpu_shader4 : enable


uniform samplerBuffer per_instance_data_position_radius;
uniform samplerBuffer per_instance_data_attribute;


varying vec3 eye_position;
varying vec3 eye_normal;
varying vec3 data_attribute;

void main(void)
{
	vec4 instance_position_radius = texelFetchBuffer (per_instance_data_position_radius, gl_InstanceID);
	vec3 instance_attribute =  texelFetchBuffer (per_instance_data_attribute, gl_InstanceID).xyz;
        
	vec3 center = instance_position_radius.xyz;
	float r = instance_position_radius.w;
	// float a = instance_attribute;

	// data_attribute = a;
	data_attribute = instance_attribute;

	float theta = gl_Vertex.x;
	float phi = gl_Vertex.y;

	
	float cos_theta = cos(theta);
	float sin_theta = sin(theta);

	float cos_phi = cos(phi);
	float sin_phi = sin(phi);

	float x = sin_phi * cos_theta ;
	float y = sin_phi * sin_theta  ;
	float z = cos_phi;

	vec3 object_position = vec3(x,y,z);
	vec3 object_normal = vec3(x,y,z);

	//object_position = vec3(gl_Vertex.x, gl_Vertex.y, 0);
	//object_normal  = vec3(0,0,1);
	//data_attribute = gl_VertexID /  12.0f;

	vec4 position = vec4(center + r * object_position, 1);
	
	eye_normal = gl_NormalMatrix * object_normal;
	eye_position = (gl_ModelViewMatrix * position).xyz;
	

	gl_Position     = gl_ModelViewProjectionMatrix * position;
	
}
