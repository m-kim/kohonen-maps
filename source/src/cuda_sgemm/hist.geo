#version 120
#extension GL_EXT_geometry_shader4 : enable
void drawTop(vec4 position)
{
    vec4 v1, v2, v3, tmp;
    vec3 v_n1, v_n2;
		vec3 normal;
		v1 = position + vec4(-.5,-.5,0,0);
		v2 = position + vec4(-.5,.5,0,0);
		v3 = position + vec4(.5,-.5,0,0);
		v_n1 = v3.xyz - v2.xyz;
		v_n2 = v1.xyz - v2.xyz;
		normal = normalize(cross(v_n2, v_n1));
		gl_Position = v1;
		gl_Position = gl_ModelViewProjectionMatrix * gl_Position;
    EmitVertex();
		gl_Position = v2;
		gl_Position = gl_ModelViewProjectionMatrix * gl_Position;
    EmitVertex();
		gl_Position = v3;
		gl_Position = gl_ModelViewProjectionMatrix * gl_Position;
    EmitVertex();
		v1 = v2;
		v2 = v3;
		v3 = position + vec4(.5,.5,0,0);
		v_n1 = v3.xyz - v2.xyz;
		v_n2 = v1.xyz - v2.xyz;
		normal = normalize(cross(v_n1, v_n2));
		gl_Position = v3; 
		gl_Position = gl_ModelViewProjectionMatrix * gl_Position;
		EmitVertex();

		EndPrimitive();
}

void setLight(vec4 v2, vec3 normal)
{
vec3 eye_normal, lightDir;
		vec4 diffuse, ambient, globalAmbient;
		float NdotL;
		
		eye_normal = normalize(gl_NormalMatrix * normal);
		lightDir = normalize(vec3(gl_LightSource[0].position));
		NdotL = max(dot(eye_normal, lightDir), 0.0);
		diffuse = gl_FrontMaterial.diffuse * gl_LightSource[0].diffuse;
		
		/* Compute the ambient and globalAmbient terms */
		ambient = gl_FrontMaterial.ambient * gl_LightSource[0].ambient;
		globalAmbient = gl_LightModel.ambient * gl_FrontMaterial.ambient;
		
		gl_FrontColor =  NdotL * diffuse + globalAmbient + ambient;
}

void main(void)
{
  vec4 v1, v2, v3, tmp;
  vec3 v_n1, v_n2;
	int i;
	for(i=0; i< gl_VerticesIn; i++){
		//drawTop(gl_PositionIn[i]);

		gl_FrontColor = gl_FrontColorIn[i];
		v1 = gl_PositionIn[i] + vec4(-.5,-.5,0,0);
		v2 = gl_PositionIn[i] + vec4(-.5,-.5,-gl_PositionIn[i].z,0);
		v3 = gl_PositionIn[i] + vec4(.5,-.5,0,0);
		v_n1 = v3.xyz - v2.xyz;
		v_n2 = v1.xyz - v2.xyz;
		
		setLight(v2,cross(v_n1, v_n2));
    gl_Position = v1;
		gl_Position = gl_ModelViewProjectionMatrix * gl_Position;
		EmitVertex();
		gl_Position = v2;
		gl_Position = gl_ModelViewProjectionMatrix * gl_Position;
		EmitVertex();
		gl_Position = v3;
		gl_Position = gl_ModelViewProjectionMatrix * gl_Position;
		EmitVertex();

		
    v1 = v2;
    v2 = v3;
		v3 = gl_PositionIn[i] + vec4(.5,-.5,-gl_PositionIn[i].z,0);
    v_n1 = v3.xyz - v2.xyz;
		v_n2 = v1.xyz - v2.xyz;
		
		setLight(v2,normalize(cross(v_n2, v_n1)));
		gl_Position = v3;
		gl_Position = gl_ModelViewProjectionMatrix * gl_Position;
		EmitVertex();
		
		
    v1 = v2;
    v2 = v3;
    v3 = gl_PositionIn[i] + vec4(.5,.5,0,0);
		v_n1 = v3.xyz - v2.xyz;
		v_n2 = v1.xyz - v2.xyz;
		
		setLight(v2,normalize(cross(v_n1, v_n2)));
		gl_Position = v3; 
		gl_Position = gl_ModelViewProjectionMatrix * gl_Position;
    EmitVertex();
    
    
    v1 = v2;
    v2 = v3;
    v3 = gl_PositionIn[i] + vec4(.5,.5,-gl_PositionIn[i].z,0);
		v_n1 = v3.xyz - v2.xyz;
		v_n2 = v1.xyz - v2.xyz;
		
		setLight(v2,normalize(cross(v_n2, v_n1)));
		gl_Position = v3; 
		gl_Position = gl_ModelViewProjectionMatrix * gl_Position;
    EmitVertex();


    v1 = v2;
    v2 = v3;
    v3 = gl_PositionIn[i] + vec4(-.5,.5,-gl_PositionIn[i].z,0);
    v_n1 = v3.xyz - v2.xyz;
		v_n2 = v1.xyz - v2.xyz;
		
		setLight(v2,normalize(cross(v_n1, v_n2)));
		gl_Position = v3;
		gl_Position = gl_ModelViewProjectionMatrix * gl_Position;
		EmitVertex();


    v1 = v2;
    v2 = v3;
    v3 = gl_PositionIn[i] + vec4(-.5,.5,0,0);
    v_n1 = v3.xyz - v2.xyz;
		v_n2 = v1.xyz - v2.xyz;
		
		setLight(v2,normalize(cross(v_n2, v_n1)));
		gl_Position = v3;
		gl_Position = gl_ModelViewProjectionMatrix * gl_Position;
		EmitVertex();

		
		v1 = v2;
		v2 = v3;
    v3 = gl_PositionIn[i] + vec4(-.5,-.5,-gl_PositionIn[i].z,0);
    v_n1 = v3.xyz - v2.xyz;
		v_n2 = v1.xyz - v2.xyz;
		
		setLight(v2,normalize(cross(v_n1, v_n2)));
		gl_Position = v3;
		gl_Position = gl_ModelViewProjectionMatrix * gl_Position;
    EmitVertex();

    
    v1 = v2;
    v2 = v3;
    v3 = gl_PositionIn[i] + vec4(-.5,-.5,0,0);
    v_n1 = v3.xyz - v2.xyz;
		v_n2 = v1.xyz - v2.xyz;
		
		setLight(v2,normalize(cross(v_n2, v_n1)));
		gl_Position = v3;
		gl_Position = gl_ModelViewProjectionMatrix * gl_Position;
		EmitVertex();
		EndPrimitive();
	}
}