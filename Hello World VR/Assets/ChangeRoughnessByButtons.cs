using System.Collections;
using System.Collections.Generic;
using UnityEngine;



public class ChangeRoughnessByButtons : MonoBehaviour
{
	
	public Material BRDF1Material;
	public Material BRDF2Material;
	public Material BRDF3Material;
	
	
	float smoothness = 0.9f;
    // Start is called before the first frame update
    void Start()
    {
        
    }
	

	public void SetRoughness(float alpha){
		float smoothness = 1-alpha;
		BRDF1Material.SetFloat("_GlossMapScale",smoothness);
		BRDF2Material.SetFloat("_GlossMapScale",smoothness);
		BRDF3Material.SetFloat("_GlossMapScale",smoothness);
		
	}
	
	public void SetSpecularScale(float s){
		Color speccolor = s*Color.white;//new Color(d,d,d,1.0f);
		BRDF1Material.SetColor("_SpecColor",speccolor);
		BRDF2Material.SetColor("_SpecColor",speccolor);
		BRDF3Material.SetColor("_SpecColor",speccolor);
		
	}
	
	public void SetDiffuseScale(float d){
		Color difcolor = d*Color.white;//new Color(d,d,d,1.0f);
		BRDF1Material.SetColor("_Color",difcolor);
		BRDF2Material.SetColor("_Color",difcolor);
		BRDF3Material.SetColor("_Color",difcolor);
		
	}
	
	public void SetGeometry(int i){
		if(false);
		
		
		
	}
    // Update is called once per frame
    void Update()
    {
		
    }
}
