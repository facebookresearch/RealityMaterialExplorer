using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ChangeMaterial : MonoBehaviour
{	
	public bool iamroot;
	
	public void changeMaterial(Material m){
		if(iamroot){
			GetComponent<Renderer>().material = m;
		}
		else{
			for(int i = 0; i < transform.childCount; i++){
			ChangeMaterial childscript = (ChangeMaterial) transform.GetChild(i).GetComponent(typeof(ChangeMaterial));
			childscript.changeMaterial(m);
			}
		}
		
	}
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        //GetComponent<Renderer>().material.SetMatrix("_rotmat",Matrix4x4.Rotate(Quaternion.Inverse(GetComponent<Transform>().parent.rotation)));
    }
}
