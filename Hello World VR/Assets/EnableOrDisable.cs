using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EnableOrDisable : MonoBehaviour
{
	public GameObject thisobject;
    // Start is called before the first frame update
    void Start()
    {
        
    }
	
	public void enableObject(){
		thisobject.SetActive(true);
	}
	
	public void disableObject(){
		thisobject.SetActive(false);
	}

    // Update is called once per frame
    void Update()
    {
        
    }
}
