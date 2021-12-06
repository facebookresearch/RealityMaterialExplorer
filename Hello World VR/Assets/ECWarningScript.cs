using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ECWarningScript : MonoBehaviour
{	
	private float difval;
	private float specval;
    // Start is called before the first frame update
    void Start()
    {
        difval = 0.1f;
		specval = 0.88f;
		gameObject.SetActive(false);
    }
	
	public void updateDifVal(float d){
		difval = d;
		updateECWarning();
	}
	
	public void updateSpecVal(float s){
		specval = s;
		updateECWarning();
	}
	
	private void updateECWarning(){
		bool shouldWarn = (difval+specval>1.0f);
		gameObject.SetActive(shouldWarn);
	}
    // Update is called once per frame
    void Update()
    {
        
    }
}
